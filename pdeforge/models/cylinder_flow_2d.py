"""
2D flow around a cylinder (FEniCSx-based).

Steady Navier-Stokes for flow around circular cylinder.
For low Re we solve steady NS, higher Re needs time-dependent solver.

Equations:
    rho(u.grad)u - mu*laplacian(u) + grad(p) = f
    div(u) = 0

BCs: parabolic inlet, zero-stress outlet, no-slip walls and cylinder.

Operator learning task: inlet_velocity_scale -> (u, v, p)
"""

import numpy as np
from typing import Dict, Tuple

from pdeforge.core.registry import register_model
from pdeforge.core.params import ParamSpec, ParamType

# check FEniCSx
try:
    import dolfinx
    from dolfinx import fem, mesh as dfx_mesh, io
    from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
    import ufl
    from ufl import inner, grad, div, dx, ds, TrialFunction, TestFunction, split, FacetNormal
    from mpi4py import MPI
    from petsc4py import PETSc
    import basix

    from pdeforge.core.fenics_base import FEniCSModel
    from pdeforge.solvers.fenics_utils import create_rectangle_with_hole

    HAS_FENICSX = True
except ImportError:
    HAS_FENICSX = False

    class FEniCSModel:
        pass


if HAS_FENICSX:
    @register_model("cylinder_flow_2d")
    class CylinderFlow2D(FEniCSModel):
        """
        2D cylinder flow using FEniCSx. Solves steady Navier-Stokes
        (or Stokes for low Re).
        """

        NDIM = 2
        BACKEND = "fenicsx"
        INPUT_NAMES = ["inlet_velocity_scale"]
        OUTPUT_NAMES = ["u", "v", "p"]

        USER_PARAMS = [
            ParamSpec(
                name="inlet_velocity",
                description="Mean inlet velocity",
                default=0.3,
                param_type=ParamType.PHYSICAL,
                bounds=(0.01, 2.0),
            ),
            ParamSpec(
                name="viscosity",
                description="Dynamic viscosity",
                default=0.001,
                param_type=ParamType.PHYSICAL,
                bounds=(1e-5, 0.1),
            ),
            ParamSpec(
                name="cylinder_radius",
                description="Cylinder radius",
                default=0.05,
                param_type=ParamType.GEOMETRY,
                bounds=(0.01, 0.1),
            ),
        ]

        DEFAULT_PARAMS = {
            "viscosity": 0.001,
            "inlet_velocity": 0.3,
            "cylinder_radius": 0.05,
            "cylinder_center": (0.2, 0.2),
            "density": 1.0,
            "_use_stokes": False,
            "_mesh_resolution": 0.02,
            "_channel_length": 2.2,
            "_channel_height": 0.41,
        }

        from pdeforge.core.params import ParamSpec, ParamType

        def __init__(self, resolution, domain=None, **params):
            # set domain from channel geometry if not provided
            if domain is None:
                L = params.get("_channel_length", self.DEFAULT_PARAMS["_channel_length"])
                H = params.get("_channel_height", self.DEFAULT_PARAMS["_channel_height"])
                domain = {"x": (0.0, L), "y": (0.0, H)}

            # store geometry params BEFORE super().__init__ calls create_mesh()
            merged_params = {**self.DEFAULT_PARAMS, **params}
            self.cx, self.cy = merged_params.get("cylinder_center", (0.2, 0.2))
            self.r = merged_params["cylinder_radius"]
            self.mu = merged_params["viscosity"]
            self.rho = merged_params.get("density", 1.0)
            self.U_mean = merged_params["inlet_velocity"]

            super().__init__(resolution, domain, **params)

            self.L = self.domain.bounds["x"][1]
            self.H = self.domain.bounds["y"][1]

            self._setup_boundary_conditions()

        def create_mesh(self):
            """Create mesh with cylinder hole."""
            return create_rectangle_with_hole(
                L=self.params.get("_channel_length", 2.2),
                H=self.params.get("_channel_height", 0.41),
                cx=self.cx,
                cy=self.cy,
                r=self.r,
                resolution=self.params.get("_mesh_resolution", 0.02),
                comm=MPI.COMM_WORLD,
            )

        def create_function_spaces(self):
            """Taylor-Hood elements (P2-P1) for velocity-pressure."""
            P2 = basix.ufl.element("Lagrange", self.mesh.basix_cell(), 2, shape=(2,))
            P1 = basix.ufl.element("Lagrange", self.mesh.basix_cell(), 1)

            TH = basix.ufl.mixed_element([P2, P1])
            self.W = fem.functionspace(self.mesh, TH)

            self.V = fem.functionspace(
                self.mesh,
                basix.ufl.element("Lagrange", self.mesh.basix_cell(), 2, shape=(2,))
            )
            self.Q = fem.functionspace(
                self.mesh,
                basix.ufl.element("Lagrange", self.mesh.basix_cell(), 1)
            )

        def _setup_boundary_conditions(self):
            """Setup BCs for flow problem."""
            facet_tags = self.mesh.facet_tags

            # parabolic inlet: u(y) = 4*U_max*y*(H-y)/H^2
            def inlet_velocity(x):
                U_max = 1.5 * self.U_mean
                values = np.zeros((2, x.shape[1]))
                values[0] = 4 * U_max * x[1] * (self.H - x[1]) / (self.H ** 2)
                return values

            self._inlet_velocity_func = inlet_velocity

            def no_slip(x):
                return np.zeros((2, x.shape[1]))

            self._no_slip_func = no_slip
            self._facet_tags = facet_tags

        def solve(self, inlet_scale=1.0, return_functions=False):
            """
            Solve flow problem.

            inlet_scale: scaling for inlet velocity
            return_functions: if True return FEM functions instead of arrays
            """
            U_max = 1.5 * self.U_mean * inlet_scale

            def inlet_velocity(x):
                values = np.zeros((2, x.shape[1]))
                values[0] = 4 * U_max * x[1] * (self.H - x[1]) / (self.H ** 2)
                return values

            W = self.W
            V, _ = W.sub(0).collapse()

            # inlet BC
            inlet_bc_func = fem.Function(V)
            inlet_bc_func.interpolate(inlet_velocity)

            inlet_facets = self._facet_tags.find(1)
            inlet_dofs = fem.locate_dofs_topological(
                (W.sub(0), V),
                self.mesh.topology.dim - 1,
                inlet_facets
            )
            bc_inlet = fem.dirichletbc(inlet_bc_func, inlet_dofs, W.sub(0))

            # no-slip for walls (tag 3) and cylinder (tag 4)
            noslip_func = fem.Function(V)
            noslip_func.interpolate(self._no_slip_func)

            wall_facets = self._facet_tags.find(3)
            wall_dofs = fem.locate_dofs_topological(
                (W.sub(0), V),
                self.mesh.topology.dim - 1,
                wall_facets
            )
            bc_walls = fem.dirichletbc(noslip_func, wall_dofs, W.sub(0))

            cylinder_facets = self._facet_tags.find(4)
            cylinder_dofs = fem.locate_dofs_topological(
                (W.sub(0), V),
                self.mesh.topology.dim - 1,
                cylinder_facets
            )
            bc_cylinder = fem.dirichletbc(noslip_func, cylinder_dofs, W.sub(0))

            bcs = [bc_inlet, bc_walls, bc_cylinder]

            # variational problem
            w = fem.Function(W)
            (u, p) = ufl.split(w)
            (v, q) = ufl.TestFunctions(W)

            if self.params.get("_use_stokes", False):
                # Stokes (low Re limit)
                # F = self.mu * inner(grad(u), grad(v)) * dx
                F = (
                    self.mu * inner(grad(u), grad(v)) * dx
                    - p * div(v) * dx
                    - q * div(u) * dx
                )
            else:
                # Navier-Stokes
                F = (
                    self.rho * inner(grad(u) * u, v) * dx
                    + self.mu * inner(grad(u), grad(v)) * dx
                    - p * div(v) * dx
                    - q * div(u) * dx
                )

            # solve
            if self.params.get("_use_stokes", False):
                u_trial = TrialFunction(W)
                (u_t, p_t) = ufl.split(u_trial)

                a = (
                    self.mu * inner(grad(u_t), grad(v)) * dx
                    - p_t * div(v) * dx
                    - q * div(u_t) * dx
                )
                L = fem.Constant(self.mesh, PETSc.ScalarType(0.0)) * q * dx

                problem = LinearProblem(a, L, bcs=bcs, petsc_options={
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps",
                })
                w = problem.solve()
            else:
                # dolfinx 0.10+ API uses NonlinearProblem with built-in SNES solver
                problem = NonlinearProblem(
                    F, w, bcs=bcs,
                    petsc_options_prefix="navier_stokes_",
                    petsc_options={
                        "snes_type": "newtonls",
                        "snes_linesearch_type": "bt",
                        "snes_rtol": 1e-6,
                        "snes_max_it": 50,
                        "ksp_type": "preonly",
                        "pc_type": "lu",
                        "pc_factor_mat_solver_type": "mumps",
                    }
                )
                w = problem.solve()
                if problem.solver.getConvergedReason() <= 0:
                    raise RuntimeError("Newton solver did not converge")

            if return_functions:
                return w

            # extract and interpolate to grid
            u_sol = w.sub(0).collapse()
            p_sol = w.sub(1).collapse()

            u_grid = self.interpolate_to_grid(u_sol, fill_value=0.0)
            p_grid = self.interpolate_to_grid(p_sol, fill_value=0.0)

            # u_grid is (nx, ny, 2), split into u, v
            u_component = u_grid[:, :, 0]
            v_component = u_grid[:, :, 1]

            solution = np.stack([u_component, v_component, p_grid], axis=-1)

            return solution

        def generate_ic(self, generator="default", generator_params=None, seed=None):
            """Generate random inlet velocity scale."""
            if generator_params is None:
                generator_params = {}

            if seed is not None:
                np.random.seed(seed)

            scale_min = generator_params.get("scale_min", 0.5)
            scale_max = generator_params.get("scale_max", 2.0)

            scale = np.random.uniform(scale_min, scale_max)

            return np.array([scale])

        def generate_sample(self, generator="default", generator_params=None,
                           seed=None, validate=True, max_attempts=10):
            """Generate single (input, output) sample."""
            if generator_params is None:
                generator_params = {}

            for attempt in range(max_attempts):
                current_seed = seed + attempt if seed is not None else None

                inlet_scale = self.generate_ic(
                    generator=generator,
                    generator_params=generator_params,
                    seed=current_seed,
                )

                try:
                    solution = self.solve(inlet_scale=float(inlet_scale[0]))

                    if validate:
                        validation = self.validate_solution(inlet_scale, solution)
                        if validation['valid']:
                            return inlet_scale, solution, validation
                    else:
                        return inlet_scale, solution, {'valid': True}

                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    continue

            raise RuntimeError(f"Failed to generate valid sample after {max_attempts} attempts")

        def validate_solution(self, inlet_scale, solution, tol=1e-6):
            is_valid = (
                not np.isnan(solution).any() and
                not np.isinf(solution).any()
            )

            return {
                'valid': is_valid,
                'max_velocity': np.sqrt(solution[:,:,0]**2 + solution[:,:,1]**2).max(),
            }
