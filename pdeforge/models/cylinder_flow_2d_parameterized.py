"""
2D flow around a cylinder with parameterized cylinder position (FEniCSx-based).

This model extends cylinder_flow_2d to allow the cylinder position (cx, cy)
to be varied as input parameters for data generation, enabling learning of
flow patterns for different cylinder positions.

Equations:
    rho(u.grad)u - mu*laplacian(u) + grad(p) = f
    div(u) = 0

BCs: parabolic inlet, zero-stress outlet, no-slip walls and cylinder.

Operator learning task: (inlet_velocity_scale, cx, cy) -> (u, v, p)
"""

import numpy as np
from typing import Dict, Tuple, Optional

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
    @register_model("cylinder_flow_2d_parameterized")
    class CylinderFlow2DParameterized(FEniCSModel):
        """
        2D cylinder flow with parameterized cylinder position.

        The cylinder center (cx, cy) can be varied as part of the input,
        allowing the model to generate datasets where the cylinder position
        varies across samples.

        Parameters
        ----------
        resolution : Dict[str, int]
            Output grid resolution, e.g., {"x": 110, "y": 41}
        inlet_velocity : float
            Mean inlet velocity (default: 0.3)
        viscosity : float
            Dynamic viscosity μ (default: 0.001)
        cylinder_radius : float
            Cylinder radius (default: 0.05)
        cx_range : Tuple[float, float]
            Range for cylinder x-position (default: (0.15, 0.5))
        cy_range : Tuple[float, float]
            Range for cylinder y-position (default: (0.15, 0.26))

        Examples
        --------
        >>> dataset = generate_dataset(
        ...     model="cylinder_flow_2d_parameterized",
        ...     n_samples=10,
        ...     resolution={"x": 110, "y": 41},
        ...     params={"cx_range": (0.2, 0.4), "cy_range": (0.15, 0.25)},
        ... )
        """

        NDIM = 2
        BACKEND = "fenicsx"
        INPUT_NAMES = ["inlet_velocity_scale", "cylinder_x", "cylinder_y"]
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
            ParamSpec(
                name="cx_range",
                description="Range for cylinder x-position [min, max]",
                default=(0.15, 0.5),
                param_type=ParamType.GEOMETRY,
            ),
            ParamSpec(
                name="cy_range",
                description="Range for cylinder y-position [min, max]",
                default=(0.15, 0.26),
                param_type=ParamType.GEOMETRY,
            ),
        ]

        DEFAULT_PARAMS = {
            "viscosity": 0.001,
            "inlet_velocity": 0.3,
            "cylinder_radius": 0.05,
            "cx_range": (0.15, 0.5),
            "cy_range": (0.15, 0.26),
            "density": 1.0,
            "_use_stokes": False,
            "_mesh_resolution": 0.02,
            "_channel_length": 2.2,
            "_channel_height": 0.41,
        }

        def __init__(self, resolution, domain=None, **params):
            # set domain from channel geometry if not provided
            if domain is None:
                L = params.get("_channel_length", self.DEFAULT_PARAMS["_channel_length"])
                H = params.get("_channel_height", self.DEFAULT_PARAMS["_channel_height"])
                domain = {"x": (0.0, L), "y": (0.0, H)}

            # store geometry params BEFORE super().__init__ calls create_mesh()
            merged_params = {**self.DEFAULT_PARAMS, **params}
            self.r = merged_params["cylinder_radius"]
            self.mu = merged_params["viscosity"]
            self.rho = merged_params.get("density", 1.0)
            self.U_mean = merged_params["inlet_velocity"]
            self.cx_range = merged_params.get("cx_range", (0.15, 0.5))
            self.cy_range = merged_params.get("cy_range", (0.15, 0.26))

            # Default cylinder position for initial mesh
            self.cx = (self.cx_range[0] + self.cx_range[1]) / 2
            self.cy = (self.cy_range[0] + self.cy_range[1]) / 2

            super().__init__(resolution, domain, **params)

            self.L = self.domain.bounds["x"][1]
            self.H = self.domain.bounds["y"][1]

        def create_mesh(self):
            """Create mesh with cylinder at current position."""
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

        def _create_mesh_and_spaces(self, cx, cy):
            """Recreate mesh and function spaces for a new cylinder position."""
            self.cx = cx
            self.cy = cy

            # Create new mesh with updated cylinder position
            self.mesh = self.create_mesh()
            self.comm = self.mesh.comm

            # Recreate function spaces
            self.create_function_spaces()

            # Recreate output grid interpolation points
            self._setup_output_grid()

        def solve(self, inlet_scale=1.0, cx=None, cy=None, return_functions=False):
            """
            Solve flow problem with specified cylinder position.

            Parameters
            ----------
            inlet_scale : float
                Scaling for inlet velocity
            cx : float, optional
                Cylinder x-position. If None, uses default.
            cy : float, optional
                Cylinder y-position. If None, uses default.
            return_functions : bool
                If True return FEM functions instead of arrays
            """
            if cx is None:
                cx = self.cx
            if cy is None:
                cy = self.cy

            # Recreate mesh for new cylinder position
            self._create_mesh_and_spaces(cx, cy)

            U_max = 1.5 * self.U_mean * inlet_scale

            def inlet_velocity(x):
                values = np.zeros((2, x.shape[1]))
                values[0] = 4 * U_max * x[1] * (self.H - x[1]) / (self.H ** 2)
                return values

            def no_slip(x):
                return np.zeros((2, x.shape[1]))

            facet_tags = self.mesh.facet_tags
            W = self.W
            V, _ = W.sub(0).collapse()

            # inlet BC
            inlet_bc_func = fem.Function(V)
            inlet_bc_func.interpolate(inlet_velocity)

            inlet_facets = facet_tags.find(1)
            inlet_dofs = fem.locate_dofs_topological(
                (W.sub(0), V),
                self.mesh.topology.dim - 1,
                inlet_facets
            )
            bc_inlet = fem.dirichletbc(inlet_bc_func, inlet_dofs, W.sub(0))

            # no-slip for walls (tag 3) and cylinder (tag 4)
            noslip_func = fem.Function(V)
            noslip_func.interpolate(no_slip)

            wall_facets = facet_tags.find(3)
            wall_dofs = fem.locate_dofs_topological(
                (W.sub(0), V),
                self.mesh.topology.dim - 1,
                wall_facets
            )
            bc_walls = fem.dirichletbc(noslip_func, wall_dofs, W.sub(0))

            cylinder_facets = facet_tags.find(4)
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
            """Generate random inlet velocity scale and cylinder position."""
            if generator_params is None:
                generator_params = {}

            if seed is not None:
                np.random.seed(seed)

            scale_min = generator_params.get("scale_min", 0.5)
            scale_max = generator_params.get("scale_max", 2.0)

            scale = np.random.uniform(scale_min, scale_max)
            cx = np.random.uniform(self.cx_range[0], self.cx_range[1])
            cy = np.random.uniform(self.cy_range[0], self.cy_range[1])

            return np.array([scale, cx, cy])

        def generate_sample(self, generator="default", generator_params=None,
                           seed=None, validate=True, max_attempts=10):
            """Generate single (input, output) sample."""
            if generator_params is None:
                generator_params = {}

            for attempt in range(max_attempts):
                current_seed = seed + attempt if seed is not None else None

                inputs = self.generate_ic(
                    generator=generator,
                    generator_params=generator_params,
                    seed=current_seed,
                )

                inlet_scale = inputs[0]
                cx = inputs[1]
                cy = inputs[2]

                try:
                    solution = self.solve(inlet_scale=inlet_scale, cx=cx, cy=cy)

                    if validate:
                        validation = self.validate_solution(inputs, solution)
                        if validation['valid']:
                            return inputs, solution, validation
                    else:
                        return inputs, solution, {'valid': True}

                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    continue

            raise RuntimeError(f"Failed to generate valid sample after {max_attempts} attempts")

        def validate_solution(self, inputs, solution, tol=1e-6):
            is_valid = (
                not np.isnan(solution).any() and
                not np.isinf(solution).any()
            )

            return {
                'valid': is_valid,
                'max_velocity': np.sqrt(solution[:,:,0]**2 + solution[:,:,1]**2).max(),
                'cylinder_position': (inputs[1], inputs[2]),
            }
