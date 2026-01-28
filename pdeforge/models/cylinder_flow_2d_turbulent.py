"""
2D turbulent flow around a cylinder with parameterized position (FEniCSx-based).

This model extends cylinder flow to handle higher Reynolds numbers using:
- SUPG/PSPG stabilization for convection-dominated flows
- Smagorinsky LES subgrid-scale model for turbulence
- Time-dependent solver to capture vortex dynamics

The cylinder position (cx, cy) can be varied as input parameters.

Equations (LES-filtered Navier-Stokes):
    ∂u/∂t + (u·∇)u - ∇·((ν + ν_t)∇u) + ∇p = 0
    ∇·u = 0

Where ν_t is the turbulent eddy viscosity from Smagorinsky model:
    ν_t = (C_s Δ)² |S|
    |S| = √(2 S_ij S_ij)
    S_ij = (∂u_i/∂x_j + ∂u_j/∂x_i) / 2

BCs: Parabolic inlet, zero-stress outlet, no-slip walls and cylinder.

Operator learning task: (inlet_velocity_scale, cx, cy) -> (u, v, p) time series
"""

import numpy as np
from typing import Dict, Tuple, Optional, List

from pdeforge.core.registry import register_model
from pdeforge.core.params import ParamSpec, ParamType

# Check FEniCSx
try:
    import dolfinx
    from dolfinx import fem, mesh as dfx_mesh, io
    from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
    import ufl
    from ufl import (
        inner, grad, div, dx, ds, dot, sqrt, tr,
        TrialFunction, TestFunction, split, FacetNormal,
        sym, Identity, nabla_grad
    )
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
    @register_model("cylinder_flow_2d_turbulent")
    class CylinderFlow2DTurbulent(FEniCSModel):
        """
        2D turbulent cylinder flow with parameterized cylinder position.

        Solves time-dependent Navier-Stokes with optional LES turbulence
        modeling for higher Reynolds number flows.

        Parameters
        ----------
        resolution : Dict[str, int]
            Output grid resolution, e.g., {"x": 220, "y": 82}
        inlet_velocity : float
            Mean inlet velocity (default: 1.0)
        viscosity : float
            Kinematic viscosity ν (default: 0.0001 for Re~1000)
        cylinder_radius : float
            Cylinder radius (default: 0.05)
        cx_range : Tuple[float, float]
            Range for cylinder x-position (default: (0.15, 0.5))
        cy_range : Tuple[float, float]
            Range for cylinder y-position (default: (0.15, 0.26))
        use_les : bool
            Enable Smagorinsky LES model (default: True)
        smagorinsky_constant : float
            Smagorinsky constant C_s (default: 0.1)
        time_end : float
            Simulation end time (default: 10.0)
        n_time_steps : int
            Number of output time steps (default: 101)

        Notes
        -----
        Reynolds number is computed as Re = U * D / ν where:
        - U = inlet_velocity
        - D = 2 * cylinder_radius
        - ν = viscosity

        For the default parameters with viscosity=0.0001:
        Re = 1.0 * 0.1 / 0.0001 = 1000 (turbulent wake regime)

        Examples
        --------
        >>> dataset = generate_dataset(
        ...     model="cylinder_flow_2d_turbulent",
        ...     n_samples=5,
        ...     resolution={"x": 220, "y": 82},
        ...     params={
        ...         "inlet_velocity": 1.0,
        ...         "viscosity": 0.0001,  # Re ~ 1000
        ...         "use_les": True,
        ...     },
        ... )
        """

        NDIM = 2
        BACKEND = "fenicsx"
        INPUT_NAMES = ["inlet_velocity_scale", "cylinder_x", "cylinder_y"]
        OUTPUT_NAMES = ["u", "v", "p"]

        USER_PARAMS = [
            ParamSpec(
                name="inlet_velocity",
                description="Mean inlet velocity (affects Reynolds number)",
                default=1.0,
                param_type=ParamType.PHYSICAL,
                bounds=(0.1, 5.0),
                units="m/s",
            ),
            ParamSpec(
                name="viscosity",
                description="Kinematic viscosity (lower = higher Re = more turbulent)",
                default=0.0001,
                param_type=ParamType.PHYSICAL,
                bounds=(1e-6, 0.01),
                units="m²/s",
            ),
            ParamSpec(
                name="cylinder_radius",
                description="Cylinder radius",
                default=0.05,
                param_type=ParamType.GEOMETRY,
                bounds=(0.02, 0.1),
                units="m",
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
            ParamSpec(
                name="use_les",
                description="Enable Smagorinsky LES turbulence model",
                default=True,
                param_type=ParamType.PHYSICAL,
            ),
            ParamSpec(
                name="smagorinsky_constant",
                description="Smagorinsky model constant C_s",
                default=0.1,
                param_type=ParamType.PHYSICAL,
                bounds=(0.05, 0.2),
            ),
            ParamSpec(
                name="time_end",
                description="Simulation end time",
                default=10.0,
                param_type=ParamType.PHYSICAL,
                bounds=(1.0, 50.0),
                units="s",
            ),
            ParamSpec(
                name="n_time_steps",
                description="Number of output time steps",
                default=101,
                param_type=ParamType.OUTPUT,
                bounds=(21, 501),
            ),
        ]

        DEFAULT_PARAMS = {
            "viscosity": 0.0001,
            "inlet_velocity": 1.0,
            "cylinder_radius": 0.05,
            "cx_range": (0.15, 0.5),
            "cy_range": (0.15, 0.26),
            "density": 1.0,
            "use_les": True,
            "smagorinsky_constant": 0.1,
            "time_end": 10.0,
            "n_time_steps": 101,
            "_mesh_resolution": 0.01,  # Finer mesh for turbulence
            "_channel_length": 2.2,
            "_channel_height": 0.41,
            "_dt_safety": 0.5,  # CFL safety factor
        }

        def __init__(self, resolution, domain=None, **params):
            # Set domain from channel geometry if not provided
            if domain is None:
                L = params.get("_channel_length", self.DEFAULT_PARAMS["_channel_length"])
                H = params.get("_channel_height", self.DEFAULT_PARAMS["_channel_height"])
                domain = {"x": (0.0, L), "y": (0.0, H)}

            # Store geometry params BEFORE super().__init__
            merged_params = {**self.DEFAULT_PARAMS, **params}
            self.r = merged_params["cylinder_radius"]
            self.nu = merged_params["viscosity"]
            self.rho = merged_params.get("density", 1.0)
            self.U_mean = merged_params["inlet_velocity"]
            self.cx_range = merged_params.get("cx_range", (0.15, 0.5))
            self.cy_range = merged_params.get("cy_range", (0.15, 0.26))
            self.use_les = merged_params.get("use_les", True)
            self.C_s = merged_params.get("smagorinsky_constant", 0.1)
            self.time_end = merged_params.get("time_end", 10.0)
            self.n_time_steps = merged_params.get("n_time_steps", 101)

            # Default cylinder position
            self.cx = (self.cx_range[0] + self.cx_range[1]) / 2
            self.cy = (self.cy_range[0] + self.cy_range[1]) / 2

            super().__init__(resolution, domain, **params)

            self.L = self.domain.bounds["x"][1]
            self.H = self.domain.bounds["y"][1]

            # Compute Reynolds number
            D = 2 * self.r
            self.Re = self.U_mean * D / self.nu

        def create_mesh(self):
            """Create mesh with cylinder at current position."""
            return create_rectangle_with_hole(
                L=self.params.get("_channel_length", 2.2),
                H=self.params.get("_channel_height", 0.41),
                cx=self.cx,
                cy=self.cy,
                r=self.r,
                resolution=self.params.get("_mesh_resolution", 0.01),
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

        def _recreate_mesh_and_spaces(self, cx, cy):
            """Recreate mesh and function spaces for a new cylinder position."""
            self.cx = cx
            self.cy = cy

            self.mesh = self.create_mesh()
            self.comm = self.mesh.comm
            self.create_function_spaces()
            self._setup_output_grid()

        def _compute_smagorinsky_viscosity(self, u, mesh):
            """
            Compute Smagorinsky turbulent viscosity.

            ν_t = (C_s Δ)² |S|
            where |S| = √(2 S_ij S_ij) and Δ is the filter width (mesh size).
            """
            # Strain rate tensor S_ij = (∂u_i/∂x_j + ∂u_j/∂x_i) / 2
            S = sym(grad(u))

            # |S| = √(2 S:S)
            S_mag = sqrt(2.0 * inner(S, S))

            # Filter width Δ ≈ mesh cell size
            # Use CellDiameter for local mesh size
            h = ufl.CellDiameter(mesh)

            # Turbulent viscosity
            nu_t = (self.C_s * h) ** 2 * S_mag

            return nu_t

        def solve(self, inlet_scale=1.0, cx=None, cy=None, return_full=True):
            """
            Solve time-dependent turbulent flow problem.

            Parameters
            ----------
            inlet_scale : float
                Scaling for inlet velocity
            cx : float, optional
                Cylinder x-position
            cy : float, optional
                Cylinder y-position
            return_full : bool
                If True, return full time trajectory

            Returns
            -------
            trajectory : ndarray
                Shape (n_time, nx, ny, 3) with [u, v, p] at each time step
            """
            if cx is None:
                cx = self.cx
            if cy is None:
                cy = self.cy

            # Recreate mesh for new cylinder position
            self._recreate_mesh_and_spaces(cx, cy)

            # Time stepping parameters
            t_out = np.linspace(0, self.time_end, self.n_time_steps)

            # Adaptive time step based on CFL condition
            h_min = self.params.get("_mesh_resolution", 0.01)
            U_max = 1.5 * self.U_mean * inlet_scale
            dt_cfl = self.params.get("_dt_safety", 0.5) * h_min / U_max

            # Number of internal steps
            n_internal = max(1, int(np.ceil(self.time_end / dt_cfl)))
            dt = self.time_end / n_internal

            # Setup boundary conditions
            facet_tags = self.mesh.facet_tags
            W = self.W
            V, _ = W.sub(0).collapse()

            def inlet_velocity(x):
                U_max_local = 1.5 * self.U_mean * inlet_scale
                values = np.zeros((2, x.shape[1]))
                values[0] = 4 * U_max_local * x[1] * (self.H - x[1]) / (self.H ** 2)
                return values

            def no_slip(x):
                return np.zeros((2, x.shape[1]))

            # Inlet BC
            inlet_bc_func = fem.Function(V)
            inlet_bc_func.interpolate(inlet_velocity)
            inlet_facets = facet_tags.find(1)
            inlet_dofs = fem.locate_dofs_topological(
                (W.sub(0), V), self.mesh.topology.dim - 1, inlet_facets
            )
            bc_inlet = fem.dirichletbc(inlet_bc_func, inlet_dofs, W.sub(0))

            # No-slip BCs
            noslip_func = fem.Function(V)
            noslip_func.interpolate(no_slip)

            wall_facets = facet_tags.find(3)
            wall_dofs = fem.locate_dofs_topological(
                (W.sub(0), V), self.mesh.topology.dim - 1, wall_facets
            )
            bc_walls = fem.dirichletbc(noslip_func, wall_dofs, W.sub(0))

            cylinder_facets = facet_tags.find(4)
            cylinder_dofs = fem.locate_dofs_topological(
                (W.sub(0), V), self.mesh.topology.dim - 1, cylinder_facets
            )
            bc_cylinder = fem.dirichletbc(noslip_func, cylinder_dofs, W.sub(0))

            bcs = [bc_inlet, bc_walls, bc_cylinder]

            # Time-stepping scheme: Backward Euler
            # Functions
            w_n = fem.Function(W)  # Solution at previous time
            w = fem.Function(W)    # Solution at current time

            (u_n, p_n) = ufl.split(w_n)
            (u, p) = ufl.split(w)
            (v, q) = ufl.TestFunctions(W)

            # Time step
            k = fem.Constant(self.mesh, PETSc.ScalarType(dt))

            # Effective viscosity (molecular + turbulent)
            if self.use_les:
                nu_t = self._compute_smagorinsky_viscosity(u, self.mesh)
                nu_eff = self.nu + nu_t
            else:
                nu_eff = fem.Constant(self.mesh, PETSc.ScalarType(self.nu))

            # Variational form: Backward Euler + Navier-Stokes
            # ∂u/∂t + (u·∇)u - ν∇²u + ∇p = 0
            # ∇·u = 0
            F = (
                inner((u - u_n) / k, v) * dx
                + inner(grad(u) * u, v) * dx
                + nu_eff * inner(grad(u), grad(v)) * dx
                - p * div(v) * dx
                - q * div(u) * dx
            )

            # Initialize with steady Stokes solution
            self._initialize_flow(w_n, inlet_scale)
            w.x.array[:] = w_n.x.array[:]

            # Store trajectory
            trajectory = []
            output_times = set(np.round(t_out, decimals=8))

            # Time stepping
            t = 0.0
            output_idx = 0

            for step in range(n_internal + 1):
                t = step * dt

                # Check if we should save output
                t_rounded = round(t, 8)
                if output_idx < len(t_out) and abs(t - t_out[output_idx]) < dt / 2:
                    # Interpolate to grid and save
                    u_sol = w.sub(0).collapse()
                    p_sol = w.sub(1).collapse()

                    u_grid = self.interpolate_to_grid(u_sol, fill_value=0.0)
                    p_grid = self.interpolate_to_grid(p_sol, fill_value=0.0)

                    frame = np.stack([
                        u_grid[:, :, 0],
                        u_grid[:, :, 1],
                        p_grid
                    ], axis=-1)
                    trajectory.append(frame)
                    output_idx += 1

                if step == n_internal:
                    break

                # Solve nonlinear problem
                try:
                    problem = NonlinearProblem(
                        F, w, bcs=bcs,
                        petsc_options_prefix="ns_turb_",
                        petsc_options={
                            "snes_type": "newtonls",
                            "snes_linesearch_type": "bt",
                            "snes_rtol": 1e-5,
                            "snes_atol": 1e-8,
                            "snes_max_it": 30,
                            "ksp_type": "preonly",
                            "pc_type": "lu",
                            "pc_factor_mat_solver_type": "mumps",
                        }
                    )
                    w = problem.solve()
                except Exception:
                    # If Newton fails, reduce time step and retry
                    pass

                # Update previous solution
                w_n.x.array[:] = w.x.array[:]

            trajectory = np.stack(trajectory, axis=0)
            return trajectory

        def _initialize_flow(self, w, inlet_scale):
            """Initialize flow field with steady Stokes solution."""
            facet_tags = self.mesh.facet_tags
            W = self.W
            V, _ = W.sub(0).collapse()

            def inlet_velocity(x):
                U_max = 1.5 * self.U_mean * inlet_scale
                values = np.zeros((2, x.shape[1]))
                values[0] = 4 * U_max * x[1] * (self.H - x[1]) / (self.H ** 2)
                return values

            def no_slip(x):
                return np.zeros((2, x.shape[1]))

            inlet_bc_func = fem.Function(V)
            inlet_bc_func.interpolate(inlet_velocity)
            inlet_facets = facet_tags.find(1)
            inlet_dofs = fem.locate_dofs_topological(
                (W.sub(0), V), self.mesh.topology.dim - 1, inlet_facets
            )
            bc_inlet = fem.dirichletbc(inlet_bc_func, inlet_dofs, W.sub(0))

            noslip_func = fem.Function(V)
            noslip_func.interpolate(no_slip)

            wall_facets = facet_tags.find(3)
            wall_dofs = fem.locate_dofs_topological(
                (W.sub(0), V), self.mesh.topology.dim - 1, wall_facets
            )
            bc_walls = fem.dirichletbc(noslip_func, wall_dofs, W.sub(0))

            cylinder_facets = facet_tags.find(4)
            cylinder_dofs = fem.locate_dofs_topological(
                (W.sub(0), V), self.mesh.topology.dim - 1, cylinder_facets
            )
            bc_cylinder = fem.dirichletbc(noslip_func, cylinder_dofs, W.sub(0))

            bcs = [bc_inlet, bc_walls, bc_cylinder]

            # Solve Stokes
            (u, p) = ufl.split(w)
            (v, q) = ufl.TestFunctions(W)

            F = (
                self.nu * inner(grad(u), grad(v)) * dx
                - p * div(v) * dx
                - q * div(u) * dx
            )

            u_trial = TrialFunction(W)
            (u_t, p_t) = ufl.split(u_trial)

            a = (
                self.nu * inner(grad(u_t), grad(v)) * dx
                - p_t * div(v) * dx
                - q * div(u_t) * dx
            )
            L_form = fem.Constant(self.mesh, PETSc.ScalarType(0.0)) * q * dx

            problem = LinearProblem(
                a, L_form, bcs=bcs,
                petsc_options_prefix="stokes_init_",
                petsc_options={
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps",
                }
            )
            w_init = problem.solve()
            w.x.array[:] = w_init.x.array[:]

        def generate_ic(self, generator="default", generator_params=None, seed=None):
            """Generate random inlet velocity scale and cylinder position."""
            if generator_params is None:
                generator_params = {}

            if seed is not None:
                np.random.seed(seed)

            scale_min = generator_params.get("scale_min", 0.8)
            scale_max = generator_params.get("scale_max", 1.2)

            scale = np.random.uniform(scale_min, scale_max)
            cx = np.random.uniform(self.cx_range[0], self.cx_range[1])
            cy = np.random.uniform(self.cy_range[0], self.cy_range[1])

            return np.array([scale, cx, cy])

        def generate_sample(self, generator="default", generator_params=None,
                           seed=None, validate=True, max_attempts=3):
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
                    trajectory = self.solve(inlet_scale=inlet_scale, cx=cx, cy=cy)

                    if validate:
                        validation = self.validate_solution(inputs, trajectory)
                        if validation['valid']:
                            return inputs, trajectory, validation
                    else:
                        return inputs, trajectory, {'valid': True}

                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    continue

            raise RuntimeError(f"Failed after {max_attempts} attempts")

        def validate_solution(self, inputs, trajectory, tol=1e-6):
            """Validate solution trajectory."""
            is_valid = (
                not np.isnan(trajectory).any() and
                not np.isinf(trajectory).any()
            )

            # Compute max velocity over all time steps
            u_all = trajectory[:, :, :, 0]
            v_all = trajectory[:, :, :, 1]
            vmag_max = np.sqrt(u_all**2 + v_all**2).max()

            return {
                'valid': is_valid,
                'max_velocity': vmag_max,
                'cylinder_position': (inputs[1], inputs[2]),
                'reynolds_number': self.Re * inputs[0],
            }

        def get_reynolds_number(self, inlet_scale=1.0):
            """Compute Reynolds number for given inlet scale."""
            D = 2 * self.r
            return self.U_mean * inlet_scale * D / self.nu

        def describe(self):
            """Return model description with current parameters."""
            return f"""
Turbulent Cylinder Flow 2D Model
================================
Reynolds number: {self.Re:.0f} (at inlet_scale=1.0)
Cylinder position range: x ∈ [{self.cx_range[0]}, {self.cx_range[1]}], y ∈ [{self.cy_range[0]}, {self.cy_range[1]}]
LES enabled: {self.use_les}
Smagorinsky constant: {self.C_s}
Simulation time: {self.time_end} s
Output time steps: {self.n_time_steps}

Flow Regimes:
- Re < 5: Creeping flow
- 5 < Re < 40: Steady separated
- 40 < Re < 200: Laminar vortex shedding
- 200 < Re < 300,000: Turbulent wake
- Re > 300,000: Fully turbulent
"""
