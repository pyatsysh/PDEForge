"""
2D Unsteady Flow Around a Cylinder (FEniCSx-based)

This model solves the time-dependent Navier-Stokes equations for flow around
a circular cylinder, capturing vortex shedding (von Kármán vortex street).

Equations (unsteady Navier-Stokes):
    ρ(∂u/∂t + (u·∇)u) - μ∇²u + ∇p = 0
    ∇·u = 0

Boundary conditions:
    - Inlet: Parabolic velocity profile (with optional ramp-up)
    - Outlet: Zero-stress (do-nothing)
    - Walls: No-slip (u = 0)
    - Cylinder: No-slip (u = 0)

Time discretization:
    Backward Euler (implicit) for stability at moderate Reynolds numbers.

Operator Learning Task:
    (inlet_velocity, initial_state) → trajectory u(x, t) for t ∈ [0, T]
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union, Callable, Any, List

from pdeforge.core.registry import register_model

# Check if FEniCSx is available
try:
    import dolfinx
    from dolfinx import fem, mesh as dfx_mesh
    from dolfinx.fem.petsc import NonlinearProblem
    import ufl
    from ufl import inner, grad, div, dx, TrialFunction, TestFunction, split
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
    from pdeforge.core.params import ParamSpec, ParamType
    
    @register_model("cylinder_flow_2d_unsteady")
    class CylinderFlow2DUnsteady(FEniCSModel):
        """
        2D unsteady flow around a cylinder using FEniCSx.
        
        Solves time-dependent Navier-Stokes equations to capture vortex
        shedding dynamics behind a circular cylinder.
        
        Parameters
        ----------
        resolution : Dict[str, int]
            Output grid resolution, e.g., {"x": 128, "y": 64}
        inlet_velocity : float
            Mean inlet velocity (default: 1.0)
        viscosity : float
            Dynamic viscosity μ (default: 0.001)
        time_end : float
            Final simulation time (default: 8.0)
        n_time_steps : int
            Number of output time steps (default: 81)
        
        Examples
        --------
        >>> dataset = generate_dataset(
        ...     model="cylinder_flow_2d_unsteady",
        ...     n_samples=5,
        ...     resolution={"x": 110, "y": 41},
        ...     params={"inlet_velocity": 1.0, "time_end": 8.0},
        ... )
        """
        
        NDIM = 2
        BACKEND = "fenicsx"
        INPUT_NAMES = ["inlet_velocity_scale"]
        OUTPUT_NAMES = ["u", "v", "p"]
        
        USER_PARAMS = [
            ParamSpec(
                name="inlet_velocity",
                description="Mean inlet velocity (affects Reynolds number)",
                default=1.0,
                param_type=ParamType.PHYSICAL,
                bounds=(0.1, 3.0),
                units="m/s",
                affects="Higher velocity → more pronounced vortex shedding",
            ),
            ParamSpec(
                name="viscosity",
                description="Dynamic viscosity of the fluid",
                default=0.001,
                param_type=ParamType.PHYSICAL,
                bounds=(1e-5, 0.01),
                units="Pa·s",
                affects="Lower viscosity → higher Re → stronger vortices",
            ),
            ParamSpec(
                name="time_end",
                description="Final simulation time",
                default=8.0,
                param_type=ParamType.PHYSICAL,
                bounds=(1.0, 20.0),
                units="s",
                affects="Longer time → more vortex shedding cycles",
            ),
        ]
        
        DEFAULT_PARAMS = {
            # User-facing
            "viscosity": 0.001,
            "inlet_velocity": 1.0,
            "time_end": 8.0,
            "cylinder_radius": 0.05,
            "cylinder_center": (0.2, 0.2),
            # Internal
            "density": 1.0,
            "_n_time_steps": 81,
            "_dt": None,  # Computed from time_end and n_time_steps
            "_mesh_resolution": 0.02,
            "_channel_length": 2.2,
            "_channel_height": 0.41,
            "_ramp_time": 0.5,  # Time to ramp up inlet velocity
        }
        
        def __init__(
            self,
            resolution: Dict[str, int],
            domain: Dict[str, Tuple[float, float]] = None,
            **params
        ):
            # Set domain based on channel geometry
            if domain is None:
                L = params.get("_channel_length", self.DEFAULT_PARAMS["_channel_length"])
                H = params.get("_channel_height", self.DEFAULT_PARAMS["_channel_height"])
                domain = {"x": (0.0, L), "y": (0.0, H)}
            
            super().__init__(resolution, domain, **params)
            
            # Geometry
            self.L = self.domain.bounds["x"][1]
            self.H = self.domain.bounds["y"][1]
            self.cx, self.cy = self.params.get("cylinder_center", (0.2, 0.2))
            self.r = self.params.get("cylinder_radius", 0.05)
            
            # Physics
            self.mu = self.params["viscosity"]
            self.rho = self.params.get("density", 1.0)
            self.U_mean = self.params["inlet_velocity"]
            
            # Time
            self.T = self.params.get("time_end", 8.0)
            self.n_t = self.params.get("_n_time_steps", 81)
            self.dt = self.params.get("_dt") or (self.T / (self.n_t - 1))
            self.ramp_time = self.params.get("_ramp_time", 0.5)
            
            # Setup BCs
            self._setup_boundary_conditions()
        
        def create_mesh(self) -> "dolfinx.mesh.Mesh":
            """Create mesh with cylinder hole."""
            # Get geometry from params (create_mesh called before attributes set)
            cx, cy = self.params.get("cylinder_center", (0.2, 0.2))
            r = self.params.get("cylinder_radius", 0.05)
            return create_rectangle_with_hole(
                L=self.params.get("_channel_length", 2.2),
                H=self.params.get("_channel_height", 0.41),
                cx=cx,
                cy=cy,
                r=r,
                resolution=self.params.get("_mesh_resolution", 0.02),
                comm=MPI.COMM_WORLD,
            )
        
        def create_function_spaces(self) -> None:
            """Create Taylor-Hood elements (P2-P1)."""
            P2 = basix.ufl.element("Lagrange", self.mesh.basix_cell(), 2, shape=(2,))
            P1 = basix.ufl.element("Lagrange", self.mesh.basix_cell(), 1)
            TH = basix.ufl.mixed_element([P2, P1])
            self.W = fem.functionspace(self.mesh, TH)
            
            # Separate spaces for interpolation
            self.V = fem.functionspace(
                self.mesh,
                basix.ufl.element("Lagrange", self.mesh.basix_cell(), 2, shape=(2,))
            )
            self.Q = fem.functionspace(
                self.mesh,
                basix.ufl.element("Lagrange", self.mesh.basix_cell(), 1)
            )
        
        def _setup_boundary_conditions(self) -> None:
            """Setup boundary conditions."""
            self._facet_tags = self.mesh.facet_tags
            
            def no_slip(x):
                return np.zeros((2, x.shape[1]))
            self._no_slip_func = no_slip
        
        def _get_inlet_velocity_func(self, t: float, scale: float = 1.0):
            """Get inlet velocity function with optional ramp-up."""
            # Smooth ramp from 0 to 1 over ramp_time
            if t < self.ramp_time:
                ramp = 0.5 * (1 - np.cos(np.pi * t / self.ramp_time))
            else:
                ramp = 1.0
            
            U_max = 1.5 * self.U_mean * scale * ramp
            H = self.H
            
            def inlet_velocity(x):
                values = np.zeros((2, x.shape[1]))
                values[0] = 4 * U_max * x[1] * (H - x[1]) / (H ** 2)
                return values
            
            return inlet_velocity
        
        def solve(
            self,
            inlet_scale: float = 1.0,
            return_full: bool = True,
            progress_callback: Callable = None,
        ) -> np.ndarray:
            """
            Solve the unsteady flow problem.
            
            Parameters
            ----------
            inlet_scale : float
                Scaling factor for inlet velocity
            return_full : bool
                If True (default), return full time trajectory
            progress_callback : Callable, optional
                Called with (step, total_steps) for progress reporting
                
            Returns
            -------
            np.ndarray
                Solution trajectory, shape (n_t, ny, nx, 3) for (u, v, p)
            """
            W = self.W
            V, _ = W.sub(0).collapse()
            
            # Time stepping arrays
            time_steps = np.linspace(0, self.T, self.n_t)
            dt = self.dt
            k = fem.Constant(self.mesh, PETSc.ScalarType(dt))
            
            # Functions for time stepping
            w_n = fem.Function(W)   # Solution at t_n
            w = fem.Function(W)     # Solution at t_{n+1}
            
            # Initialize with zero (fluid at rest)
            w_n.x.array[:] = 0.0
            w.x.array[:] = 0.0
            
            # Test functions
            (u, p) = ufl.split(w)
            (u_n, p_n) = ufl.split(w_n)
            (v, q) = ufl.TestFunctions(W)
            
            # Weak form: Backward Euler time discretization
            # ρ(u - u_n)/dt + ρ(u·∇)u - μ∇²u + ∇p = 0
            # ∇·u = 0
            F = (
                self.rho * inner((u - u_n) / k, v) * dx
                + self.rho * inner(grad(u) * u, v) * dx
                + self.mu * inner(grad(u), grad(v)) * dx
                - p * div(v) * dx
                - q * div(u) * dx
            )
            
            # Store solutions
            solutions = []
            
            # Time stepping loop
            for step, t in enumerate(time_steps):
                # Update boundary conditions for current time
                inlet_func = self._get_inlet_velocity_func(t, inlet_scale)
                
                inlet_bc_func = fem.Function(V)
                inlet_bc_func.interpolate(inlet_func)
                
                inlet_facets = self._facet_tags.find(1)
                inlet_dofs = fem.locate_dofs_topological(
                    (W.sub(0), V), self.mesh.topology.dim - 1, inlet_facets
                )
                bc_inlet = fem.dirichletbc(inlet_bc_func, inlet_dofs, W.sub(0))
                
                noslip_func = fem.Function(V)
                noslip_func.interpolate(self._no_slip_func)
                
                wall_facets = self._facet_tags.find(3)
                wall_dofs = fem.locate_dofs_topological(
                    (W.sub(0), V), self.mesh.topology.dim - 1, wall_facets
                )
                bc_walls = fem.dirichletbc(noslip_func, wall_dofs, W.sub(0))
                
                cylinder_facets = self._facet_tags.find(4)
                cylinder_dofs = fem.locate_dofs_topological(
                    (W.sub(0), V), self.mesh.topology.dim - 1, cylinder_facets
                )
                bc_cylinder = fem.dirichletbc(noslip_func, cylinder_dofs, W.sub(0))
                
                bcs = [bc_inlet, bc_walls, bc_cylinder]
                
                if step == 0:
                    # Store initial state
                    solution = self._extract_solution(w)
                    solutions.append(solution)
                else:
                    # Solve for this time step - dolfinx 0.10+ API
                    problem = NonlinearProblem(
                        F, w, bcs=bcs,
                        petsc_options_prefix=f"navier_stokes_step{step}_",
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
                    try:
                        w = problem.solve()
                        if problem.solver.getConvergedReason() <= 0:
                            print(f"Warning: Newton solver did not converge at t={t:.3f}")
                    except Exception as e:
                        print(f"Solver error at t={t:.3f}: {e}")
                        # Keep previous solution
                        w.x.array[:] = w_n.x.array[:]
                    
                    # Store solution
                    solution = self._extract_solution(w)
                    solutions.append(solution)
                    
                    # Update for next step
                    w_n.x.array[:] = w.x.array[:]
                
                if progress_callback:
                    progress_callback(step, len(time_steps))
            
            # Stack into trajectory array: (n_t, ny, nx, 3)
            trajectory = np.stack(solutions, axis=0)
            
            if return_full:
                return trajectory
            else:
                return trajectory[-1]
        
        def _extract_solution(self, w) -> np.ndarray:
            """Extract solution and interpolate to grid."""
            u_sol = w.sub(0).collapse()
            p_sol = w.sub(1).collapse()
            
            u_grid = self.interpolate_to_grid(u_sol, fill_value=0.0)
            p_grid = self.interpolate_to_grid(p_sol, fill_value=0.0)
            
            u_component = u_grid[:, :, 0]
            v_component = u_grid[:, :, 1]
            
            return np.stack([u_component, v_component, p_grid], axis=-1)
        
        def generate_ic(
            self,
            generator: Union[str, Callable] = "default",
            generator_params: Dict = None,
            seed: int = None,
        ) -> np.ndarray:
            """Generate random inlet velocity scale."""
            if generator_params is None:
                generator_params = {}
            
            if seed is not None:
                np.random.seed(seed)
            
            scale_min = generator_params.get("scale_min", 0.8)
            scale_max = generator_params.get("scale_max", 1.5)
            
            scale = np.random.uniform(scale_min, scale_max)
            return np.array([scale])
        
        def generate_sample(
            self,
            generator: Union[str, Callable] = "default",
            generator_params: Dict = None,
            seed: int = None,
            validate: bool = True,
            max_attempts: int = 3,
        ) -> Tuple[np.ndarray, np.ndarray, Dict]:
            """Generate a single trajectory sample."""
            if generator_params is None:
                generator_params = {}
            
            inlet_scale = self.generate_ic(generator, generator_params, seed)
            
            trajectory = self.solve(inlet_scale=float(inlet_scale[0]))
            
            validation = self.validate_solution(inlet_scale, trajectory)
            
            return inlet_scale, trajectory, validation
        
        def validate_solution(
            self,
            inlet_scale: np.ndarray,
            solution: np.ndarray,
            tol: float = 1e-6,
        ) -> Dict:
            """Validate the solution trajectory."""
            is_valid = (
                not np.isnan(solution).any() and
                not np.isinf(solution).any()
            )
            
            # Compute max velocity over trajectory
            vmag = np.sqrt(solution[:, :, :, 0]**2 + solution[:, :, :, 1]**2)
            
            return {
                'valid': is_valid,
                'max_velocity': vmag.max(),
                'n_time_steps': solution.shape[0],
            }
        
        def generate_dataset(
            self,
            n_samples: int,
            generator: Union[str, Callable] = "default",
            generator_params: Dict = None,
            seed: int = None,
            validate: bool = True,
            show_progress: bool = True,
        ):
            """
            Generate a dataset of time trajectories.
            
            Returns
            -------
            PDEDataset
                Dataset with trajectory data. outputs has shape (n_samples, n_t, ny, nx, 3)
            """
            from tqdm import tqdm
            from pdeforge.core.types import PDEDataset
            
            if generator_params is None:
                generator_params = {}
            
            inputs_list = []
            outputs_list = []
            
            iterator = range(n_samples)
            if show_progress:
                iterator = tqdm(iterator, desc="Generating trajectories")
            
            for i in iterator:
                sample_seed = seed + i if seed is not None else None
                
                inlet_scale, trajectory, info = self.generate_sample(
                    generator=generator,
                    generator_params=generator_params,
                    seed=sample_seed,
                    validate=validate,
                )
                
                inputs_list.append(inlet_scale)
                outputs_list.append(trajectory)
            
            inputs = np.stack(inputs_list, axis=0)
            outputs = np.stack(outputs_list, axis=0)
            
            # Time array for metadata
            time_array = np.linspace(0, self.T, self.n_t)
            
            return PDEDataset(
                inputs=inputs,
                outputs=outputs,
                grid=self.grids,
                input_names=self.INPUT_NAMES,
                output_names=self.OUTPUT_NAMES,
                metadata={
                    "model": "cylinder_flow_2d_unsteady",
                    "params": self.params,
                    "time": time_array.tolist(),
                    "n_time_steps": self.n_t,
                    "time_end": self.T,
                    "backend": self.BACKEND,
                },
            )
