"""
2D Rayleigh-Benard convection in a closed cavity (FEniCSx-based).

Boussinesq equations, nondimensionalized on the cavity height and the
thermal diffusion time:

    u_t + (u·∇)u = -∇p + Pr ∇²u + Ra Pr T ŷ
    T_t + u·∇T   = ∇²T
    ∇·u = 0

BCs: no-slip on all walls; T = 1 on the bottom plate, T = 0 on the top,
adiabatic sidewalls. Below Ra_c ≈ 1708 the conduction state T = 1 - y is
stable and Nu = 1; above it convection rolls set in and Nu grows.

Operator learning task: (T-perturbation, Ra, Pr) → (u, v, T). The initial
perturbation seeds WHICH steady roll state the flow settles into (roll
multiplicity), so Ra/Pr sweeps with per-sample perturbations give both
parametric and structural variability.

Time stepping: semi-implicit Oseen steps (advection linearized at the
previous velocity, diffusion implicit) on Taylor-Hood P2/P1 with one
pressure dof pinned, then an implicit advection-diffusion step for T (P2).

Nusselt validation: Nu is the plate-averaged -∂T/∂y; at steady state the
bottom and top values agree, and the sub-critical cavity returns Nu = 1.
Measured (2026-07): Ra = 1e4, Pr = 0.71 on a 48x48 Taylor-Hood mesh gives
Nu = 2.155/2.162 (bottom/top, 0.3% flux imbalance) against the square-
cavity benchmark 2.158 of Ouertatani et al. (2008); at Ra = 800 the
cavity returns Nu = 1.000 with the velocity decaying to zero.
"""

from typing import Callable, Dict, Tuple, Union

import numpy as np

from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model

try:
    import basix
    import dolfinx
    import ufl
    from dolfinx import default_scalar_type, fem
    from dolfinx import mesh as dfx_mesh
    from dolfinx.fem.petsc import LinearProblem
    from mpi4py import MPI

    from pdeforge.core.fenics_base import FEniCSModel

    HAS_FENICSX = True
except ImportError:
    HAS_FENICSX = False

    class FEniCSModel:
        pass


if HAS_FENICSX:

    @register_model("rayleigh_benard_2d")
    class RayleighBenard2D(FEniCSModel):
        """
        Rayleigh-Benard convection in the unit square cavity.

        Examples
        --------
        >>> dataset = generate_dataset(
        ...     model="rayleigh_benard_2d",
        ...     n_samples=20,
        ...     resolution={"x": 64, "y": 64},
        ...     params={"rayleigh": 1e4, "prandtl": 0.71},
        ... )
        """

        NDIM = 2
        TIME_DEPENDENT = True
        INPUT_NAMES = ["T0"]
        OUTPUT_NAMES = ["u", "v", "T"]

        USER_PARAMS = [
            ParamSpec(
                name="rayleigh",
                description="Rayleigh number (onset at Ra_c ~ 1708)",
                default=1e4,
                param_type=ParamType.PHYSICAL,
                bounds=(1e2, 1e6),
                affects="Higher Ra: stronger convection, higher Nusselt",
            ),
            ParamSpec(
                name="prandtl",
                description="Prandtl number",
                default=0.71,
                param_type=ParamType.PHYSICAL,
                bounds=(0.01, 100.0),
            ),
            ParamSpec(
                name="time_end",
                description="March time in thermal-diffusion units",
                default=0.5,
                param_type=ParamType.PHYSICAL,
                bounds=(0.01, 10.0),
                affects="Long enough for a steady state at moderate Ra",
            ),
            ParamSpec(
                name="perturbation",
                description="Amplitude of the random initial T perturbation",
                default=0.05,
                param_type=ParamType.INPUT,
                bounds=(0.0, 0.5),
                affects="Seeds which roll state the flow settles into",
            ),
        ]

        DEFAULT_PARAMS = {
            "rayleigh": 1e4,
            "prandtl": 0.71,
            "time_end": 0.5,
            "perturbation": 0.05,
            "n_time_steps": 11,
            "_mesh_n": 48,
            "_dt": None,  # None: 2.5e-4 scaled by (64 / _mesh_n)
        }

        def __init__(self, resolution, domain=None, **params):
            if domain is None:
                domain = {"x": (0.0, 1.0), "y": (0.0, 1.0)}
            merged = {**self.DEFAULT_PARAMS, **params}
            self.Ra = float(merged["rayleigh"])
            self.Pr = float(merged["prandtl"])
            self.time_end = float(merged["time_end"])
            self.n_time_steps = int(merged["n_time_steps"])
            self._mesh_n = int(merged.get("_mesh_n", 48))
            dt = merged.get("_dt")
            self.dt = float(dt) if dt else 2.5e-4 * (64.0 / self._mesh_n)
            super().__init__(resolution, domain, **params)

        def create_mesh(self):
            x0, x1 = self.domain.bounds["x"]
            y0, y1 = self.domain.bounds["y"]
            return dfx_mesh.create_rectangle(
                MPI.COMM_WORLD,
                [np.array([x0, y0]), np.array([x1, y1])],
                [self._mesh_n, self._mesh_n],
                cell_type=dfx_mesh.CellType.triangle,
            )

        def create_function_spaces(self):
            el_u = basix.ufl.element("Lagrange", self.mesh.basix_cell(), 2, shape=(2,))
            el_p = basix.ufl.element("Lagrange", self.mesh.basix_cell(), 1)
            self.W = fem.functionspace(self.mesh, basix.ufl.mixed_element([el_u, el_p]))
            self.Q = fem.functionspace(self.mesh, ("Lagrange", 2))

        def _sample_grid_at(self, space, grid_field):
            """Nearest-grid sample of a (nx, ny) field at a space's dofs."""
            x = self.grids["x"]
            y = self.grids["y"]
            pts = space.tabulate_dof_coordinates()
            ix = np.clip(
                np.rint((pts[:, 0] - x[0]) / (x[1] - x[0])).astype(int), 0, len(x) - 1
            )
            iy = np.clip(
                np.rint((pts[:, 1] - y[0]) / (y[1] - y[0])).astype(int), 0, len(y) - 1
            )
            return np.asarray(grid_field, dtype=np.float64)[ix, iy]

        def _nusselt(self, T, tag):
            """Plate-averaged -dT/dy across a horizontal boundary."""
            mesh = self.mesh
            fdim = mesh.topology.dim - 1
            y0, y1 = self.domain.bounds["y"]
            yv = y0 if tag == "bottom" else y1
            facets = dfx_mesh.locate_entities_boundary(
                mesh, fdim, lambda x: np.isclose(x[1], yv)
            )
            tags = dfx_mesh.meshtags(
                mesh, fdim, np.sort(facets), np.full(len(facets), 1, dtype=np.int32)
            )
            ds = ufl.Measure("ds", domain=mesh, subdomain_data=tags)(1)
            lx = self.domain.bounds["x"][1] - self.domain.bounds["x"][0]
            val = fem.assemble_scalar(fem.form(-T.dx(1) * ds)) / lx
            return float(val)

        def solve(self, ic: np.ndarray, return_full: bool = False) -> np.ndarray:
            """
            March from the perturbed conduction state to time_end.

            Parameters
            ----------
            ic : np.ndarray
                Initial temperature perturbation, shape (nx, ny); added to
                the conduction profile 1 - y (clipped to [0, 1]).
            return_full : bool
                If True return (n_time_steps, nx, ny, 3) output snapshots.

            Returns
            -------
            np.ndarray
                (nx, ny, 3) final fields (u, v, T), or the trajectory.
                Stores plate Nusselt numbers in ``self._last_nusselt``.
            """
            mesh = self.mesh
            W, Q = self.W, self.Q
            y0, y1 = self.domain.bounds["y"]

            # temperature: conduction profile + perturbation
            T_n = fem.Function(Q)
            ly = y1 - y0
            pts = Q.tabulate_dof_coordinates()
            cond = 1.0 - (pts[:, 1] - y0) / ly
            T_n.x.array[:] = np.clip(cond + self._sample_grid_at(Q, ic), 0.0, 1.0)

            wsol = fem.Function(W)

            # velocity BCs: no-slip everywhere
            V_sub, V_map = W.sub(0).collapse()
            # u_n must live in the collapsed subspace: a standalone P2
            # space has a different dof ordering, and a raw-array copy
            # across the two scrambles the field
            u_n = fem.Function(V_sub)
            zero_v = fem.Function(V_sub)

            def all_walls(x):
                return (
                    np.isclose(x[0], self.domain.bounds["x"][0])
                    | np.isclose(x[0], self.domain.bounds["x"][1])
                    | np.isclose(x[1], y0)
                    | np.isclose(x[1], y1)
                )

            wall_dofs = fem.locate_dofs_geometrical((W.sub(0), V_sub), all_walls)
            bc_u = fem.dirichletbc(zero_v, wall_dofs, W.sub(0))

            # pin one pressure dof (all-Dirichlet velocity leaves p defined
            # up to a constant)
            P_sub, _ = W.sub(1).collapse()
            corner = fem.locate_dofs_geometrical(
                (W.sub(1), P_sub),
                lambda x: np.isclose(x[0], self.domain.bounds["x"][0])
                & np.isclose(x[1], y0),
            )
            zero_p = fem.Function(P_sub)
            bc_p = fem.dirichletbc(zero_p, corner, W.sub(1))

            # temperature BCs: hot bottom, cold top
            bot_dofs = fem.locate_dofs_geometrical(Q, lambda x: np.isclose(x[1], y0))
            top_dofs = fem.locate_dofs_geometrical(Q, lambda x: np.isclose(x[1], y1))
            bc_T = [
                fem.dirichletbc(default_scalar_type(1.0), bot_dofs, Q),
                fem.dirichletbc(default_scalar_type(0.0), top_dofs, Q),
            ]

            dt = self.dt
            n_steps = max(1, int(np.ceil(self.time_end / dt)))
            dt = self.time_end / n_steps
            out_every = max(1, n_steps // max(1, self.n_time_steps - 1))

            u, p = ufl.TrialFunctions(W)
            w, q = ufl.TestFunctions(W)
            ey = ufl.as_vector((0.0, 1.0))
            a_ns = (
                (1.0 / dt) * ufl.dot(u, w) * ufl.dx
                + ufl.dot(ufl.dot(u_n, ufl.nabla_grad(u)), w) * ufl.dx
                + self.Pr * ufl.inner(ufl.grad(u), ufl.grad(w)) * ufl.dx
                - p * ufl.div(w) * ufl.dx
                - q * ufl.div(u) * ufl.dx
            )
            L_ns = (1.0 / dt) * ufl.dot(
                u_n, w
            ) * ufl.dx + self.Ra * self.Pr * T_n * ufl.dot(ey, w) * ufl.dx

            T = ufl.TrialFunction(Q)
            s = ufl.TestFunction(Q)
            a_T = (
                (1.0 / dt) * T * s * ufl.dx
                + ufl.dot(u_n, ufl.grad(T)) * s * ufl.dx
                + ufl.inner(ufl.grad(T), ufl.grad(s)) * ufl.dx
            )
            L_T = (1.0 / dt) * T_n * s * ufl.dx

            lu = {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }

            frames = []

            def snapshot():
                vel = self.interpolate_to_grid(u_n)  # (nx, ny, 2)
                temp = self.interpolate_to_grid(T_n)  # (nx, ny)
                return np.concatenate([vel, temp[..., None]], axis=-1)

            if return_full:
                frames.append(snapshot())

            for step in range(n_steps):
                problem_ns = LinearProblem(
                    a_ns,
                    L_ns,
                    bcs=[bc_u, bc_p],
                    u=wsol,
                    petsc_options_prefix="rb_ns_",
                    petsc_options=lu,
                )
                problem_ns.solve()
                u_new = wsol.sub(0).collapse()
                u_n.x.array[:] = u_new.x.array

                problem_T = LinearProblem(
                    a_T,
                    L_T,
                    bcs=bc_T,
                    petsc_options_prefix="rb_T_",
                    petsc_options=lu,
                )
                T_new = problem_T.solve()
                T_n.x.array[:] = T_new.x.array

                if return_full and (step + 1) % out_every == 0:
                    frames.append(snapshot())

            nu_b = self._nusselt(T_n, "bottom")
            nu_t = self._nusselt(T_n, "top")
            self._last_nusselt = {
                "Nu_bottom": nu_b,
                "Nu_top": nu_t,
                "imbalance": abs(nu_b - nu_t) / max(abs(nu_b), 1e-30),
            }

            if return_full:
                return np.stack(frames, axis=0)
            return snapshot()

        def generate_ic(
            self,
            generator: Union[str, Callable] = "default",
            generator_params: Dict = None,
            seed: int = None,
        ) -> np.ndarray:
            """Smooth random T perturbation, zero on the plates."""
            if generator_params is None:
                generator_params = {}
            rng = np.random.default_rng(seed)
            nx = self.output_resolution["x"]
            ny = self.output_resolution["y"]
            amp = float(generator_params.get("amplitude", self.params["perturbation"]))
            cutoff = int(generator_params.get("cutoff", 3))

            noise = rng.standard_normal((nx, ny))
            noise_hat = np.fft.fft2(noise)
            kx = np.fft.fftfreq(nx) * nx
            ky = np.fft.fftfreq(ny) * ny
            KX, KY = np.meshgrid(kx, ky, indexing="ij")
            mask = np.sqrt(KX**2 + KY**2) <= cutoff
            smooth = np.fft.ifft2(noise_hat * mask).real
            smooth = amp * smooth / (np.abs(smooth).max() + 1e-12)

            y = self.grids["y"]
            y0, y1 = self.domain.bounds["y"]
            envelope = np.sin(np.pi * (y - y0) / (y1 - y0))  # zero at plates
            return smooth * envelope[None, :]

        def generate_sample(
            self,
            generator: Union[str, Callable] = "default",
            generator_params: Dict = None,
            seed: int = None,
            validate: bool = True,
            max_attempts: int = 10,
        ) -> Tuple[np.ndarray, np.ndarray, Dict]:
            for attempt in range(max_attempts):
                current_seed = seed + attempt if seed is not None else None
                ic = self.generate_ic(generator, generator_params, current_seed)
                solution = self.solve(ic)
                if not validate:
                    return ic, solution, {"valid": True}
                validation = self.validate_solution(ic, solution)
                if validation["valid"]:
                    return ic, solution, validation
            raise RuntimeError(
                f"Failed to generate valid sample after {max_attempts} attempts"
            )

        def validate_solution(self, ic, solution, tol: float = 0.05) -> Dict:
            """Finite fields, T within bounds, and plate flux balance
            (meaningful once the flow is statistically steady)."""
            nus = getattr(self, "_last_nusselt", None)
            imbalance = nus["imbalance"] if nus else np.inf
            T = solution[..., 2] if solution.ndim == 3 else solution[-1, ..., 2]
            is_valid = (
                not np.isnan(solution).any()
                and not np.isinf(solution).any()
                and T.min() > -0.05
                and T.max() < 1.05
                and imbalance < tol
            )
            out = {"valid": bool(is_valid), "nusselt_imbalance": float(imbalance)}
            if nus:
                out.update(nus)
            return out
