"""
2D linear elasticity with random stiff inclusions (FEniCSx-based).

Plane-strain elasticity on the unit square:

    -∇·σ(u) = 0,   σ = λ(x) tr(ε) I + 2 μ(x) ε,   ε = (∇u + ∇uᵀ)/2

The Lamé fields derive from a heterogeneous Young's modulus E(x): a matrix
phase seeded with random circular inclusions of contrasting stiffness. The
bottom edge is clamped, a uniform traction acts on the top edge, and the
sides are free.

Operator learning task: E(x, y) → (u, v, von Mises) — the elasticity
analogue of the Darcy coefficient-to-solution map.

Validation is Clapeyron's theorem: for the discrete Galerkin solution the
strain energy equals half the external work exactly (test w = u_h), so the
energy balance checks assembly and solver together at solver precision.
"""

from typing import Callable, Dict, Tuple, Union

import numpy as np

from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model

try:
    import basix  # noqa: F401
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

    @register_model("elasticity_2d")
    class Elasticity2D(FEniCSModel):
        """
        Plane-strain linear elasticity with random inclusions.

        Maps a heterogeneous Young's-modulus field to displacement and von
        Mises stress under a fixed top traction; the bottom edge is
        clamped. The traction components are physical parameters, so the
        UQ layer can draw them per sample alongside the random inclusions.

        Examples
        --------
        >>> dataset = generate_dataset(
        ...     model="elasticity_2d",
        ...     n_samples=100,
        ...     resolution={"x": 64, "y": 64},
        ...     params={"e_inclusion": 10.0, "traction_y": -1.0},
        ... )
        """

        NDIM = 2
        TIME_DEPENDENT = False
        INPUT_NAMES = ["E"]
        OUTPUT_NAMES = ["u", "v", "von_mises"]

        USER_PARAMS = [
            ParamSpec(
                name="e_matrix",
                description="Young's modulus of the matrix phase",
                default=1.0,
                param_type=ParamType.PHYSICAL,
                bounds=(0.01, 100.0),
                affects="Overall compliance scale",
            ),
            ParamSpec(
                name="e_inclusion",
                description="Young's modulus of the inclusions",
                default=10.0,
                param_type=ParamType.PHYSICAL,
                bounds=(0.01, 1000.0),
                affects="Stiffness contrast: >1 stiff, <1 soft inclusions",
            ),
            ParamSpec(
                name="poisson",
                description="Poisson ratio (uniform)",
                default=0.3,
                param_type=ParamType.PHYSICAL,
                bounds=(0.05, 0.45),
            ),
            ParamSpec(
                name="traction_x",
                description="Traction x-component on the top edge",
                default=0.0,
                param_type=ParamType.PHYSICAL,
                bounds=(-10.0, 10.0),
            ),
            ParamSpec(
                name="traction_y",
                description="Traction y-component on the top edge",
                default=-1.0,
                param_type=ParamType.PHYSICAL,
                bounds=(-10.0, 10.0),
                affects="Negative = compression toward the clamped edge",
            ),
            ParamSpec(
                name="n_inclusions",
                description="Number of random circular inclusions",
                default=6,
                param_type=ParamType.INPUT,
                bounds=(0, 40),
            ),
        ]

        DEFAULT_PARAMS = {
            "e_matrix": 1.0,
            "e_inclusion": 10.0,
            "poisson": 0.3,
            "traction_x": 0.0,
            "traction_y": -1.0,
            "n_inclusions": 6,
            "_mesh_n": 64,
            "_r_min": 0.05,
            "_r_max": 0.15,
        }

        def __init__(self, resolution, domain=None, **params):
            if domain is None:
                domain = {"x": (0.0, 1.0), "y": (0.0, 1.0)}
            merged = {**self.DEFAULT_PARAMS, **params}
            self.nu = merged["poisson"]
            self._mesh_n = int(merged.get("_mesh_n", 64))
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
            self.V = fem.functionspace(self.mesh, ("Lagrange", 1, (2,)))
            self.W0 = fem.functionspace(self.mesh, ("DG", 0))

        def _field_at_cells(self, grid_field):
            """Sample a (nx, ny) grid field at DG0 dof (cell-mid) points."""
            x = self.grids["x"]
            y = self.grids["y"]
            pts = self.W0.tabulate_dof_coordinates()
            ix = np.clip(
                np.rint((pts[:, 0] - x[0]) / (x[1] - x[0])).astype(int), 0, len(x) - 1
            )
            iy = np.clip(
                np.rint((pts[:, 1] - y[0]) / (y[1] - y[0])).astype(int), 0, len(y) - 1
            )
            return grid_field[ix, iy]

        def solve(self, ic: np.ndarray) -> np.ndarray:
            """
            Solve for a Young's-modulus grid field.

            Parameters
            ----------
            ic : np.ndarray
                E(x, y), shape (nx, ny).

            Returns
            -------
            np.ndarray
                Shape (nx, ny, 3): displacement (u, v) and von Mises
                stress. Also stores the energy balance in
                ``self._last_energy``.
            """
            mesh = self.mesh
            E = fem.Function(self.W0)
            E.x.array[:] = self._field_at_cells(np.asarray(ic, dtype=np.float64))

            nu = self.nu
            lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            mu = E / (2.0 * (1.0 + nu))

            def eps(w):
                return ufl.sym(ufl.grad(w))

            def sigma(w):
                return lam * ufl.tr(eps(w)) * ufl.Identity(2) + 2.0 * mu * eps(w)

            u = ufl.TrialFunction(self.V)
            w = ufl.TestFunction(self.V)

            y1 = self.domain.bounds["y"][1]
            top_facets = dfx_mesh.locate_entities_boundary(
                mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[1], y1)
            )
            tags = dfx_mesh.meshtags(
                mesh,
                mesh.topology.dim - 1,
                top_facets,
                np.full(len(top_facets), 1, dtype=np.int32),
            )
            ds_top = ufl.Measure("ds", domain=mesh, subdomain_data=tags)(1)

            traction = fem.Constant(
                mesh,
                np.array(
                    [self.params["traction_x"], self.params["traction_y"]],
                    dtype=default_scalar_type,
                ),
            )

            a = ufl.inner(sigma(u), eps(w)) * ufl.dx
            L = ufl.dot(traction, w) * ds_top

            y0 = self.domain.bounds["y"][0]
            clamped = fem.locate_dofs_geometrical(
                self.V, lambda x: np.isclose(x[1], y0)
            )
            bc = fem.dirichletbc(
                np.zeros(2, dtype=default_scalar_type), clamped, self.V
            )

            problem = LinearProblem(
                a,
                L,
                bcs=[bc],
                petsc_options_prefix="elasticity_",
                petsc_options={
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps",
                },
            )
            uh = problem.solve()

            # Clapeyron check at FEM level, before grid interpolation
            strain_energy = fem.assemble_scalar(
                fem.form(0.5 * ufl.inner(sigma(uh), eps(uh)) * ufl.dx)
            )
            external_work = fem.assemble_scalar(
                fem.form(ufl.dot(traction, uh) * ds_top)
            )
            self._last_energy = {
                "strain_energy": float(strain_energy),
                "external_work": float(external_work),
                "balance": abs(2.0 * strain_energy - external_work)
                / max(abs(external_work), 1e-30),
            }

            # von Mises with the plane-strain out-of-plane stress
            s2 = sigma(uh)
            szz = lam * ufl.tr(eps(uh))
            tr3 = (ufl.tr(s2) + szz) / 3.0
            dev_xx = s2[0, 0] - tr3
            dev_yy = s2[1, 1] - tr3
            dev_zz = szz - tr3
            vm_expr = ufl.sqrt(
                1.5 * (dev_xx**2 + dev_yy**2 + dev_zz**2 + 2.0 * s2[0, 1] ** 2)
            )
            vm = fem.Function(self.W0)
            vm.interpolate(
                fem.Expression(vm_expr, self.W0.element.interpolation_points)
            )

            disp = self.interpolate_to_grid(uh)  # (nx, ny, 2)
            vm_grid = self.interpolate_to_grid(vm)  # (nx, ny)
            return np.concatenate([disp, vm_grid[..., None]], axis=-1)

        def generate_ic(
            self,
            generator: Union[str, Callable] = "default",
            generator_params: Dict = None,
            seed: int = None,
        ) -> np.ndarray:
            """Matrix stiffness seeded with random circular inclusions."""
            if generator_params is None:
                generator_params = {}
            rng = np.random.default_rng(seed)

            n_incl = int(
                generator_params.get("n_inclusions", self.params["n_inclusions"])
            )
            r_min = generator_params.get("r_min", self.params.get("_r_min", 0.05))
            r_max = generator_params.get("r_max", self.params.get("_r_max", 0.15))

            x = self.grids["x"]
            y = self.grids["y"]
            X, Y = np.meshgrid(x, y, indexing="ij")
            lx = x[-1] - x[0]
            ly = y[-1] - y[0]

            E = np.full_like(X, float(self.params["e_matrix"]))
            for _ in range(n_incl):
                cx = x[0] + rng.uniform(0.1, 0.9) * lx
                cy = y[0] + rng.uniform(0.1, 0.9) * ly
                r = rng.uniform(r_min, r_max) * min(lx, ly)
                E[(X - cx) ** 2 + (Y - cy) ** 2 <= r**2] = float(
                    self.params["e_inclusion"]
                )
            return E

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

        def validate_solution(self, ic, solution, tol: float = 1e-6) -> Dict:
            """Finite fields + the Clapeyron energy balance from the last solve."""
            energy = getattr(self, "_last_energy", None)
            balance = energy["balance"] if energy else np.inf
            is_valid = (
                not np.isnan(solution).any()
                and not np.isinf(solution).any()
                and balance < tol
            )
            out = {"valid": bool(is_valid), "energy_balance": float(balance)}
            if energy:
                out.update(energy)
            return out
