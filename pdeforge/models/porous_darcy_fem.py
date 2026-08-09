"""
Darcy flow through Cahn-Hilliard porous morphologies (FEniCSx-based).

The cross-model pipeline: a seeded `cahn_hilliard` run generates a spinodal
two-phase morphology; thresholding it at u = 0 yields a binary permeability
field (permeable pores, near-impermeable solid); steady Darcy flow

    -∇·(k(x) ∇p) = 0,   p = 1 at x = x_min,   p = 0 at x = x_max,
    no-flux on the side walls

is then driven through the microstructure by the unit pressure drop.

Operator learning task: k(x, y) → (p, ux, uy) with u = -k ∇p. The
generative geometry is the point: every seed grows a fresh labyrinth, at
any resolution, rather than re-slicing one frozen micro-CT image.

Validation: global flux balance (inflow vs outflow across the pressure
boundaries) and the maximum principle for p; the effective permeability
k_eff = Q / Δp is stored per sample.
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

    @register_model("porous_darcy_fem")
    class PorousDarcyFEM(FEniCSModel):
        """
        Steady Darcy flow through a binarized Cahn-Hilliard morphology.

        `ch_time` sets the coarsening stage of the microstructure (early:
        fine spinodal maze; late: coarse blobs); `permeability_contrast`
        sets k_pore / k_solid.

        Examples
        --------
        >>> dataset = generate_dataset(
        ...     model="porous_darcy_fem",
        ...     n_samples=100,
        ...     resolution={"x": 64, "y": 64},
        ...     params={"ch_time": 8.0, "permeability_contrast": 1e3},
        ... )
        """

        NDIM = 2
        TIME_DEPENDENT = False
        INPUT_NAMES = ["k"]
        OUTPUT_NAMES = ["p", "ux", "uy"]

        USER_PARAMS = [
            ParamSpec(
                name="permeability_contrast",
                description="k_pore / k_solid ratio of the two phases",
                default=1e3,
                param_type=ParamType.PHYSICAL,
                bounds=(10.0, 1e6),
                affects="Higher contrast: flow confined to the pore labyrinth",
            ),
            ParamSpec(
                name="ch_time",
                description="Cahn-Hilliard coarsening time of the morphology",
                default=8.0,
                param_type=ParamType.INPUT,
                bounds=(1.0, 100.0),
                affects="Early: fine spinodal maze; late: coarse domains",
            ),
        ]

        DEFAULT_PARAMS = {
            "permeability_contrast": 1e3,
            "ch_time": 8.0,
            "_mesh_n": 64,
        }

        def __init__(self, resolution, domain=None, **params):
            if domain is None:
                domain = {"x": (0.0, 1.0), "y": (0.0, 1.0)}
            merged = {**self.DEFAULT_PARAMS, **params}
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
            self.V = fem.functionspace(self.mesh, ("Lagrange", 1))
            self.W0 = fem.functionspace(self.mesh, ("DG", 0))
            self.W0v = fem.functionspace(self.mesh, ("DG", 0, (2,)))

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
            Solve for a permeability grid field.

            Parameters
            ----------
            ic : np.ndarray
                k(x, y), shape (nx, ny).

            Returns
            -------
            np.ndarray
                Shape (nx, ny, 3): pressure and Darcy velocity (ux, uy).
                Stores flux diagnostics in ``self._last_flux``.
            """
            mesh = self.mesh
            k = fem.Function(self.W0)
            k.x.array[:] = self._field_at_cells(np.asarray(ic, dtype=np.float64))

            p = ufl.TrialFunction(self.V)
            w = ufl.TestFunction(self.V)
            a = ufl.inner(k * ufl.grad(p), ufl.grad(w)) * ufl.dx
            L = fem.Constant(mesh, default_scalar_type(0.0)) * w * ufl.dx

            x0, x1 = self.domain.bounds["x"]
            left = fem.locate_dofs_geometrical(self.V, lambda x: np.isclose(x[0], x0))
            right = fem.locate_dofs_geometrical(self.V, lambda x: np.isclose(x[0], x1))
            bcs = [
                fem.dirichletbc(default_scalar_type(1.0), left, self.V),
                fem.dirichletbc(default_scalar_type(0.0), right, self.V),
            ]

            problem = LinearProblem(
                a,
                L,
                bcs=bcs,
                petsc_options_prefix="porous_darcy_",
                petsc_options={
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps",
                },
            )
            ph = problem.solve()

            # boundary fluxes on the two pressure boundaries
            fdim = mesh.topology.dim - 1
            left_facets = dfx_mesh.locate_entities_boundary(
                mesh, fdim, lambda x: np.isclose(x[0], x0)
            )
            right_facets = dfx_mesh.locate_entities_boundary(
                mesh, fdim, lambda x: np.isclose(x[0], x1)
            )
            facets = np.concatenate([left_facets, right_facets])
            markers = np.concatenate(
                [
                    np.full(len(left_facets), 1, dtype=np.int32),
                    np.full(len(right_facets), 2, dtype=np.int32),
                ]
            )
            order = np.argsort(facets)
            tags = dfx_mesh.meshtags(mesh, fdim, facets[order], markers[order])
            ds = ufl.Measure("ds", domain=mesh, subdomain_data=tags)
            n = ufl.FacetNormal(mesh)
            flux = -k * ufl.dot(ufl.grad(ph), n)
            q_in = -fem.assemble_scalar(fem.form(flux * ds(1)))  # into the domain
            q_out = fem.assemble_scalar(fem.form(flux * ds(2)))  # out of the domain
            self._last_flux = {
                "Q_in": float(q_in),
                "Q_out": float(q_out),
                "imbalance": abs(q_in - q_out) / max(abs(q_in), 1e-30),
                "k_eff": float(q_in),  # unit pressure drop, unit cross-section
            }

            vel = fem.Function(self.W0v)
            vel.interpolate(
                fem.Expression(-k * ufl.grad(ph), self.W0v.element.interpolation_points)
            )
            p_grid = self.interpolate_to_grid(ph)  # (nx, ny)
            v_grid = self.interpolate_to_grid(vel)  # (nx, ny, 2)
            return np.concatenate([p_grid[..., None], v_grid], axis=-1)

        def generate_ic(
            self,
            generator: Union[str, Callable] = "default",
            generator_params: Dict = None,
            seed: int = None,
        ) -> np.ndarray:
            """Grow a Cahn-Hilliard morphology and binarize it into k(x)."""
            if generator_params is None:
                generator_params = {}
            from pdeforge.core.registry import get_model

            ch_time = float(generator_params.get("ch_time", self.params["ch_time"]))
            ch = get_model("cahn_hilliard")(
                resolution={
                    "x": self.output_resolution["x"],
                    "y": self.output_resolution["y"],
                },
                time_end=ch_time,
            )
            phi = ch.solve(ch.generate_ic(seed=seed))  # (ny, nx) spectral layout
            phi = np.asarray(phi).T  # -> (nx, ny), matching the FEM layout
            contrast = float(self.params["permeability_contrast"])
            return np.where(phi > 0.0, 1.0, 1.0 / contrast)

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
            """Finite fields, flux balance to discretization tolerance, and
            the maximum principle for the pressure."""
            flux = getattr(self, "_last_flux", None)
            imbalance = flux["imbalance"] if flux else np.inf
            p = solution[..., 0]
            is_valid = (
                not np.isnan(solution).any()
                and not np.isinf(solution).any()
                and imbalance < tol
                and p.min() > -1e-6
                and p.max() < 1.0 + 1e-6
            )
            out = {
                "valid": bool(is_valid),
                "flux_imbalance": float(imbalance),
                "p_range": (float(p.min()), float(p.max())),
            }
            if flux:
                out.update(flux)
            return out
