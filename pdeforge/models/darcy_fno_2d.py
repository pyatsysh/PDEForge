"""
Canonical Darcy flow (the FNO benchmark family), faithfully.

    -div( a(x, y) grad u ) = f    on the unit square,
    u = 0 on the boundary (Dirichlet), f = 1 by default.

This is a transcription of the MATLAB generator that produced the classic
FNO Darcy datasets (`GRF.m` + `solve_gwf.m`), reproducing the distributed
421 x 421 data to float32 round-off. Coefficients are drawn from the
canonical Gaussian measure N(0, tau^(2 alpha - 2) (-Laplacian + tau^2)^(-alpha))
under two pushforwards:

- coeff="lognormal":  a = exp(psi)              (the Darcy421 family;
  canonical alpha = 2, tau = 3, giving sigma = 0.292083)
- coeff="piececonst": a = kappa_plus where psi >= threshold else kappa_minus
  (the piececonst family; the distributed files use 12 and 3, the published
  demo script 12 and 4 -- both are knobs here)

The grid convention is the subtle part, and getting it wrong costs 0.5%.
The original solves on the NODE grid (K points, spacing 1/(K-1), zero
Dirichlet at the boundary nodes) but takes its input and returns its output
on the CELL-CENTRE grid (K points at (2i+1)/(2K), spacing 1/K), moving
between the two with a not-a-knot cubic spline. So the published arrays are
samples of a node-grid solution at cell centres, which is why their boundary
values are small but not zero. `grid="canonical"` reproduces that exactly;
`grid="node"` drops both transfers for a clean solve with an exactly zero
boundary, which is the better choice for new data.

Every knob of the original stays exposed: alpha, tau, sigma (field scale),
kappa_plus/kappa_minus/threshold, the forcing constant, the face-averaging
scheme, and the grid convention. Unlike the frozen .mat files, any
resolution works.

Operator learning task: a(x, y) -> u(x, y).
"""

from typing import Optional

import numpy as np

from pdeforge.core.base import PDEModel
from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model
from pdeforge.generators.initial_conditions import GRFNeumannGenerator


def _spline_transfer(field: np.ndarray, src: list, dst: list) -> np.ndarray:
    """
    MATLAB's ``interp2(..., 'spline')``: a tensor-product not-a-knot cubic
    spline, extrapolating outside the source grid (which is exactly what the
    canonical generator relies on at the boundary).
    """
    from scipy.interpolate import CubicSpline

    for axis, (s, dd) in enumerate(zip(src, dst)):
        field = CubicSpline(s, field, axis=axis, bc_type="not-a-knot")(dd)
    return field


@register_model("darcy_fno_2d")
class DarcyFNO2D(PDEModel):
    """
    The canonical Darcy benchmark, regenerable: the original's FD Dirichlet
    solve and coefficient measure, with knobs.
    """

    NDIM = 2
    TIME_DEPENDENT = False  # steady elliptic solve
    INPUT_NAMES = ["a"]
    OUTPUT_NAMES = ["u"]

    USER_PARAMS = [
        ParamSpec(
            name="coeff",
            description="Coefficient family",
            default="lognormal",
            param_type=ParamType.INPUT,
            choices=["lognormal", "piececonst"],
            affects="lognormal = Darcy421 family; piececonst = two-phase family",
        ),
        ParamSpec(
            name="grid",
            description="Grid convention for the stored fields",
            default="canonical",
            param_type=ParamType.GEOMETRY,
            choices=["canonical", "node"],
            affects=(
                "canonical = cell centres with the original's spline transfers "
                "(bit-reproduces the distributed data); node = direct node-grid "
                "solve, exactly zero on the boundary"
            ),
        ),
        ParamSpec(
            name="alpha",
            description="GRF spectral decay exponent",
            default=2.0,
            param_type=ParamType.INPUT,
            bounds=(1.1, 6.0),
            affects="Higher alpha -> smoother coefficient fields",
        ),
        ParamSpec(
            name="tau",
            description="GRF inverse correlation length",
            default=3.0,
            param_type=ParamType.INPUT,
            bounds=(0.5, 30.0),
            affects="Higher tau -> shorter-range correlations",
        ),
        ParamSpec(
            name="sigma",
            description="GRF pointwise std (None = canonical tau^(alpha-1))",
            default=None,
            param_type=ParamType.INPUT,
            affects="exp(+-sigma) sets the permeability contrast",
        ),
        ParamSpec(
            name="kappa_plus",
            description="High permeability (piececonst)",
            default=12.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.1, 100.0),
        ),
        ParamSpec(
            name="kappa_minus",
            description="Low permeability (piececonst)",
            default=3.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.01, 100.0),
        ),
        ParamSpec(
            name="threshold",
            description="Level-set threshold on psi (piececonst)",
            default=0.0,
            param_type=ParamType.INPUT,
            bounds=(-1.0, 1.0),
            affects="Shifts the volume fraction of the two phases",
        ),
        ParamSpec(
            name="forcing",
            description="Constant source term f",
            default=1.0,
            param_type=ParamType.PHYSICAL,
            bounds=(-100.0, 100.0),
        ),
    ]

    DEFAULT_PARAMS = {
        "coeff": "lognormal",
        "grid": "canonical",
        "alpha": 2.0,
        "tau": 3.0,
        "sigma": None,
        "kappa_plus": 12.0,
        "kappa_minus": 3.0,
        "threshold": 0.0,
        "forcing": 1.0,
        "_face_average": "arithmetic",  # canonical MATLAB convention
    }

    def __init__(self, resolution, domain=None, **params):
        super().__init__(resolution, domain, **params)

        p = self.params
        self.coeff = p["coeff"]
        self.grid_mode = p.get("grid", "canonical")
        if self.grid_mode not in ("canonical", "node"):
            raise ValueError(
                f"grid must be 'canonical' or 'node', got {self.grid_mode!r}"
            )
        self.alpha_grf = p["alpha"]
        self.tau_grf = p["tau"]
        self.sigma_grf = p["sigma"]
        self.f_const = p.get("forcing", 1.0)
        self.face_average = p.get("_face_average", "arithmetic")

        self.nx, self.ny = resolution["x"], resolution["y"]
        # The solve always happens on the node grid, which includes both
        # endpoints -- so the spacing is 1/(n-1) whatever the stored fields
        # are sampled on.
        self.hx = 1.0 / (self.nx - 1)
        self.hy = 1.0 / (self.ny - 1)
        self.h = self.hx  # square-grid shorthand, kept for callers

    def _setup_grids(self):
        """Where the STORED fields live (the solve grid is always the nodes)."""
        mode = {**self.DEFAULT_PARAMS, **self.params}.get("grid", "canonical")
        self.grids = {}
        for dim, n in self.grid_spec.resolution.items():
            lo, hi = self.domain.bounds[dim]
            if mode == "canonical":  # cell centres: (2i+1)/(2n)
                self.grids[dim] = lo + (hi - lo) * (2 * np.arange(n) + 1) / (2 * n)
            else:
                self.grids[dim] = np.linspace(lo, hi, n, endpoint=True)

    # -- grid transfers (the canonical generator's two interp2 calls) --

    @property
    def _cell_grids(self):
        return [(2 * np.arange(n) + 1) / (2 * n) for n in (self.ny, self.nx)]

    @property
    def _node_grids(self):
        return [np.linspace(0.0, 1.0, n) for n in (self.ny, self.nx)]

    # -- coefficient sampling (the input measure) --

    def generate_ic(self, generator="default", generator_params=None, seed=None):
        """Draw a coefficient field a(x, y) from the canonical measure."""
        if generator_params is None:
            generator_params = {}
        gen = GRFNeumannGenerator(
            alpha=generator_params.get("alpha", self.alpha_grf),
            tau=generator_params.get("tau", self.tau_grf),
            sigma=generator_params.get("sigma", self.sigma_grf),
        )
        psi = gen.generate(shape=(self.ny, self.nx), seed=seed)
        if self.coeff == "lognormal":
            return np.exp(psi)
        if self.coeff == "piececonst":
            thr = self.params.get("threshold", 0.0) * gen.expected_std(
                (self.ny, self.nx)
            )
            return np.where(
                psi >= thr, self.params["kappa_plus"], self.params["kappa_minus"]
            ).astype(float)
        raise ValueError(f"Unknown coeff family: {self.coeff!r}")

    # -- the FD Dirichlet solve --

    def _face_coeffs(self, a):
        """Coefficients at cell faces (arithmetic or harmonic mean)."""
        if self.face_average == "harmonic":
            ax = 2.0 * a[:, :-1] * a[:, 1:] / (a[:, :-1] + a[:, 1:])
            ay = 2.0 * a[:-1, :] * a[1:, :] / (a[:-1, :] + a[1:, :])
        else:  # arithmetic (canonical)
            ax = 0.5 * (a[:, :-1] + a[:, 1:])
            ay = 0.5 * (a[:-1, :] + a[1:, :])
        return ax, ay

    def _solve_nodes(self, a, f):
        """5-point Dirichlet solve on the node grid; a and the result are nodal."""
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        ny, nx = a.shape
        ax, ay = self._face_coeffs(a)  # ax: (ny, nx-1), ay: (ny-1, nx)

        m, n = ny - 2, nx - 2  # interior unknowns
        idx = np.arange(m * n).reshape(m, n)

        # flux-form 5-point stencil on interior nodes (i=1..ny-2, j=1..nx-2)
        aW = ax[1:-1, :-1] / self.hx**2  # face between (i, j-1) and (i, j)
        aE = ax[1:-1, 1:] / self.hx**2
        aS = ay[:-1, 1:-1] / self.hy**2
        aN = ay[1:, 1:-1] / self.hy**2
        diag = (aW + aE + aS + aN).ravel()

        rows, cols, vals = [idx.ravel()], [idx.ravel()], [diag]

        def couple(mask_rows, mask_cols, coeff):
            rows.append(mask_rows.ravel())
            cols.append(mask_cols.ravel())
            vals.append(-coeff.ravel())

        couple(idx[:, 1:], idx[:, :-1], aW[:, 1:])  # west neighbours
        couple(idx[:, :-1], idx[:, 1:], aE[:, :-1])  # east
        couple(idx[1:, :], idx[:-1, :], aS[1:, :])  # south
        couple(idx[:-1, :], idx[1:, :], aN[:-1, :])  # north

        A = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(m * n, m * n),
        )
        u = np.zeros((ny, nx))
        u[1:-1, 1:-1] = spla.spsolve(A, np.asarray(f)[1:-1, 1:-1].ravel()).reshape(m, n)
        return u

    def solve(self, ic, return_info=False):
        """a(x, y) -> u(x, y), on whichever grid `grid` selects."""
        a = np.asarray(ic, dtype=float)
        f = np.full(a.shape, float(self.f_const))

        if self.grid_mode == "canonical":
            cell, node = self._cell_grids, self._node_grids
            # the original interpolates BOTH the coefficient and the forcing
            # onto the node grid before solving
            a = _spline_transfer(a, cell, node)
            f = _spline_transfer(f, cell, node)
            u = _spline_transfer(self._solve_nodes(a, f), node, cell)
        else:
            u = self._solve_nodes(a, f)

        if return_info:
            return u, {"grid": self.grid_mode, "solver": "spsolve"}
        return u

    def apply_operator(self, a, u):
        """
        Discrete -div(a grad u) on interior NODES (verification helper).

        Both arguments must be nodal, so this pairs with grid="node"; under
        the canonical convention the stored fields are cell-centred samples
        and do not satisfy the node-grid equation exactly.
        """
        ax, ay = self._face_coeffs(np.asarray(a, dtype=float))
        u = np.asarray(u, dtype=float)
        aW, aE = ax[1:-1, :-1], ax[1:-1, 1:]
        aS, aN = ay[:-1, 1:-1], ay[1:, 1:-1]
        return (
            aW * (u[1:-1, 1:-1] - u[1:-1, :-2]) - aE * (u[1:-1, 2:] - u[1:-1, 1:-1])
        ) / self.hx**2 + (
            aS * (u[1:-1, 1:-1] - u[:-2, 1:-1]) - aN * (u[2:, 1:-1] - u[1:-1, 1:-1])
        ) / self.hy**2

    def validate_solution(self, ic, solution, tol=1e-6):
        is_valid = (
            not np.isnan(solution).any()
            and not np.isinf(solution).any()
            and np.abs(solution).max() < 1e6
        )
        return {"valid": is_valid, "max_value": float(np.abs(solution).max())}
