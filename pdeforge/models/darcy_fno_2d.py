"""
Canonical Darcy flow (the FNO benchmark family), faithfully.

    -div( a(x, y) grad u ) = f    on the unit square,
    u = 0 on the boundary (Dirichlet), f = 1 by default.

This reproduces the setup of the classic MATLAB-generated FNO Darcy datasets:
a 5-point finite-difference solve on a grid INCLUDING the boundary (the
canonical 421 x 421 has spacing 1/420), with coefficients drawn from the
canonical Gaussian measure N(0, (-Laplacian + tau^2)^(-alpha)) under two
pushforwards:

- coeff="lognormal":  a = exp(psi)              (the Darcy421 family;
  measured from the distributed data: alpha = 2, tau = 3, sigma = 0.2918)
- coeff="piececonst": a = kappa_plus where psi >= threshold else kappa_minus
  (the piececonst_r421 family; canonical values 12 and 3)

Every knob of the original generator is exposed: alpha, tau, sigma (field
scale), kappa_plus/kappa_minus/threshold, the forcing constant, and the
face-averaging scheme. Unlike the frozen .mat files, any resolution works.

Operator learning task: a(x, y) -> u(x, y).
"""

from typing import Callable, Dict, Tuple, Union

import numpy as np

from pdeforge.core.base import PDEModel
from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model
from pdeforge.generators.initial_conditions import GRFNeumannGenerator


@register_model("darcy_fno_2d")
class DarcyFNO2D(PDEModel):
    """
    The canonical Darcy benchmark, regenerable: FD Dirichlet solve +
    the exact coefficient measure of the classic datasets, with knobs.
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
            affects="lognormal = Darcy421 family; piececonst = {3,12} family",
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
            description="GRF pointwise std (lognormal contrast)",
            default=0.2918,
            param_type=ParamType.INPUT,
            bounds=(0.01, 2.0),
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
        "alpha": 2.0,
        "tau": 3.0,
        "sigma": 0.2918,
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
        self.alpha_grf = p["alpha"]
        self.tau_grf = p["tau"]
        self.sigma_grf = p["sigma"]
        self.f_const = p.get("forcing", 1.0)
        self.face_average = p.get("_face_average", "arithmetic")

        self.nx, self.ny = resolution["x"], resolution["y"]
        self.h = 1.0 / (self.nx - 1)  # grid INCLUDES boundaries

    def _setup_grids(self):
        # Non-periodic problem: grids include both endpoints (canonical
        # r x r convention, spacing 1/(r-1)).
        self.grids = {}
        for dim in self.grid_spec.resolution.keys():
            self.grids[dim] = self.grid_spec.get_grid(dim, endpoint=True)

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
            thr = self.params.get("threshold", 0.0) * self.sigma_grf
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

    def solve(self, ic, return_info=False):
        """a(x, y) -> u(x, y); u = 0 on the boundary."""
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        a = np.asarray(ic, dtype=float)
        ny, nx = a.shape
        h2 = self.h**2
        ax, ay = self._face_coeffs(a)  # ax: (ny, nx-1), ay: (ny-1, nx)

        m, n = ny - 2, nx - 2  # interior unknowns
        idx = np.arange(m * n).reshape(m, n)

        # flux-form 5-point stencil on interior nodes (i=1..ny-2, j=1..nx-2)
        aW = ax[1:-1, :-1]  # face between (i, j-1) and (i, j)
        aE = ax[1:-1, 1:]
        aS = ay[:-1, 1:-1]
        aN = ay[1:, 1:-1]
        diag = (aW + aE + aS + aN).ravel() / h2

        rows, cols, vals = [idx.ravel()], [idx.ravel()], [diag]

        def couple(mask_rows, mask_cols, coeff):
            rows.append(mask_rows.ravel())
            cols.append(mask_cols.ravel())
            vals.append(-coeff.ravel() / h2)

        couple(idx[:, 1:], idx[:, :-1], aW[:, 1:])  # west neighbours
        couple(idx[:, :-1], idx[:, 1:], aE[:, :-1])  # east
        couple(idx[1:, :], idx[:-1, :], aS[1:, :])  # south
        couple(idx[:-1, :], idx[1:, :], aN[:-1, :])  # north

        A = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(m * n, m * n),
        )
        b = np.full(m * n, float(self.f_const))

        u_int = spla.spsolve(A, b)

        u = np.zeros((ny, nx))
        u[1:-1, 1:-1] = u_int.reshape(m, n)
        return u

    def apply_operator(self, a, u):
        """Discrete -div(a grad u) on interior nodes (verification helper)."""
        h2 = self.h**2
        ax, ay = self._face_coeffs(np.asarray(a, dtype=float))
        u = np.asarray(u, dtype=float)
        aW, aE = ax[1:-1, :-1], ax[1:-1, 1:]
        aS, aN = ay[:-1, 1:-1], ay[1:, 1:-1]
        return (
            aW * (u[1:-1, 1:-1] - u[1:-1, :-2])
            - aE * (u[1:-1, 2:] - u[1:-1, 1:-1])
            + aS * (u[1:-1, 1:-1] - u[:-2, 1:-1])
            - aN * (u[2:, 1:-1] - u[1:-1, 1:-1])
        ) / h2

    def validate_solution(self, ic, solution, tol=1e-6):
        is_valid = (
            not np.isnan(solution).any()
            and not np.isinf(solution).any()
            and np.abs(solution).max() < 1e6
        )
        return {"valid": is_valid, "max_value": float(np.abs(solution).max())}
