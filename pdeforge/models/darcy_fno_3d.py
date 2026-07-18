"""
Canonical Darcy flow extended to 3D — the knob no frozen dataset has.

    -div( a(x, y, z) grad u ) = f    on the unit cube,
    u = 0 on the boundary (Dirichlet), f = 1 by default.

The 2D canonical Darcy (darcy_fno_2d) is validated against the distributed
FNO data; this model extends the SAME input measure — the cosine-KL Gaussian
N(0, (-Laplacian + tau^2)^(-alpha)), trace-class in 3D for 2*alpha > 3 —
and the same coefficient pushforwards (lognormal / two-phase piececonst)
to the unit cube. No canonical 3D Darcy file exists to download anywhere;
dimension itself is the knob here.

Solver: 7-point finite differences on the boundary-inclusive grid. Direct
sparse LU is used for small grids; larger grids switch to Jacobi-
preconditioned conjugate gradients (3D LU fill-in is prohibitive). The
matrix is SPD, and both paths agree to solver tolerance (tested).

Operator learning task: a(x, y, z) -> u(x, y, z).
"""

from typing import Dict, Tuple

import numpy as np

from pdeforge.core.base import PDEModel
from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model
from pdeforge.generators.initial_conditions import GRFNeumannGenerator


@register_model("darcy_fno_3d")
class DarcyFNO3D(PDEModel):
    """
    3D Darcy with the canonical coefficient measure, regenerable at any
    resolution. Mind memory: pair large runs with chunked generation (to=).
    """

    NDIM = 3
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
        ),
        ParamSpec(
            name="alpha",
            description="GRF spectral decay exponent",
            default=2.0,
            param_type=ParamType.INPUT,
            bounds=(1.6, 6.0),
            affects="Trace-class in 3D needs alpha > 1.5; canonical is 2.0",
        ),
        ParamSpec(
            name="tau",
            description="GRF inverse correlation length",
            default=3.0,
            param_type=ParamType.INPUT,
            bounds=(0.5, 30.0),
        ),
        ParamSpec(
            name="sigma",
            description="GRF pointwise std (lognormal contrast)",
            default=0.2918,
            param_type=ParamType.INPUT,
            bounds=(0.01, 2.0),
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
        "_face_average": "arithmetic",
        "_cg_tol": 1e-10,
        "_direct_max_unknowns": 30_000,
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

        self.nx = resolution["x"]
        self.ny = resolution["y"]
        self.nz = resolution["z"]
        if not (self.nx == self.ny == self.nz):
            raise ValueError("darcy_fno_3d expects an isotropic n^3 grid")
        self.h = 1.0 / (self.nx - 1)  # grid INCLUDES boundaries
        self.field_shape = (self.nz, self.ny, self.nx)

    def _setup_grids(self):
        # Non-periodic: grids include both endpoints (spacing 1/(n-1)).
        self.grids = {}
        for dim in self.grid_spec.resolution.keys():
            self.grids[dim] = self.grid_spec.get_grid(dim, endpoint=True)

    # -- coefficient sampling ----------------------------------------------

    def generate_ic(self, generator="default", generator_params=None, seed=None):
        """Draw a coefficient field a(x, y, z) from the canonical measure."""
        if generator_params is None:
            generator_params = {}
        gen = GRFNeumannGenerator(
            alpha=generator_params.get("alpha", self.alpha_grf),
            tau=generator_params.get("tau", self.tau_grf),
            sigma=generator_params.get("sigma", self.sigma_grf),
        )
        psi = gen.generate(shape=self.field_shape, seed=seed)
        if self.coeff == "lognormal":
            return np.exp(psi)
        if self.coeff == "piececonst":
            thr = self.params.get("threshold", 0.0) * self.sigma_grf
            return np.where(
                psi >= thr, self.params["kappa_plus"], self.params["kappa_minus"]
            ).astype(float)
        raise ValueError(f"Unknown coeff family: {self.coeff!r}")

    # -- FD Dirichlet solve -------------------------------------------------

    def _face_coeffs(self, a):
        """Coefficients on the three face families (axis 0, 1, 2)."""
        if self.face_average == "harmonic":
            f = lambda p, q: 2.0 * p * q / (p + q)
        else:
            f = lambda p, q: 0.5 * (p + q)
        a0 = f(a[:-1, :, :], a[1:, :, :])  # z-faces
        a1 = f(a[:, :-1, :], a[:, 1:, :])  # y-faces
        a2 = f(a[:, :, :-1], a[:, :, 1:])  # x-faces
        return a0, a1, a2

    def _assemble(self, a):
        import scipy.sparse as sp

        h2 = self.h**2
        a0, a1, a2 = self._face_coeffs(a)
        m = self.nz - 2
        idx = np.arange(m**3).reshape(m, m, m)

        # neighbour face coefficients seen from interior nodes
        aD = a0[:-1, 1:-1, 1:-1]  # towards -z ("down")
        aU = a0[1:, 1:-1, 1:-1]  # towards +z
        aS = a1[1:-1, :-1, 1:-1]  # towards -y
        aN = a1[1:-1, 1:, 1:-1]  # towards +y
        aW = a2[1:-1, 1:-1, :-1]  # towards -x
        aE = a2[1:-1, 1:-1, 1:]  # towards +x

        diag = (aD + aU + aS + aN + aW + aE).ravel() / h2
        rows = [idx.ravel()]
        cols = [idx.ravel()]
        vals = [diag]

        def couple(r, c, coeff):
            rows.append(r.ravel())
            cols.append(c.ravel())
            vals.append(-coeff.ravel() / h2)

        couple(idx[1:, :, :], idx[:-1, :, :], aD[1:, :, :])
        couple(idx[:-1, :, :], idx[1:, :, :], aU[:-1, :, :])
        couple(idx[:, 1:, :], idx[:, :-1, :], aS[:, 1:, :])
        couple(idx[:, :-1, :], idx[:, 1:, :], aN[:, :-1, :])
        couple(idx[:, :, 1:], idx[:, :, :-1], aW[:, :, 1:])
        couple(idx[:, :, :-1], idx[:, :, 1:], aE[:, :, :-1])

        A = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(m**3, m**3),
        )
        b = np.full(m**3, float(self.f_const))
        return A, b

    def solve(self, ic, return_info=False):
        """a(x, y, z) -> u(x, y, z); u = 0 on the boundary."""
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        a = np.asarray(ic, dtype=float)
        A, b = self._assemble(a)
        n_unknowns = A.shape[0]

        if n_unknowns <= self.params.get("_direct_max_unknowns", 30_000):
            u_int = spla.spsolve(A, b)
            info = {"solver": "spsolve"}
        else:
            # SPD system: Jacobi-preconditioned CG (3D LU fill-in is
            # prohibitive at scale).
            Minv = sp.diags(1.0 / A.diagonal())
            tol = self.params.get("_cg_tol", 1e-10)
            u_int, cg_info = spla.cg(A, b, M=Minv, rtol=tol, maxiter=20_000)
            if cg_info != 0:
                raise RuntimeError(f"CG did not converge (info={cg_info})")
            info = {"solver": "cg"}

        m = self.nz - 2
        u = np.zeros(self.field_shape)
        u[1:-1, 1:-1, 1:-1] = u_int.reshape(m, m, m)
        if return_info:
            return u, info
        return u

    def apply_operator(self, a, u):
        """Discrete -div(a grad u) on interior nodes (verification helper)."""
        h2 = self.h**2
        a0, a1, a2 = self._face_coeffs(np.asarray(a, dtype=float))
        u = np.asarray(u, dtype=float)
        c = u[1:-1, 1:-1, 1:-1]
        return (
            a0[:-1, 1:-1, 1:-1] * (c - u[:-2, 1:-1, 1:-1])
            - a0[1:, 1:-1, 1:-1] * (u[2:, 1:-1, 1:-1] - c)
            + a1[1:-1, :-1, 1:-1] * (c - u[1:-1, :-2, 1:-1])
            - a1[1:-1, 1:, 1:-1] * (u[1:-1, 2:, 1:-1] - c)
            + a2[1:-1, 1:-1, :-1] * (c - u[1:-1, 1:-1, :-2])
            - a2[1:-1, 1:-1, 1:] * (u[1:-1, 1:-1, 2:] - c)
        ) / h2

    def validate_solution(self, ic, solution, tol=1e-6):
        is_valid = (
            not np.isnan(solution).any()
            and not np.isinf(solution).any()
            and np.abs(solution).max() < 1e6
        )
        return {"valid": is_valid, "max_value": float(np.abs(solution).max())}
