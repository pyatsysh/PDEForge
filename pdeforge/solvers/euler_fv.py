"""
Finite-volume compressible Euler on a structured body-fitted mesh.

    d/dt (rho, rho u, rho v, rho E) + div F = 0

Cell-centred, HLLC fluxes (Toro), MUSCL reconstruction with a minmod limiter,
and SSP-RK2 advanced with a LOCAL time step — the mesh spans five orders of
magnitude in cell size between the wall and the far field, so a global step
would make steady state unreachable.

This is the shock-capturing seam the spectral solver cannot provide: transonic
flow puts a supersonic pocket terminated by a shock on the upper surface, and
a Fourier method would ring across it.

Boundary conditions on the C-grid topology:

- eta = 0 over the airfoil: slip wall, imposed by mirroring the velocity of
  the adjacent cells about the wall normal.
- eta = 0 over the wake cut: the two sides are geometrically coincident and
  the flow is continuous across them, so the ghost cell is simply the partner
  cell on the other side. The Kutta condition then emerges from the sharp
  trailing edge rather than being enforced.
- eta = max and both xi ends: characteristic far field built from Riemann
  invariants, with the Thomas-Salas point-vortex correction (see
  ``_freestream_at``).
"""

from typing import Dict, Tuple

import numpy as np

GAMMA = 1.4
NG = 2  # ghost layers (two, so MUSCL slopes exist in the first ghost)


def mesh_metrics(X: np.ndarray, Y: np.ndarray) -> Dict[str, np.ndarray]:
    """Cell volumes, centres, and unit face normals with face lengths."""
    x1, y1 = X[:-1, :-1], Y[:-1, :-1]
    x2, y2 = X[1:, :-1], Y[1:, :-1]
    x3, y3 = X[1:, 1:], Y[1:, 1:]
    x4, y4 = X[:-1, 1:], Y[:-1, 1:]
    vol = 0.5 * np.abs(
        (x1 * y2 - x2 * y1)
        + (x2 * y3 - x3 * y2)
        + (x3 * y4 - x4 * y3)
        + (x4 * y1 - x1 * y4)
    )
    xc = 0.25 * (x1 + x2 + x3 + x4)
    yc = 0.25 * (y1 + y2 + y3 + y4)

    ex, ey = X[:, 1:] - X[:, :-1], Y[:, 1:] - Y[:, :-1]
    si = np.hypot(ex, ey)
    nix, niy = ey / si, -ex / si

    fx, fy = X[1:, :] - X[:-1, :], Y[1:, :] - Y[:-1, :]
    sj = np.hypot(fx, fy)
    njx, njy = -fy / sj, fx / sj

    # Point the normals along increasing xi / eta (the sign depends on how the
    # mesh was traversed, so it is measured rather than assumed).
    if np.sum((xc[1:] - xc[:-1]) * nix[1:-1] + (yc[1:] - yc[:-1]) * niy[1:-1]) < 0:
        nix, niy = -nix, -niy
    if (
        np.sum(
            (xc[:, 1:] - xc[:, :-1]) * njx[:, 1:-1]
            + (yc[:, 1:] - yc[:, :-1]) * njy[:, 1:-1]
        )
        < 0
    ):
        njx, njy = -njx, -njy

    return dict(vol=vol, xc=xc, yc=yc, si=si, nix=nix, niy=niy, sj=sj, njx=njx, njy=njy)


def to_primitive(U):
    r = np.maximum(U[0], 1e-9)
    u, v = U[1] / r, U[2] / r
    p = np.maximum((GAMMA - 1.0) * (U[3] - 0.5 * r * (u * u + v * v)), 1e-9)
    return r, u, v, p


def to_conservative(r, u, v, p):
    return np.stack([r, r * u, r * v, p / (GAMMA - 1.0) + 0.5 * r * (u * u + v * v)])


def _safe(x):
    return np.where(np.abs(x) < 1e-30, 1e-30, x)


def hllc_flux(WL, WR, nx, ny):
    """HLLC flux through faces with unit normals (nx, ny)."""
    rL, uL, vL, pL = WL
    rR, uR, vR, pR = WR
    qL, qR = uL * nx + vL * ny, uR * nx + vR * ny
    cL, cR = np.sqrt(GAMMA * pL / rL), np.sqrt(GAMMA * pR / rR)

    SL = np.minimum(qL - cL, qR - cR)
    SR = np.maximum(qL + cL, qR + cR)
    SM = (pR - pL + rL * qL * (SL - qL) - rR * qR * (SR - qR)) / _safe(
        rL * (SL - qL) - rR * (SR - qR)
    )

    EL = pL / (GAMMA - 1.0) + 0.5 * rL * (uL * uL + vL * vL)
    ER = pR / (GAMMA - 1.0) + 0.5 * rR * (uR * uR + vR * vR)

    def flux(r, u, v, p, q, E):
        return np.stack([r * q, r * u * q + p * nx, r * v * q + p * ny, (E + p) * q])

    def star(r, u, v, p, q, E, S):
        rs = r * (S - q) / _safe(S - SM)
        return np.stack(
            [
                rs,
                rs * (u + (SM - q) * nx),
                rs * (v + (SM - q) * ny),
                rs * (E / r + (SM - q) * (SM + p / _safe(r * (S - q)))),
            ]
        )

    UL = np.stack([rL, rL * uL, rL * vL, EL])
    UR = np.stack([rR, rR * uR, rR * vR, ER])
    FL = flux(rL, uL, vL, pL, qL, EL)
    FR = flux(rR, uR, vR, pR, qR, ER)

    return np.where(
        SL >= 0.0,
        FL,
        np.where(
            SM >= 0.0,
            FL + SL * (star(rL, uL, vL, pL, qL, EL, SL) - UL),
            np.where(SR >= 0.0, FR + SR * (star(rR, uR, vR, pR, qR, ER, SR) - UR), FR),
        ),
    )


def minmod(a, b):
    return np.where(a * b <= 0.0, 0.0, np.where(np.abs(a) < np.abs(b), a, b))


class EulerCGrid:
    """Steady transonic Euler solve on an airfoil C-grid."""

    def __init__(
        self,
        X,
        Y,
        n_wall,
        n_wake,
        mach=0.8,
        aoa_deg=1.25,
        cfl=0.7,
        order=2,
        vortex_correction=True,
    ):
        self.g = mesh_metrics(X, Y)
        self.ni, self.nj = self.g["vol"].shape
        self.order = order
        self.cfl = cfl
        self.vortex_correction = vortex_correction

        a = np.radians(aoa_deg)
        self.mach = mach
        self.aoa = a
        # Non-dimensionalised so that c_inf = 1 and |V_inf| = M_inf.
        self.r_inf, self.p_inf = 1.0, 1.0 / GAMMA
        self.u_inf, self.v_inf = mach * np.cos(a), mach * np.sin(a)
        self.q_inf = 0.5 * self.r_inf * mach**2
        self.circulation = 0.0

        shape = (self.ni + 2 * NG, self.nj + 2 * NG)
        self.U = to_conservative(
            np.full(shape, self.r_inf),
            np.full(shape, self.u_inf),
            np.full(shape, self.v_inf),
            np.full(shape, self.p_inf),
        )

        self.wall_lo = n_wake
        self.wall_hi = n_wake + n_wall - 1

    @property
    def interior(self):
        return self.U[:, NG:-NG, NG:-NG]

    # ------------------------------------------------------------ boundaries
    def _freestream_at(self, xb, yb):
        """
        Freestream with the Thomas-Salas point-vortex correction.

        A lifting airfoil's far field is not uniform: it decays like a point
        vortex of circulation Gamma = 0.5 |V| c C_l. Imposing a uniform
        freestream at 20 chords loses a few percent of the lift, so the vortex
        is added back rather than pushing the boundary out and paying for the
        extra mesh.
        """
        if not self.vortex_correction or self.circulation == 0.0:
            return self.u_inf, self.v_inf
        dx, dy = xb - 0.25, yb  # quarter chord
        r = np.maximum(np.hypot(dx, dy), 1e-9)
        th = np.arctan2(dy, dx)
        beta = np.sqrt(max(1.0 - self.mach**2, 1e-6))
        vth = (
            self.circulation
            * beta
            / (2.0 * np.pi * r * (1.0 - self.mach**2 * np.sin(th - self.aoa) ** 2))
        )
        return self.u_inf + vth * np.sin(th), self.v_inf - vth * np.cos(th)

    def _farfield(self, Wi, nx, ny, xb, yb):
        """Riemann-invariant far-field state from the interior state."""
        r, u, v, p = Wi
        c = np.sqrt(GAMMA * p / r)
        qn = u * nx + v * ny
        c_inf = np.sqrt(GAMMA * self.p_inf / self.r_inf)
        u_ff, v_ff = self._freestream_at(xb, yb)
        qn_inf = u_ff * nx + v_ff * ny

        Rp = qn + 2.0 * c / (GAMMA - 1.0)  # invariant carried from inside
        Rm = qn_inf - 2.0 * c_inf / (GAMMA - 1.0)  # and from outside
        qb = 0.5 * (Rp + Rm)
        cb = 0.25 * (GAMMA - 1.0) * (Rp - Rm)

        inflow = qb <= 0.0
        ut = np.where(inflow, u_ff - qn_inf * nx, u - qn * nx)
        vt = np.where(inflow, v_ff - qn_inf * ny, v - qn * ny)
        s = np.where(inflow, self.p_inf / self.r_inf**GAMMA, p / r**GAMMA)

        rb = np.maximum(cb * cb / (GAMMA * s), 1e-12) ** (1.0 / (GAMMA - 1.0))
        pb = rb * cb * cb / GAMMA
        return np.stack([rb, ut + qb * nx, vt + qb * ny, pb])

    def apply_bc(self):
        U, ni = self.U, self.ni
        nx, ny = self.g["njx"][:, 0], self.g["njy"][:, 0]

        for gl in range(1, NG + 1):
            r, u, v, p = to_primitive(U[:, NG:-NG, NG + gl - 1])
            vn = u * nx + v * ny
            U[:, NG:-NG, NG - gl] = to_conservative(
                r, u - 2.0 * vn * nx, v - 2.0 * vn * ny, p
            )

        # wake cut: coincident sides, so the ghost is the partner cell
        idx = np.arange(ni)
        partner = ni - 1 - idx
        cut = (idx < self.wall_lo) | (idx >= self.wall_hi)
        for gl in range(1, NG + 1):
            ghost = U[:, NG:-NG, NG + gl - 1][:, partner]
            U[:, NG:-NG, NG - gl][:, cut] = ghost[:, cut]

        gx, gy = self.g["xc"], self.g["yc"]
        nxo, nyo = self.g["njx"][:, -1], self.g["njy"][:, -1]
        nx0, ny0 = -self.g["nix"][0], -self.g["niy"][0]
        nxN, nyN = self.g["nix"][-1], self.g["niy"][-1]
        for gl in range(1, NG + 1):
            Wi = to_primitive(U[:, NG:-NG, -NG - gl])
            U[:, NG:-NG, -NG - 1 + gl] = to_conservative(
                *self._farfield(Wi, nxo, nyo, gx[:, -1], gy[:, -1])
            )
            Wi = to_primitive(U[:, NG + gl - 1, NG:-NG])
            U[:, NG - gl, NG:-NG] = to_conservative(
                *self._farfield(Wi, nx0, ny0, gx[0], gy[0])
            )
            Wi = to_primitive(U[:, -NG - gl, NG:-NG])
            U[:, -NG - 1 + gl, NG:-NG] = to_conservative(
                *self._farfield(Wi, nxN, nyN, gx[-1], gy[-1])
            )

    # -------------------------------------------------------------- residual
    def residual(self):
        self.apply_bc()
        W = np.stack(to_primitive(self.U))
        ni, nj = self.ni, self.nj

        # face f (0..ni) lies between cells NG+f-1 and NG+f
        iL, iR = slice(NG - 1, NG + ni), slice(NG, NG + ni + 1)
        jL, jR = slice(NG - 1, NG + nj), slice(NG, NG + nj + 1)
        ii, jj = slice(NG, NG + ni), slice(NG, NG + nj)

        WLi, WRi = W[:, iL, jj], W[:, iR, jj]
        WLj, WRj = W[:, ii, jL], W[:, ii, jR]

        if self.order > 1:
            d = W[:, 1:] - W[:, :-1]
            s = minmod(d[:, :-1], d[:, 1:])  # s[m] is the slope of cell m+1
            WLi = WLi + 0.5 * s[:, NG - 2 : NG + ni - 1, jj]
            WRi = WRi - 0.5 * s[:, NG - 1 : NG + ni, jj]

            d = W[:, :, 1:] - W[:, :, :-1]
            s = minmod(d[:, :, :-1], d[:, :, 1:])
            WLj = WLj + 0.5 * s[:, ii, NG - 2 : NG + nj - 1]
            WRj = WRj - 0.5 * s[:, ii, NG - 1 : NG + nj]

            for w in (WLi, WRi, WLj, WRj):  # reconstruction must not vacuum
                np.maximum(w[0], 1e-9, out=w[0])
                np.maximum(w[3], 1e-9, out=w[3])

        Fi = hllc_flux(WLi, WRi, self.g["nix"], self.g["niy"]) * self.g["si"]
        Fj = hllc_flux(WLj, WRj, self.g["njx"], self.g["njy"]) * self.g["sj"]
        return -(Fi[:, 1:] - Fi[:, :-1] + Fj[:, :, 1:] - Fj[:, :, :-1]) / self.g["vol"]

    def local_dt(self):
        r, u, v, p = to_primitive(self.interior)
        c = np.sqrt(GAMMA * p / r)
        g = self.g
        si = 0.5 * (g["si"][1:] + g["si"][:-1])
        sj = 0.5 * (g["sj"][:, 1:] + g["sj"][:, :-1])
        nix = 0.5 * (g["nix"][1:] + g["nix"][:-1])
        niy = 0.5 * (g["niy"][1:] + g["niy"][:-1])
        njx = 0.5 * (g["njx"][:, 1:] + g["njx"][:, :-1])
        njy = 0.5 * (g["njy"][:, 1:] + g["njy"][:, :-1])
        lam = (np.abs(u * nix + v * niy) + c) * si + (
            np.abs(u * njx + v * njy) + c
        ) * sj
        return self.cfl * g["vol"] / lam

    def solve(self, iters=12000, tol=1e-5, verbose=False):
        """March to steady state; returns the residual drop achieved."""
        r0, drop = None, 1.0
        for it in range(iters):
            U0 = self.interior.copy()
            dt = self.local_dt()
            R = self.residual()
            self.interior[:] = U0 + dt * R
            self.interior[:] = 0.5 * (U0 + self.interior + dt * self.residual())

            if self.vortex_correction and it > 200 and it % 50 == 0:
                cl, _ = self.force_coefficients()
                self.circulation = 0.5 * self.mach * cl

            if it % 100 == 0 or it == iters - 1:
                res = float(np.sqrt(np.mean(R[0] ** 2)))
                if not np.isfinite(res):
                    raise FloatingPointError(f"Euler solve diverged at step {it}")
                r0 = res if r0 is None else r0
                drop = res / r0
                if verbose and it % 2000 == 0:
                    print(f"   it {it:6d}  residual {res:.3e}  drop {drop:.2e}")
                if drop < tol:
                    break
        self.residual_drop = drop
        return drop

    # ------------------------------------------------------------------ post
    def fields(self) -> Tuple[np.ndarray, ...]:
        return to_primitive(self.interior)

    def mach_field(self) -> np.ndarray:
        r, u, v, p = self.fields()
        return np.hypot(u, v) / np.sqrt(GAMMA * p / r)

    def wall_slice(self):
        return slice(self.wall_lo, self.wall_hi)

    def surface_cp(self):
        """(x, y, C_p) along the airfoil surface."""
        w = self.wall_slice()
        _, _, _, p = to_primitive(self.interior[:, w, 0])
        return self.g["xc"][w, 0], self.g["yc"][w, 0], (p - self.p_inf) / self.q_inf

    def force_coefficients(self):
        """
        Inviscid C_l, C_d from the wall pressure integral.

        The j-face normal at eta = 0 points from the wall into the fluid, so
        it IS the body's outward normal and the force on the body is
        -sum p n dS. Gauge pressure is used: the constant part integrates to
        zero around a closed body, and subtracting it avoids cancelling large
        numbers.
        """
        w = self.wall_slice()
        _, _, _, p = to_primitive(self.interior[:, w, 0])
        nx, ny, ds = self.g["njx"][w, 0], self.g["njy"][w, 0], self.g["sj"][w, 0]
        fx = -np.sum((p - self.p_inf) * nx * ds)
        fy = -np.sum((p - self.p_inf) * ny * ds)
        cl = (fy * np.cos(self.aoa) - fx * np.sin(self.aoa)) / self.q_inf
        cd = (fx * np.cos(self.aoa) + fy * np.sin(self.aoa)) / self.q_inf
        return float(cl), float(cd)
