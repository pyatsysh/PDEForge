"""
The semi-linear spectral solver seam.

A large family of PDEs has the form

    u_t = L u + N(u)

with a stiff linear part L that is diagonal in Fourier space and a mild
nonlinearity N. ETDRK4 (Cox-Matthews, with the Kassam-Trefethen contour trick
for the phi-coefficients) integrates L exactly and steps only N explicitly —
robust for diffusion-dominated, dispersive, and pattern-forming dynamics alike.

Models declare a SPEC — `linear_symbol()` and `nonlinear_hat(v, u, ops)` —
written against the small ops surface in pdeforge.solvers.ops. The same spec
runs on the NumPy engine (here) and the vmapped JAX engine
(pdeforge.solvers.engine_jax) without change. ETDRK4 coefficients are always
precomputed in NumPy on the host and shipped to the engine.
"""

from typing import Dict, Tuple

import numpy as np

from pdeforge.core.base import PDEModel
from pdeforge.solvers.ops import get_ops


def etdrk4_coeffs(L, dt, n_contour=32):
    """
    ETDRK4 coefficients for a (possibly complex) diagonal linear symbol L.

    Uses the Kassam & Trefethen (2005) contour-integral evaluation of the
    phi-functions, which is stable for |L*dt| both tiny and large. Returns a
    dict of arrays with the same shape as L: E, E2 (full/half-step
    propagators) and Q, f1, f2, f3 (nonlinear weights).
    """
    L = np.asarray(L)
    z = L * dt

    E = np.exp(z)
    E2 = np.exp(z / 2.0)

    # Contour points around each z (unit circle; complex-safe).
    theta = (np.arange(1, n_contour + 1) - 0.5) * np.pi / n_contour
    ring = np.exp(1j * theta)  # (M,)

    r = z[..., None] + ring  # (..., M)
    er = np.exp(r)

    Q = dt * np.mean((np.exp(r / 2.0) - 1.0) / r, axis=-1)
    f1 = dt * np.mean((-4.0 - r + er * (4.0 - 3.0 * r + r**2)) / r**3, axis=-1)
    f2 = dt * np.mean((2.0 + r + er * (-2.0 + r)) / r**3, axis=-1)
    f3 = dt * np.mean((-4.0 - 3.0 * r - r**2 + er * (4.0 - r)) / r**3, axis=-1)

    if np.isrealobj(L):
        Q, f1, f2, f3 = Q.real, f1.real, f2.real, f3.real

    return {"E": E, "E2": E2, "Q": Q, "f1": f1, "f2": f2, "f3": f3}


def etdrk4_step(v, u_of, N_of, C, ops):
    """
    One ETDRK4 step in spectral space.

    v: spectral state; u_of(v): inverse transform to real space;
    N_of(v, u): spectral nonlinear term; C: coefficient dict; ops: backend ops.
    """
    Nv = N_of(v, u_of(v))
    a = C["E2"] * v + C["Q"] * Nv
    Na = N_of(a, u_of(a))
    b = C["E2"] * v + C["Q"] * Na
    Nb = N_of(b, u_of(b))
    c = C["E2"] * a + C["Q"] * (2.0 * Nb - Nv)
    Nc = N_of(c, u_of(c))
    return C["E"] * v + Nv * C["f1"] + 2.0 * (Na + Nb) * C["f2"] + Nc * C["f3"]


class SemiLinearSpectralModel(PDEModel):
    """
    Base class for semi-linear spectral models: u_t = L u + N(u).

    Subclass contract
    -----------------
    - call ``self._setup_spectral()`` after ``super().__init__`` (builds the
      wavenumber grids, dealias mask, and spatial-axes bookkeeping);
    - set ``self.T`` (horizon), ``self.dt`` (substep), ``self.n_t`` (frames);
    - implement ``linear_symbol(self) -> ndarray`` over the field shape
      (leading component axis allowed for diagonal-in-component systems);
    - implement ``nonlinear_hat(self, v, u, ops) -> ndarray | None`` returning
      the SPECTRAL nonlinear term (None for purely linear problems). Use
      ``self._fft(u, ops)`` / ``self._ifft(v, ops)`` and multiply by
      ``self.dealias`` where aliasing matters.

    The solve loop, frame sampling, and backend dispatch live here; the same
    spec runs unchanged on the JAX engine (``backend="jax"``).
    """

    BACKENDS = {"numpy", "jax"}

    def _setup_spectral(self):
        # Reverse-sorted dims → array axes (matches cahn_hilliard convention:
        # 2D -> (ny, nx), 3D -> (nz, ny, nx)).
        self.dim_order = sorted(self.resolution.keys())[::-1]
        self.field_shape = tuple(self.resolution[d] for d in self.dim_order)
        self.ndim_space = len(self.field_shape)
        self.spatial_axes = tuple(range(-self.ndim_space, 0))

        ks = []
        for d in self.dim_order:
            grid = self.grids[d]
            dx = grid[1] - grid[0]
            ks.append(2 * np.pi * np.fft.fftfreq(self.resolution[d], d=dx))
        self.K = np.meshgrid(*ks, indexing="ij")
        self.K2 = sum(Ki**2 for Ki in self.K)

        # 2/3-rule dealias mask over the field shape.
        mask = np.ones(self.field_shape, dtype=bool)
        for axis, d in enumerate(self.dim_order):
            n = self.resolution[d]
            k_index = np.fft.fftfreq(n) * n  # integer mode numbers
            cut = n // 3
            axis_mask = np.abs(k_index) <= cut
            shape = [1] * self.ndim_space
            shape[axis] = n
            mask = mask & axis_mask.reshape(shape)
        self.dealias = mask.astype(float)

    # -- transforms over spatial axes only (component axis passes through) --

    def _fft(self, u, ops):
        return ops.fftn(u, axes=self.spatial_axes)

    def _ifft(self, v, ops):
        return ops.real(ops.ifftn(v, axes=self.spatial_axes))

    # -- spec (subclass) --

    def linear_symbol(self):
        raise NotImplementedError

    def nonlinear_hat(self, v, u, ops):
        """Spectral nonlinear term; None means purely linear."""
        return None

    # -- solve --

    def _n_substeps(self):
        # Round-half-up, NOT ceil: after dt-snapping, T/dt is an integer up
        # to float epsilon, and ceil would spuriously add a step past T.
        return max(1, int(np.floor(self.T / self.dt + 0.5)))

    def effective_dt(self):
        """dt snapped so n_substeps * dt == T exactly (no overshoot past T)."""
        return self.T / self._n_substeps()

    def solve(self, ic, return_full=False):
        if self.backend == "jax":
            from pdeforge.solvers.engine_jax import solve_jax

            return solve_jax(self, np.asarray(ic), return_full=return_full)
        return self._solve_numpy(np.asarray(ic), return_full=return_full)

    def _solve_numpy(self, ic, return_full=False):
        ops = get_ops("numpy")
        L = self.linear_symbol()
        C = etdrk4_coeffs(L, self.effective_dt())

        # Purely linear problems propagate exactly (per substep AND overall).
        linear_only = self.nonlinear_hat(self._fft(ic, ops), ic, ops) is None

        u_of = lambda v: self._ifft(v, ops)
        N_of = lambda v, u: self.nonlinear_hat(v, u, ops)

        v = self._fft(ic, ops)
        n_substeps = self._n_substeps()
        output_interval = max(1, n_substeps // max(1, self.n_t - 1))

        frames = [np.asarray(ic, dtype=float).copy()]
        for step in range(n_substeps):
            if linear_only:
                v = C["E"] * v
            else:
                v = etdrk4_step(v, u_of, N_of, C, ops)
            if (step + 1) % output_interval == 0 and len(frames) < self.n_t:
                frames.append(u_of(v))

        while len(frames) < self.n_t:
            frames.append(u_of(v))
        frames = frames[: self.n_t]
        frames[-1] = u_of(v)  # last frame is always the final state

        if return_full:
            return np.stack(frames, axis=0)
        return frames[-1]
