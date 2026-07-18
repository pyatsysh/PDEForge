"""
2D wave equation with a heterogeneous (spatially varying) wave speed.

    u_tt = c(x, y)^2 Laplacian(u)

The INPUT is the wave-speed field c(x, y); a fixed, seeded Gaussian pulse is
propagated through it and the OUTPUT is the wavefield at time T. This is the
canonical inverse-problem-ready operator task (think travel-time tomography):
learn the map medium -> wavefield.

Discretisation: pseudo-spectral Laplacian + leapfrog (Stormer-Verlet) in
time. For constant c the scheme's dispersion matches the exact spectral
propagator to O(dt^2) — used as a validation invariant against wave_2d.
"""

from typing import Callable, Dict, Tuple, Union

import numpy as np

from pdeforge.core.base import PDEModel
from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model


@register_model("heterogeneous_wave_2d")
class HeterogeneousWave2D(PDEModel):
    """
    Wave propagation through a random medium on the periodic box.

    Operator task: c(x, y) -> u(x, y, T). The source pulse is fixed by
    `pulse_*` params (same pulse for every sample), so the only varying
    input is the medium.
    """

    NDIM = 2
    INPUT_NAMES = ["c"]
    OUTPUT_NAMES = ["u_T"]

    USER_PARAMS = [
        ParamSpec(
            name="c_min",
            description="Minimum wave speed",
            default=0.5,
            param_type=ParamType.PHYSICAL,
            bounds=(0.05, 10.0),
        ),
        ParamSpec(
            name="c_max",
            description="Maximum wave speed",
            default=1.5,
            param_type=ParamType.PHYSICAL,
            bounds=(0.1, 20.0),
        ),
        ParamSpec(
            name="time_end",
            description="Propagation time",
            default=0.3,
            param_type=ParamType.PHYSICAL,
            bounds=(0.01, 10.0),
        ),
        ParamSpec(
            name="pulse_width",
            description="Width of the fixed source pulse",
            default=0.05,
            param_type=ParamType.INPUT,
            bounds=(0.005, 0.5),
        ),
    ]

    DEFAULT_PARAMS = {
        "c_min": 0.5,
        "c_max": 1.5,
        "time_end": 0.3,
        "pulse_width": 0.05,
        "pulse_x": 0.5,
        "pulse_y": 0.5,
        "_n_time_steps": 101,
        "_dt": None,
    }

    def __init__(self, resolution, domain=None, **params):
        super().__init__(resolution, domain, **params)

        p = self.params
        self.c_min, self.c_max = p["c_min"], p["c_max"]
        self.T = p.get("time_end", 0.3)
        self.n_t = p.get("_n_time_steps", 101)

        self.nx, self.ny = resolution["x"], resolution["y"]
        dx = self.grids["x"][1] - self.grids["x"][0]
        dy = self.grids["y"][1] - self.grids["y"][0]
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ny, d=dy)
        self.KX, self.KY = np.meshgrid(kx, ky)
        self.K2 = self.KX**2 + self.KY**2

        # Leapfrog CFL against the maximum admissible speed, then snapped so
        # n_substeps * dt == T exactly (no overshoot past the target time).
        kmax = np.sqrt(self.K2.max())
        dt0 = p.get("_dt") or 0.5 * 2.0 / (self.c_max * kmax)
        self._n_sub = max(1, int(np.ceil(self.T / dt0)))
        self.dt = self.T / self._n_sub  # snapped; solve uses the STORED count

        # The fixed source pulse (same for every sample).
        X, Y = np.meshgrid(self.grids["x"], self.grids["y"])
        w = p.get("pulse_width", 0.05)
        x0, y0 = p.get("pulse_x", 0.5), p.get("pulse_y", 0.5)
        self._pulse = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * w**2))

    def _laplacian(self, u):
        return np.fft.ifft2(-self.K2 * np.fft.fft2(u)).real

    def solve(self, ic, return_full=False):
        """ic IS the medium c(x, y); the pulse is the fixed initial u."""
        c2 = np.asarray(ic, dtype=float) ** 2
        dt = self.dt
        n_sub = self._n_sub  # stored at init; never re-derived via ceil
        out_int = max(1, n_sub // max(1, self.n_t - 1))

        u = self._pulse.copy()
        # first leapfrog step from rest (u_t(0) = 0)
        u_prev = u.copy()
        u_next = u + 0.5 * dt**2 * c2 * self._laplacian(u)

        frames = [u.copy()]
        u_prev, u = u, u_next
        for step in range(1, n_sub):
            u_next = 2 * u - u_prev + dt**2 * c2 * self._laplacian(u)
            u_prev, u = u, u_next
            if (step + 1) % out_int == 0 and len(frames) < self.n_t:
                frames.append(u.copy())

        while len(frames) < self.n_t:
            frames.append(u.copy())
        frames = frames[: self.n_t]
        frames[-1] = u

        if return_full:
            return np.stack(frames, axis=0)
        return frames[-1]

    def generate_ic(self, generator="default", generator_params=None, seed=None):
        """Random smooth medium c(x,y) in [c_min, c_max] (low-pass field)."""
        if generator_params is None:
            generator_params = {}
        rng = np.random.default_rng(seed)

        cutoff = generator_params.get("cutoff", 4)
        noise_hat = np.fft.fft2(rng.standard_normal((self.ny, self.nx)))
        kyi = np.fft.fftfreq(self.ny) * self.ny
        kxi = np.fft.fftfreq(self.nx) * self.nx
        KXi, KYi = np.meshgrid(kxi, kyi)
        mask = np.sqrt(KXi**2 + KYi**2) <= cutoff
        f = np.fft.ifft2(noise_hat * mask).real
        f = (f - f.min()) / (f.max() - f.min() + 1e-12)
        return self.c_min + (self.c_max - self.c_min) * f

    def energy(self, u_prev, u, c2):
        """Discrete wave energy (monitoring helper)."""
        ut = (u - u_prev) / self.dt
        gx = np.fft.ifft2(1j * self.KX * np.fft.fft2(u)).real
        gy = np.fft.ifft2(1j * self.KY * np.fft.fft2(u)).real
        return float(np.mean(ut**2 / c2 + gx**2 + gy**2))

    def validate_solution(self, ic, solution, tol=1e-6):
        is_valid = (
            not np.isnan(solution).any()
            and not np.isinf(solution).any()
            and np.abs(solution).max() < 1e6
        )
        return {"valid": is_valid, "max_value": float(np.abs(solution).max())}
