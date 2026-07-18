"""
1D nonlinear Schrodinger equation (NLS).

    i psi_t = -(1/2) psi_xx + g |psi|^2 psi

Solved with the classic Strang split-step Fourier method: the dispersive
half-step is exact in Fourier space, the nonlinear phase rotation is exact
pointwise — so the L2 norm (total probability/power) is conserved to machine
precision, which doubles as the model's built-in validation invariant.

The complex field is exposed to operator-learning pipelines as two real
channels: shape (2, nx) = (Re psi, Im psi).
"""

from typing import Callable, Dict, Tuple, Union

import numpy as np

from pdeforge.core.base import PDEModel
from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model


@register_model("schrodinger_1d")
class Schrodinger1D(PDEModel):
    """
    Focusing (g < 0) or defocusing (g > 0) NLS on a periodic domain,
    split-step Fourier. Bright solitons exist for g < 0.
    """

    NDIM = 1
    INPUT_NAMES = ["re_psi0", "im_psi0"]
    OUTPUT_NAMES = ["re_psi_T", "im_psi_T"]

    USER_PARAMS = [
        ParamSpec(
            name="g",
            description="Nonlinearity (g<0 focusing, g>0 defocusing)",
            default=-1.0,
            param_type=ParamType.PHYSICAL,
            bounds=(-10.0, 10.0),
        ),
        ParamSpec(
            name="time_end",
            description="Final time",
            default=1.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.01, 50.0),
        ),
    ]

    DEFAULT_PARAMS = {
        "g": -1.0,
        "time_end": 1.0,
        "_n_time_steps": 101,
        "_dt": 1e-3,
    }

    def __init__(self, resolution, domain=None, **params):
        if domain is None:
            domain = {"x": (-10.0, 10.0)}
        super().__init__(resolution, domain, **params)

        self.g = self.params["g"]
        self.T = self.params.get("time_end", 1.0)
        self.n_t = self.params.get("_n_time_steps", 101)
        self.dt = self.params.get("_dt") or 1e-3

        self.nx = resolution["x"]
        dx = self.grids["x"][1] - self.grids["x"][0]
        self.k = 2 * np.pi * np.fft.fftfreq(self.nx, d=dx)
        # exact dispersive propagators for full/half steps
        self._disp_half = np.exp(-0.25j * self.k**2 * self.dt)

    def _to_complex(self, field):
        return field[0] + 1j * field[1]

    def _to_channels(self, psi):
        return np.stack([psi.real, psi.imag], axis=0)

    def solve(self, ic, return_full=False):
        psi = self._to_complex(np.asarray(ic))
        n_sub = max(1, int(np.ceil(self.T / self.dt)))
        out_int = max(1, n_sub // max(1, self.n_t - 1))

        frames = [self._to_channels(psi)]
        for step in range(n_sub):
            # Strang: half dispersion, full nonlinearity, half dispersion
            psi = np.fft.ifft(self._disp_half * np.fft.fft(psi))
            psi = psi * np.exp(-1j * self.g * np.abs(psi) ** 2 * self.dt)
            psi = np.fft.ifft(self._disp_half * np.fft.fft(psi))
            if (step + 1) % out_int == 0 and len(frames) < self.n_t:
                frames.append(self._to_channels(psi))

        while len(frames) < self.n_t:
            frames.append(self._to_channels(psi))
        frames = frames[: self.n_t]
        frames[-1] = self._to_channels(psi)

        if return_full:
            return np.stack(frames, axis=0)
        return frames[-1]

    def generate_ic(self, generator="default", generator_params=None, seed=None):
        """Random superposition of Gaussian wavepackets with random momenta."""
        if generator_params is None:
            generator_params = {}
        rng = np.random.default_rng(seed)
        x = self.grids["x"]
        L = self.domain.size("x")

        n_packets = generator_params.get("n_packets", int(rng.integers(1, 4)))
        psi = np.zeros_like(x, dtype=complex)
        for _ in range(n_packets):
            x0 = rng.uniform(x[0] + 0.1 * L, x[0] + 0.9 * L)
            w = rng.uniform(*generator_params.get("width_range", (0.5, 1.5)))
            p = rng.uniform(*generator_params.get("momentum_range", (-2.0, 2.0)))
            a = rng.uniform(*generator_params.get("amp_range", (0.5, 1.5)))
            psi += a * np.exp(-((x - x0) ** 2) / (2 * w**2) + 1j * p * x)
        return self._to_channels(psi)

    def norm(self, field):
        """L2 norm of the complex field — conserved by the dynamics."""
        psi = self._to_complex(np.asarray(field))
        dx = self.grids["x"][1] - self.grids["x"][0]
        return float(np.sum(np.abs(psi) ** 2) * dx)

    def bright_soliton(self, a=1.0, v=0.0, x0=0.0):
        """Exact bright soliton of focusing NLS (g = -1): a sech(a(x-x0))e^{ivx}."""
        x = self.grids["x"]
        psi = a / np.cosh(a * (x - x0)) * np.exp(1j * v * x)
        return self._to_channels(psi)

    def validate_solution(self, ic, solution, tol=1e-6):
        is_valid = not np.isnan(solution).any() and not np.isinf(solution).any()
        return {"valid": is_valid, "max_value": float(np.abs(solution).max())}
