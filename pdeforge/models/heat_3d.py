"""
3D heat equation on the periodic box.

    u_t = alpha Laplacian(u)

The seam is dimension-agnostic (fftn over however many spatial axes the
resolution dict declares), so this model is the 3D twin of heat_1d/heat_2d:
purely linear, propagated exactly. Mind memory: a (64, 64, 64) float64 field
is 2 MB per sample — pair large runs with chunked-to-disk generation (to=).

Operator learning task: u(x, y, z, 0) -> u(x, y, z, T).
"""

from typing import Callable, Dict, Tuple, Union

import numpy as np

from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model
from pdeforge.solvers.semilinear import SemiLinearSpectralModel


@register_model("heat_3d")
class Heat3D(SemiLinearSpectralModel):
    """3D diffusion, exact spectral propagation on the seam."""

    NDIM = 3
    INPUT_NAMES = ["u0"]
    OUTPUT_NAMES = ["u_T"]

    USER_PARAMS = [
        ParamSpec(
            name="diffusivity",
            description="Thermal diffusivity alpha",
            default=0.01,
            param_type=ParamType.PHYSICAL,
            bounds=(1e-6, 1.0),
        ),
        ParamSpec(
            name="time_end",
            description="Final time",
            default=1.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.01, 10.0),
        ),
    ]

    DEFAULT_PARAMS = {
        "diffusivity": 0.01,
        "time_end": 1.0,
        "_n_time_steps": 51,
        "_dt": None,
    }

    def __init__(self, resolution, domain=None, **params):
        super().__init__(resolution, domain, **params)

        self.alpha = self.params["diffusivity"]
        self.T = self.params.get("time_end", 1.0)
        self.n_t = self.params.get("_n_time_steps", 51)

        self._setup_spectral()
        self.dt = self.params.get("_dt") or self.T / max(1, self.n_t - 1)

    def linear_symbol(self):
        return -self.alpha * self.K2

    def generate_ic(self, generator="default", generator_params=None, seed=None):
        """Smooth random 3D field (spectrally low-passed white noise)."""
        if generator_params is None:
            generator_params = {}
        rng = np.random.default_rng(seed)

        cutoff = generator_params.get("cutoff", 4)
        amp = generator_params.get("amplitude", 1.0)

        noise = rng.standard_normal(self.field_shape)
        noise_hat = np.fft.fftn(noise)
        mode_axes = [np.abs(np.fft.fftfreq(n) * n) for n in self.field_shape]
        grids = np.meshgrid(*mode_axes, indexing="ij")
        mask = np.sqrt(sum(g**2 for g in grids)) <= cutoff
        u = np.fft.ifftn(noise_hat * mask).real
        return amp * u / (np.abs(u).max() + 1e-12)

    def validate_solution(self, ic, solution, tol=1e-6):
        is_valid = not np.isnan(solution).any() and not np.isinf(solution).any()
        return {"valid": is_valid, "max_value": float(np.abs(solution).max())}
