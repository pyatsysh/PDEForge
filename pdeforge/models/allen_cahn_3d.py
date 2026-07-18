"""
3D Allen-Cahn equation on the periodic box.

    u_t = eps Laplacian(u) + u - u^3

Non-conserved phase separation in three dimensions: domains of the +-1 wells
coarsen by mean-curvature flow of their interfaces. Runs on the
dimension-agnostic seam (ETDRK4, linear reaction inside L). Mind memory for
large grids — pair with chunked-to-disk generation (to=).

Operator learning task: u(x, y, z, 0) -> u(x, y, z, T).
"""

from typing import Callable, Dict, Tuple, Union

import numpy as np

from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model
from pdeforge.solvers.semilinear import SemiLinearSpectralModel


@register_model("allen_cahn_3d")
class AllenCahn3D(SemiLinearSpectralModel):
    """3D Allen-Cahn, ETDRK4 on the seam."""

    NDIM = 3
    INPUT_NAMES = ["u0"]
    OUTPUT_NAMES = ["u_T"]

    USER_PARAMS = [
        ParamSpec(
            name="epsilon",
            description="Interface width parameter",
            default=0.01,
            param_type=ParamType.PHYSICAL,
            bounds=(0.001, 0.5),
        ),
        ParamSpec(
            name="time_end",
            description="Final time",
            default=5.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.1, 100.0),
        ),
    ]

    DEFAULT_PARAMS = {
        "epsilon": 0.01,
        "time_end": 5.0,
        "_n_time_steps": 51,
        "_dt": 0.05,
    }

    def __init__(self, resolution, domain=None, **params):
        super().__init__(resolution, domain, **params)

        self.eps = self.params["epsilon"]
        self.T = self.params.get("time_end", 5.0)
        self.n_t = self.params.get("_n_time_steps", 51)
        self.dt = self.params.get("_dt") or 0.05

        self._setup_spectral()

    def linear_symbol(self):
        return -self.eps * self.K2 + 1.0

    def nonlinear_hat(self, v, u, ops):
        return -(self.dealias * self._fft(u**3, ops))

    def generate_ic(self, generator="default", generator_params=None, seed=None):
        """Small smooth random field around zero (undecided phase)."""
        if generator_params is None:
            generator_params = {}
        rng = np.random.default_rng(seed)

        cutoff = generator_params.get("cutoff", 4)
        amp = generator_params.get("amplitude", 0.25)

        noise = rng.standard_normal(self.field_shape)
        noise_hat = np.fft.fftn(noise)
        mode_axes = [np.abs(np.fft.fftfreq(n) * n) for n in self.field_shape]
        grids = np.meshgrid(*mode_axes, indexing="ij")
        mask = np.sqrt(sum(g**2 for g in grids)) <= cutoff
        u = np.fft.ifftn(noise_hat * mask).real
        return amp * u / (np.abs(u).max() + 1e-12)

    def validate_solution(self, ic, solution, tol=1e-6):
        is_valid = (
            not np.isnan(solution).any()
            and not np.isinf(solution).any()
            and np.abs(solution).max() < 2.0
        )
        return {"valid": is_valid, "max_value": float(np.abs(solution).max())}
