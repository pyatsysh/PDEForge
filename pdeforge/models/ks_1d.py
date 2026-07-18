"""
1D Kuramoto-Sivashinsky equation.

    u_t = -u u_x - u_xx - u_xxxx

The canonical chaotic PDE benchmark: the destabilising -u_xx term pumps
energy in, the hyperdiffusive -u_xxxx term drains it, and the nonlinearity
cascades between scales. On a domain of size L >~ 20 the dynamics is
spatiotemporally chaotic.

Operator learning task: u(x, 0) -> u(x, T).
"""

from typing import Callable, Dict, Tuple, Union

import numpy as np

from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model
from pdeforge.generators.initial_conditions import get_ic_generator
from pdeforge.solvers.semilinear import SemiLinearSpectralModel


@register_model("ks_1d")
class KuramotoSivashinsky1D(SemiLinearSpectralModel):
    """
    Kuramoto-Sivashinsky on a periodic domain (default size 32*pi — the
    classic chaotic setup of Kassam & Trefethen 2005).

    The linear symbol k^2 - k^4 is integrated exactly by ETDRK4; only the
    dealiased advective nonlinearity is stepped explicitly.
    """

    NDIM = 1
    INPUT_NAMES = ["u0"]
    OUTPUT_NAMES = ["u_T"]

    USER_PARAMS = [
        ParamSpec(
            name="time_horizon",
            description="Final time T (chaos develops over t ~ 50+)",
            default=50.0,
            param_type=ParamType.PHYSICAL,
            bounds=(1.0, 500.0),
        ),
    ]

    DEFAULT_PARAMS = {
        "time_horizon": 50.0,
        "_n_time_steps": 101,
        "_dt": 0.25,
    }

    def __init__(self, resolution, domain=None, **params):
        # KS needs a large box for chaos; default to the classic 32*pi.
        if domain is None:
            domain = {"x": (0.0, 32.0 * np.pi)}
        super().__init__(resolution, domain, **params)

        self.T = self.params.get("time_horizon", 50.0)
        self.n_t = self.params.get("_n_time_steps", 101)

        self._setup_spectral()
        self.nx = resolution["x"]
        self.k = self.K[0]
        self.dt = self.params.get("_dt") or 0.25

    def linear_symbol(self):
        return self.k**2 - self.k**4

    def nonlinear_hat(self, v, u, ops):
        # -u u_x = -(1/2) d/dx u^2, dealiased
        return -0.5j * self.k * (self.dealias * self._fft(u * u, ops))

    def generate_ic(self, generator="fourier", generator_params=None, seed=None):
        if generator_params is None:
            generator_params = {}
        if generator == "fourier":
            generator_params = {
                "n_modes": 6,
                "decay": 1.0,
                "amplitude": 0.5,
                **generator_params,
            }
        if isinstance(generator, str):
            gen = get_ic_generator(generator, **generator_params)
        else:
            gen = generator
        return gen.generate(shape=(self.nx,), seed=seed, grid=self.grids)

    def validate_solution(self, ic, solution, tol=1e-6):
        is_valid = (
            not np.isnan(solution).any()
            and not np.isinf(solution).any()
            and np.abs(solution).max() < 1e3  # KS stays O(1-10); blowup = bug
        )
        return {"valid": is_valid, "max_value": float(np.abs(solution).max())}
