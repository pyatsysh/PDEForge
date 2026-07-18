"""
1D Korteweg-de Vries equation.

    u_t + 6 u u_x + u_xxx = 0

The canonical dispersive-soliton PDE: sech^2 solitary waves propagate with
speed proportional to their amplitude and interact elastically.

Operator learning task: u(x, 0) -> u(x, T).
"""

from typing import Callable, Dict, Tuple, Union

import numpy as np

from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model
from pdeforge.generators.initial_conditions import get_ic_generator
from pdeforge.solvers.semilinear import SemiLinearSpectralModel


@register_model("kdv_1d")
class KdV1D(SemiLinearSpectralModel):
    """
    KdV on a periodic domain (default [0, 20]). The dispersive u_xxx term
    (symbol i k^3) is integrated exactly by ETDRK4 — the stiffness that
    cripples explicit schemes never appears; only 6 u u_x is stepped.
    """

    NDIM = 1
    INPUT_NAMES = ["u0"]
    OUTPUT_NAMES = ["u_T"]

    USER_PARAMS = [
        ParamSpec(
            name="time_end",
            description="Final time",
            default=1.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.01, 50.0),
        ),
    ]

    DEFAULT_PARAMS = {
        "time_end": 1.0,
        "_n_time_steps": 101,
        "_dt": 2e-4,
    }

    def __init__(self, resolution, domain=None, **params):
        if domain is None:
            domain = {"x": (0.0, 20.0)}
        super().__init__(resolution, domain, **params)

        self.T = self.params.get("time_end", 1.0)
        self.n_t = self.params.get("_n_time_steps", 101)

        self._setup_spectral()
        self.nx = resolution["x"]
        self.k = self.K[0]
        self.dt = self.params.get("_dt") or 2e-4

    def linear_symbol(self):
        # u_t = -u_xxx -> L = -(i k)^3 = i k^3
        return 1j * self.k**3

    def nonlinear_hat(self, v, u, ops):
        # -6 u u_x = -3 d/dx u^2
        return -3j * self.k * (self.dealias * self._fft(u * u, ops))

    def generate_ic(self, generator="solitons", generator_params=None, seed=None):
        """
        Default IC: superposition of 1-3 well-separated solitons with random
        speeds — the natural input measure for KdV. "fourier" is available
        for generic smooth fields.
        """
        if generator_params is None:
            generator_params = {}

        if generator == "solitons":
            rng = np.random.default_rng(seed)
            x = self.grids["x"]
            L = self.domain.size("x")
            n_sol = generator_params.get("n_solitons", int(rng.integers(1, 4)))
            u0 = np.zeros_like(x)
            for _ in range(n_sol):
                c = rng.uniform(*generator_params.get("speed_range", (1.0, 6.0)))
                x0 = rng.uniform(0.0, L)
                # periodic sech^2 soliton: u = c/2 sech^2(sqrt(c)/2 (x-x0))
                dxp = (x - x0 + L / 2) % L - L / 2
                u0 += 0.5 * c / np.cosh(0.5 * np.sqrt(c) * dxp) ** 2
            return u0

        if isinstance(generator, str):
            gen = get_ic_generator(generator, **generator_params)
        else:
            gen = generator
        return gen.generate(shape=(self.nx,), seed=seed, grid=self.grids)

    def soliton(self, c, x0=None):
        """Exact single-soliton profile u(x) = c/2 sech^2(sqrt(c)/2 (x-x0))."""
        x = self.grids["x"]
        L = self.domain.size("x")
        x0 = L / 2 if x0 is None else x0
        dxp = (x - x0 + L / 2) % L - L / 2
        return 0.5 * c / np.cosh(0.5 * np.sqrt(c) * dxp) ** 2

    def validate_solution(self, ic, solution, tol=1e-6):
        is_valid = (
            not np.isnan(solution).any()
            and not np.isinf(solution).any()
            and np.abs(solution).max() < 1e3
        )
        return {"valid": is_valid, "max_value": float(np.abs(solution).max())}
