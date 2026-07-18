"""
2D stochastic Allen-Cahn equation.

    du = [ eps Laplacian(u) + u - u^3 ] dt + sigma dW(t, x, y)

Phase separation with thermal fluctuations: noise nucleates and roughens
interfaces, and strong enough noise flips domains between the +-1 wells.
Numerics: exponential integrator for the linear part (diffusion + linear
reaction), explicit cubic, Euler-Maruyama noise — same conventions as the
other stochastic models, with (n_realizations, ny, nx) outputs per IC.
"""

from typing import Callable, Dict, Tuple, Union

import numpy as np

from pdeforge.core.base import PDEModel
from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model


@register_model("stochastic_allen_cahn_2d")
class StochasticAllenCahn2D(PDEModel):
    """
    Stochastic Allen-Cahn on the periodic box; sigma -> 0 recovers the
    deterministic phase-separation dynamics.
    """

    NDIM = 2
    INPUT_NAMES = ["u0"]
    OUTPUT_NAMES = ["u_T_realizations"]

    USER_PARAMS = [
        ParamSpec(
            name="epsilon",
            description="Interface parameter",
            default=0.01,
            param_type=ParamType.PHYSICAL,
            bounds=(0.001, 0.5),
        ),
        ParamSpec(
            name="noise_intensity",
            description="Noise amplitude sigma",
            default=0.05,
            param_type=ParamType.PHYSICAL,
            bounds=(0.0, 2.0),
        ),
        ParamSpec(
            name="n_realizations",
            description="Noise realizations per initial condition",
            default=10,
            param_type=ParamType.OUTPUT,
            bounds=(1, 1000),
        ),
        ParamSpec(
            name="time_end",
            description="Final time",
            default=2.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.05, 100.0),
        ),
    ]

    DEFAULT_PARAMS = {
        "epsilon": 0.01,
        "noise_intensity": 0.05,
        "n_realizations": 10,
        "time_end": 2.0,
        "noise_cutoff": 8,
        "_n_time_steps": 101,
        "_dt": 0.01,
    }

    def __init__(self, resolution, domain=None, **params):
        super().__init__(resolution, domain, **params)

        p = self.params
        self.eps = p["epsilon"]
        self.sigma = p["noise_intensity"]
        self.n_realizations = int(p["n_realizations"])
        self.T = p.get("time_end", 2.0)
        self.n_t = p.get("_n_time_steps", 101)
        self.dt = p.get("_dt") or 0.01

        self.nx, self.ny = resolution["x"], resolution["y"]
        dx = self.grids["x"][1] - self.grids["x"][0]
        dy = self.grids["y"][1] - self.grids["y"][0]
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ny, d=dy)
        KX, KY = np.meshgrid(kx, ky)
        self.K2 = KX**2 + KY**2

        # exact propagator of the linear part  eps*Lap + 1
        self._E = np.exp((-self.eps * self.K2 + 1.0) * self.dt)
        kyi = np.abs(np.fft.fftfreq(self.ny) * self.ny)[:, None]
        kxi = np.abs(np.fft.fftfreq(self.nx) * self.nx)[None, :]
        self._noise_mask = (
            (np.sqrt(kxi**2 + kyi**2) <= p.get("noise_cutoff", 8))
        ).astype(float)

    def _det_step(self, u):
        u_hat = np.fft.fft2(u)
        cubic_hat = np.fft.fft2(-(u**3))
        return np.fft.ifft2(self._E * (u_hat + self.dt * cubic_hat)).real

    def _noise_increment(self, rng):
        xi_hat = np.fft.fft2(rng.standard_normal((self.ny, self.nx)))
        return np.fft.ifft2(xi_hat * self._noise_mask).real

    def solve_single_realization(self, ic, seed=None, return_full=False):
        rng = np.random.default_rng(seed)
        u = np.asarray(ic, dtype=float).copy()
        n_sub = max(1, int(np.ceil(self.T / self.dt)))
        sq = self.sigma * np.sqrt(self.dt)

        frames = [u.copy()] if return_full else None
        for _ in range(n_sub):
            u = self._det_step(u)
            if self.sigma > 0.0:
                u = u + sq * self._noise_increment(rng)
            if return_full:
                frames.append(u.copy())
        if return_full:
            return np.stack(frames, axis=0)
        return u

    def solve(self, ic, seed=None, return_full=False):
        realizations = []
        for r in range(self.n_realizations):
            r_seed = None if seed is None else seed + r
            realizations.append(self.solve_single_realization(ic, r_seed, return_full))
        return np.stack(realizations, axis=0)

    def generate_ic(self, generator="default", generator_params=None, seed=None):
        """Small random field around 0 (undecided phase)."""
        if generator_params is None:
            generator_params = {}
        rng = np.random.default_rng(seed)
        amp = generator_params.get("amplitude", 0.1)
        return amp * rng.standard_normal((self.ny, self.nx))

    def generate_sample(
        self,
        generator="default",
        generator_params=None,
        seed=None,
        validate=True,
        max_attempts=10,
    ):
        ic = self.generate_ic(generator, generator_params, seed)
        solution = self.solve(ic, seed=seed)
        info = self.validate_solution(ic, solution) if validate else {"valid": True}
        if validate and not info["valid"]:
            raise RuntimeError("stochastic allen-cahn produced invalid sample")
        return ic, solution, info

    def validate_solution(self, ic, solution, tol=1e-6):
        is_valid = (
            not np.isnan(solution).any()
            and not np.isinf(solution).any()
            and np.abs(solution).max() < 100.0
        )
        return {"valid": is_valid, "max_value": float(np.abs(solution).max())}
