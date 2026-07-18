"""
1D stochastic Burgers equation.

    du = [ -u u_x + nu u_xx ] dt + sigma dW(t, x)

Additive space-time noise (white in time, optionally smoothed in space) on
top of Burgers dynamics. Numerics: exponential integrator for the exact
viscous part, explicit dealiased advection, Euler-Maruyama noise increments —
mirroring the stochastic_heat models' conventions.

Like the other stochastic models, each sample carries MULTIPLE realizations
per initial condition: outputs have shape (n_samples, n_realizations, nx),
the natural target for distributional operator learning P(u_T | u_0).
"""

from typing import Callable, Dict, Tuple, Union

import numpy as np

from pdeforge.core.base import PDEModel
from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model
from pdeforge.generators.initial_conditions import get_ic_generator


@register_model("stochastic_burgers_1d")
class StochasticBurgers1D(PDEModel):
    """
    Stochastic Burgers with additive noise; sigma -> 0 recovers the
    deterministic dynamics (validated in tests).
    """

    NDIM = 1
    INPUT_NAMES = ["u0"]
    OUTPUT_NAMES = ["u_T_realizations"]

    USER_PARAMS = [
        ParamSpec(
            name="viscosity",
            description="Viscosity nu",
            default=0.05,
            param_type=ParamType.PHYSICAL,
            bounds=(1e-4, 1.0),
        ),
        ParamSpec(
            name="noise_intensity",
            description="Noise amplitude sigma",
            default=0.1,
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
            name="time_horizon",
            description="Final time T",
            default=0.5,
            param_type=ParamType.PHYSICAL,
            bounds=(0.01, 10.0),
        ),
    ]

    DEFAULT_PARAMS = {
        "viscosity": 0.05,
        "noise_intensity": 0.1,
        "n_realizations": 10,
        "time_horizon": 0.5,
        "noise_cutoff": 16,  # spatial smoothing: keep |mode| <= cutoff
        "_n_time_steps": 201,
        "_dt": None,
    }

    def __init__(self, resolution, domain=None, **params):
        super().__init__(resolution, domain, **params)

        p = self.params
        self.nu = p["viscosity"]
        self.sigma = p["noise_intensity"]
        self.n_realizations = int(p["n_realizations"])
        self.T = p.get("time_horizon", 0.5)
        self.n_t = p.get("_n_time_steps", 201)

        self.nx = resolution["x"]
        self.dx = self.grids["x"][1] - self.grids["x"][0]
        self.k = 2 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)

        self.dt = p.get("_dt") or min(self.T / 200.0, 0.5 * self.dx)
        # exact viscous propagator per substep
        self._E = np.exp(-self.nu * self.k**2 * self.dt)
        # 2/3 dealias mask + spatial noise smoothing mask
        modes = np.abs(np.fft.fftfreq(self.nx) * self.nx)
        self._dealias = (modes <= self.nx // 3).astype(float)
        self._noise_mask = (modes <= p.get("noise_cutoff", 16)).astype(float)

    def _det_step(self, u):
        """One deterministic substep: exact diffusion + explicit advection."""
        u_hat = np.fft.fft(u)
        adv_hat = -0.5j * self.k * self._dealias * np.fft.fft(u * u)
        return np.fft.ifft(self._E * (u_hat + self.dt * adv_hat)).real

    def _noise_increment(self, rng):
        """Spatially smoothed white-noise increment (unit variance density)."""
        xi = rng.standard_normal(self.nx)
        xi_hat = np.fft.fft(xi) * self._noise_mask
        return np.fft.ifft(xi_hat).real

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
        """All realizations: shape (n_realizations, nx) [or (R, n_sub+1, nx)]."""
        realizations = []
        for r in range(self.n_realizations):
            r_seed = None if seed is None else seed + r
            realizations.append(self.solve_single_realization(ic, r_seed, return_full))
        return np.stack(realizations, axis=0)

    def generate_ic(self, generator="fourier", generator_params=None, seed=None):
        if generator_params is None:
            generator_params = {}
        if generator == "fourier":
            generator_params = {
                "n_modes": 10,
                "decay": 1.5,
                "amplitude": 0.7,
                "use_cos": False,
                **generator_params,
            }
        if isinstance(generator, str):
            gen = get_ic_generator(generator, **generator_params)
        else:
            gen = generator
        return gen.generate(shape=(self.nx,), seed=seed, grid=self.grids)

    def generate_sample(
        self,
        generator="fourier",
        generator_params=None,
        seed=None,
        validate=True,
        max_attempts=10,
    ):
        """IC + realization bundle (mirrors the stochastic_heat convention)."""
        ic = self.generate_ic(generator, generator_params, seed)
        solution = self.solve(ic, seed=seed)
        info = self.validate_solution(ic, solution) if validate else {"valid": True}
        if validate and not info["valid"]:
            raise RuntimeError("stochastic burgers produced invalid sample")
        return ic, solution, info

    def validate_solution(self, ic, solution, tol=1e-6):
        is_valid = (
            not np.isnan(solution).any()
            and not np.isinf(solution).any()
            and np.abs(solution).max() < 1e6
        )
        return {"valid": is_valid, "max_value": float(np.abs(solution).max())}
