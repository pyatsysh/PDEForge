"""
1D Burgers equation solver.

Advection-diffusion with shock formation:
    du/dt + u du/dx = nu d2u/dx2

Operator learning task: u(x, t=0) -> u(x, t=T)
"""

import numpy as np
from scipy.integrate import odeint

from pdeforge.core.base import PDEModel
from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model
from pdeforge.generators.initial_conditions import FourierICGenerator, get_ic_generator


@register_model("burgers_1d")
class Burgers1D(PDEModel):
    """
    1D Burgers equation. Maps initial condition to solution at final time.
    Lower viscosity = sharper shocks.
    """

    NDIM = 1
    INPUT_NAMES = ["u0"]
    OUTPUT_NAMES = ["u_T"]

    USER_PARAMS = [
        ParamSpec(
            name="viscosity",
            description="Diffusion coeff (lower = sharper shocks)",
            default=0.01 / np.pi,
            param_type=ParamType.PHYSICAL,
            bounds=(1e-6, 1.0),
        ),
        ParamSpec(
            name="time_horizon",
            description="Final time T",
            default=1.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.1, 10.0),
        ),
    ]

    DEFAULT_PARAMS = {
        "viscosity": 0.01 / np.pi,
        "advection": 1.0,
        "time_horizon": 1.0,
        "_n_time_steps": 401,
    }

    def __init__(self, resolution, domain=None, **params):
        super().__init__(resolution, domain, **params)

        self.nu = self.params["viscosity"]
        self.mu = self.params.get("advection", 1.0)
        self.T = self.params.get("time_horizon", self.params.get("time_end", 1.0))
        self.n_t = self.params.get("_n_time_steps", 401)

        # precompute wavenumbers
        self.nx = resolution["x"]
        self.dx = self.grids["x"][1] - self.grids["x"][0]
        self.k = 2 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)

    def _rhs(self, u, t):
        # cf. Anima et al
        u_hat = np.fft.fft(u)
        u_hat_x = 1j * self.k * u_hat
        u_hat_xx = -self.k**2 * u_hat

        u_x = np.fft.ifft(u_hat_x).real
        u_xx = np.fft.ifft(u_hat_xx).real

        return -self.mu * u * u_x + self.nu * u_xx

    def solve(self, ic, return_full=False):
        """
        Solve Burgers equation.

        ic: initial condition u(x, t=0)
        return_full: if True, return solution at all timesteps
        """
        t = np.linspace(0, self.T, self.n_t)
        U = odeint(self._rhs, ic, t, mxstep=5000)

        if return_full:
            return U  # (n_t, nx)
        else:
            return U[-1]  # (nx,)

    def generate_ic(self, generator="fourier", generator_params=None, seed=None):
        """Generate random IC using Fourier sine series."""
        if generator_params is None:
            generator_params = {}

        # defaults for Burgers
        default_params = {
            "n_modes": 10,
            "decay": 1.5,
            "amplitude": 0.7,
            "use_cos": False,
        }
        generator_params = {**default_params, **generator_params}

        if isinstance(generator, str):
            gen = get_ic_generator(generator, **generator_params)
        else:
            gen = generator

        return gen.generate(
            shape=(self.nx,),
            seed=seed,
            grid=self.grids,
        )

    def validate_solution(self, ic, solution, tol=1e-6):
        """Check for NaN/Inf and blowup."""
        is_valid = (
            not np.isnan(solution).any()
            and not np.isinf(solution).any()
            and np.abs(solution).max() < 1e10
        )
        return {
            "valid": is_valid,
            "max_value": np.abs(solution).max(),
        }
