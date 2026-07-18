"""
1D Burgers equation solver.

Advection-diffusion with shock formation:
    du/dt + u du/dx = nu d2u/dx2

Operator learning task: u(x, t=0) -> u(x, t=T)
"""

import numpy as np

from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model
from pdeforge.generators.initial_conditions import FourierICGenerator, get_ic_generator
from pdeforge.solvers.semilinear import SemiLinearSpectralModel


@register_model("burgers_1d")
class Burgers1D(SemiLinearSpectralModel):
    """
    1D Burgers equation. Maps initial condition to solution at final time.
    Lower viscosity = sharper shocks.

    Solved with ETDRK4 on the spectral seam: the stiff diffusion term is
    integrated exactly, only the advection nonlinearity is stepped explicitly
    (in conservative form, 2/3-dealiased).
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
        "_dt": None,
    }

    def __init__(self, resolution, domain=None, **params):
        super().__init__(resolution, domain, **params)

        self.nu = self.params["viscosity"]
        self.mu = self.params.get("advection", 1.0)
        self.T = self.params.get("time_horizon", self.params.get("time_end", 1.0))
        self.n_t = self.params.get("_n_time_steps", 401)

        self._setup_spectral()
        self.nx = resolution["x"]
        self.dx = self.grids["x"][1] - self.grids["x"][0]
        self.k = self.K[0]

        # Advective CFL for the explicit nonlinear term; diffusion is exact.
        dt = self.params.get("_dt")
        if dt is None:
            dt = min(self.T / 400.0, 0.5 * self.dx)
        self.dt = dt

    def linear_symbol(self):
        return -self.nu * self.k**2

    def nonlinear_hat(self, v, u, ops):
        # Conservative form: -mu * d/dx (u^2 / 2), dealiased.
        u2_hat = self._fft(u * u, ops)
        return -0.5j * self.mu * self.k * (self.dealias * u2_hat)

    def generate_ic(self, generator="fourier", generator_params=None, seed=None):
        """Generate random IC using Fourier sine series."""
        if generator_params is None:
            generator_params = {}

        # defaults for Burgers (Fourier generator only — other generators
        # have their own parameter sets)
        if generator == "fourier":
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
