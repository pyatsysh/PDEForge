"""
1D linear advection equation.

    u_t + c u_x = 0

The exactly-solvable sanity anchor: the solution is the initial condition
translated by c*T. Any operator-learning pipeline (and any solver change in
PDEForge itself) should reproduce this to machine precision.

Operator learning task: u(x, 0) -> u(x, T) = u(x - c T, 0).
"""

from typing import Callable, Dict, Tuple, Union

import numpy as np

from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model
from pdeforge.generators.initial_conditions import get_ic_generator
from pdeforge.solvers.semilinear import SemiLinearSpectralModel


@register_model("advection_1d")
class Advection1D(SemiLinearSpectralModel):
    """
    Linear advection on the periodic domain. Purely linear on the seam: the
    spectral propagator exp(-i c k t) is applied exactly, so the discrete
    solution IS the band-limited translation of the IC.
    """

    NDIM = 1
    INPUT_NAMES = ["u0"]
    OUTPUT_NAMES = ["u_T"]

    USER_PARAMS = [
        ParamSpec(
            name="speed",
            description="Advection speed c",
            default=1.0,
            param_type=ParamType.PHYSICAL,
            bounds=(-100.0, 100.0),
        ),
        ParamSpec(
            name="time_end",
            description="Final time",
            default=0.5,
            param_type=ParamType.PHYSICAL,
            bounds=(0.01, 100.0),
        ),
    ]

    DEFAULT_PARAMS = {
        "speed": 1.0,
        "time_end": 0.5,
        "_n_time_steps": 101,
        "_dt": None,
    }

    def __init__(self, resolution, domain=None, **params):
        super().__init__(resolution, domain, **params)

        self.c = self.params["speed"]
        self.T = self.params.get("time_end", 0.5)
        self.n_t = self.params.get("_n_time_steps", 101)

        self._setup_spectral()
        self.nx = resolution["x"]
        self.k = self.K[0]
        self.dt = self.params.get("_dt") or self.T / max(1, self.n_t - 1)

    def linear_symbol(self):
        # u_t = -c u_x -> L = -i c k (purely dispersive; |exp(L t)| = 1)
        return -1j * self.c * self.k

    def generate_ic(self, generator="fourier", generator_params=None, seed=None):
        if generator_params is None:
            generator_params = {}
        if generator == "fourier":
            generator_params = {
                "n_modes": 8,
                "decay": 1.5,
                "amplitude": 1.0,
                **generator_params,
            }
        if isinstance(generator, str):
            gen = get_ic_generator(generator, **generator_params)
        else:
            gen = generator
        return gen.generate(shape=(self.nx,), seed=seed, grid=self.grids)

    def exact_solution(self, ic, t=None):
        """Band-limited exact solution: spectral translation of the IC."""
        t = self.T if t is None else t
        return np.fft.ifft(np.fft.fft(ic) * np.exp(-1j * self.c * self.k * t)).real
