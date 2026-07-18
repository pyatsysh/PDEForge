"""
2D Helmholtz equation (frequency domain) on the periodic box.

    (Laplacian + kappa^2 + i gamma kappa) u = f

The small absorption term i*gamma*kappa regularises resonances (as in
physically lossy media), making the periodic problem well-posed for every
wavenumber. Solved DIRECTLY in Fourier space — one division, machine-precision
exact for the discrete operator.

Operator learning task: source f(x, y) -> real wavefield Re u(x, y). This is
the standard frequency-domain scattering/elliptic benchmark family.
"""

from typing import Callable, Dict, Tuple, Union

import numpy as np

from pdeforge.core.base import PDEModel
from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model
from pdeforge.generators.initial_conditions import get_ic_generator


@register_model("helmholtz_2d")
class Helmholtz2D(PDEModel):
    """
    Damped Helmholtz on the periodic box, exact spectral solve.
    Steady (frequency-domain) problem: no time axis.
    """

    NDIM = 2
    TIME_DEPENDENT = False  # frequency-domain elliptic solve
    INPUT_NAMES = ["f"]
    OUTPUT_NAMES = ["u"]

    USER_PARAMS = [
        ParamSpec(
            name="wavenumber",
            description="Helmholtz wavenumber kappa",
            default=20.0,
            param_type=ParamType.PHYSICAL,
            bounds=(1.0, 200.0),
        ),
        ParamSpec(
            name="damping",
            description="Absorption gamma (regularises resonances)",
            default=1.0,
            param_type=ParamType.PHYSICAL,
            bounds=(1e-3, 20.0),
        ),
    ]

    DEFAULT_PARAMS = {
        "wavenumber": 20.0,
        "damping": 1.0,
    }

    def __init__(self, resolution, domain=None, **params):
        super().__init__(resolution, domain, **params)

        self.kappa = self.params["wavenumber"]
        self.gamma = self.params["damping"]

        self.nx, self.ny = resolution["x"], resolution["y"]
        dx = self.grids["x"][1] - self.grids["x"][0]
        dy = self.grids["y"][1] - self.grids["y"][0]
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ny, d=dy)
        KX, KY = np.meshgrid(kx, ky)
        self.K2 = KX**2 + KY**2

        # Symbol of (Laplacian + kappa^2 + i gamma kappa)
        self._symbol = -self.K2 + self.kappa**2 + 1j * self.gamma * self.kappa

    def solve(self, ic, return_info=False):
        """ic is the source f; returns the real part of the wavefield."""
        f_hat = np.fft.fft2(np.asarray(ic, dtype=float))
        u = np.fft.ifft2(f_hat / self._symbol)
        return u.real

    def solve_complex(self, f):
        """Full complex wavefield (helper for validation/inspection)."""
        f_hat = np.fft.fft2(np.asarray(f, dtype=float))
        return np.fft.ifft2(f_hat / self._symbol)

    def generate_ic(self, generator="grf", generator_params=None, seed=None):
        """Random smooth source field."""
        if generator_params is None:
            generator_params = {}
        if generator == "grf":
            generator_params = {"alpha": 2.0, "amplitude": 1.0, **generator_params}
        if isinstance(generator, str):
            gen = get_ic_generator(generator, **generator_params)
        else:
            gen = generator
        return gen.generate(shape=(self.ny, self.nx), seed=seed, grid=self.grids)

    def validate_solution(self, ic, solution, tol=1e-6):
        is_valid = not np.isnan(solution).any() and not np.isinf(solution).any()
        return {"valid": is_valid, "max_value": float(np.abs(solution).max())}
