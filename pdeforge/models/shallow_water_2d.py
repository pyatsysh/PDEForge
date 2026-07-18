"""
2D shallow water equations (conservative flux form).

    h_t  + (hu)_x + (hv)_y = 0
    (hu)_t + (hu^2 + g h^2/2)_x + (huv)_y   = -k4 Laplacian^2 (hu)
    (hv)_t + (huv)_x   + (hv^2 + g h^2/2)_y = -k4 Laplacian^2 (hv)

Pseudo-spectral with a small hyperviscous filter (the diagonal linear part,
integrated exactly) and fully explicit dealiased flux divergences. Mass
(the k = 0 mode of h) is conserved to machine precision because the flux
divergence has zero mean spectrally.

Operator learning task: (h, hu, hv)(x, y, 0) -> (h, hu, hv)(x, y, T),
components stacked: shape (3, ny, nx).
"""

from typing import Callable, Dict, Tuple, Union

import numpy as np

from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model
from pdeforge.solvers.semilinear import SemiLinearSpectralModel


@register_model("shallow_water_2d")
class ShallowWater2D(SemiLinearSpectralModel):
    """
    Shallow water on the periodic box: gravity waves over mean depth H,
    conservative flux form, spectral hyperviscosity for stability.
    """

    NDIM = 2
    INPUT_NAMES = ["h0", "hu0", "hv0"]
    OUTPUT_NAMES = ["h_T", "hu_T", "hv_T"]

    USER_PARAMS = [
        ParamSpec(
            name="gravity",
            description="Gravitational acceleration g",
            default=9.81,
            param_type=ParamType.PHYSICAL,
            bounds=(0.1, 100.0),
        ),
        ParamSpec(
            name="mean_depth",
            description="Mean water depth H",
            default=1.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.01, 100.0),
        ),
        ParamSpec(
            name="time_end",
            description="Final time",
            default=0.2,
            param_type=ParamType.PHYSICAL,
            bounds=(0.001, 10.0),
        ),
        ParamSpec(
            name="hyperviscosity",
            description="Spectral filter strength (stability)",
            default=1e-7,
            param_type=ParamType.PHYSICAL,
            bounds=(0.0, 1e-3),
        ),
    ]

    DEFAULT_PARAMS = {
        "gravity": 9.81,
        "mean_depth": 1.0,
        "time_end": 0.2,
        "hyperviscosity": 1e-7,
        "_n_time_steps": 101,
        "_dt": None,
    }

    def __init__(self, resolution, domain=None, **params):
        super().__init__(resolution, domain, **params)

        self.g = self.params["gravity"]
        self.H = self.params["mean_depth"]
        self.k4 = self.params.get("hyperviscosity", 1e-7)
        self.T = self.params.get("time_end", 0.2)
        self.n_t = self.params.get("_n_time_steps", 101)

        self._setup_spectral()
        self.KY, self.KX = self.K[0], self.K[1]

        # Gravity-wave CFL: c = sqrt(g H)
        dx = self.grids["x"][1] - self.grids["x"][0]
        c = np.sqrt(self.g * self.H)
        self.dt = self.params.get("_dt") or 0.25 * dx / max(c, 1e-8)

    def linear_symbol(self):
        L = -self.k4 * self.K2**2
        return np.stack([L, L, L], axis=0)

    def _ddx(self, f, ops):
        return ops.real(
            ops.ifftn(1j * self.KX * self._fft(f, ops), axes=self.spatial_axes)
        )

    def _ddy(self, f, ops):
        return ops.real(
            ops.ifftn(1j * self.KY * self._fft(f, ops), axes=self.spatial_axes)
        )

    def nonlinear_hat(self, v, u, ops):
        h, hu, hv = u[0], u[1], u[2]
        # velocities (h stays positive for the intended input measure)
        uu = hu / h
        vv = hv / h
        Fh = -(self._ddx(hu, ops) + self._ddy(hv, ops))
        Fu = -(self._ddx(hu * uu + 0.5 * self.g * h * h, ops) + self._ddy(hu * vv, ops))
        Fv = -(self._ddx(hv * uu, ops) + self._ddy(hv * vv + 0.5 * self.g * h * h, ops))
        return ops.stack(
            [
                self.dealias * self._fft(Fh, ops),
                self.dealias * self._fft(Fu, ops),
                self.dealias * self._fft(Fv, ops),
            ],
            axis=0,
        )

    def generate_ic(self, generator="default", generator_params=None, seed=None):
        """Still water (hu = hv = 0) with smooth random surface elevation."""
        if generator_params is None:
            generator_params = {}
        rng = np.random.default_rng(seed)
        ny, nx = self.field_shape

        amp = generator_params.get("amplitude", 0.05) * self.H
        cutoff = generator_params.get("cutoff", 4)

        noise_hat = np.fft.fft2(rng.standard_normal((ny, nx)))
        ky = np.fft.fftfreq(ny) * ny
        kx = np.fft.fftfreq(nx) * nx
        KX, KY = np.meshgrid(kx, ky)
        mask = np.sqrt(KX**2 + KY**2) <= cutoff
        eta = np.fft.ifft2(noise_hat * mask).real
        eta = amp * eta / (np.abs(eta).max() + 1e-12)

        h0 = self.H + eta
        z = np.zeros_like(h0)
        return np.stack([h0, z, z], axis=0)

    def validate_solution(self, ic, solution, tol=1e-6):
        h = solution[0] if solution.ndim == 3 else solution[:, 0]
        is_valid = (
            not np.isnan(solution).any()
            and not np.isinf(solution).any()
            and np.min(h) > 0.0  # water depth must stay positive
        )
        return {"valid": is_valid, "min_depth": float(np.min(h))}
