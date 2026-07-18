"""
2D Burgers system (vector velocity self-advection).

    u_t + u u_x + v u_y = nu Laplacian(u)
    v_t + u v_x + v v_y = nu Laplacian(v)

The standard 2D generalisation of Burgers: sharp fronts form and diffuse in
two dimensions without pressure coupling.

Operator learning task: (u, v)(x, y, 0) -> (u, v)(x, y, T), components
stacked on a leading axis: shape (2, ny, nx).
"""

from typing import Callable, Dict, Tuple, Union

import numpy as np

from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model
from pdeforge.generators.initial_conditions import get_ic_generator
from pdeforge.solvers.semilinear import SemiLinearSpectralModel


@register_model("burgers_2d")
class Burgers2D(SemiLinearSpectralModel):
    """
    2D Burgers on the periodic box, ETDRK4 on the seam: exact viscous
    diffusion, explicit dealiased self-advection for both components.
    """

    NDIM = 2
    INPUT_NAMES = ["u0", "v0"]
    OUTPUT_NAMES = ["u_T", "v_T"]

    USER_PARAMS = [
        ParamSpec(
            name="viscosity",
            description="Viscosity (lower = sharper fronts)",
            default=0.01,
            param_type=ParamType.PHYSICAL,
            bounds=(1e-5, 1.0),
        ),
        ParamSpec(
            name="time_horizon",
            description="Final time T",
            default=1.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.05, 10.0),
        ),
    ]

    DEFAULT_PARAMS = {
        "viscosity": 0.01,
        "time_horizon": 1.0,
        "_n_time_steps": 101,
        "_dt": None,
    }

    def __init__(self, resolution, domain=None, **params):
        super().__init__(resolution, domain, **params)

        self.nu = self.params["viscosity"]
        self.T = self.params.get("time_horizon", 1.0)
        self.n_t = self.params.get("_n_time_steps", 101)

        self._setup_spectral()
        self.KY, self.KX = self.K[0], self.K[1]

        dx = self.grids["x"][1] - self.grids["x"][0]
        self.dt = self.params.get("_dt") or min(0.25 * dx, self.T / 200.0)

    def linear_symbol(self):
        L = -self.nu * self.K2
        return np.stack([L, L], axis=0)

    def nonlinear_hat(self, v, u, ops):
        U, V = u[0], u[1]
        u_hat, v_hat = v[0], v[1]
        Ux = ops.real(ops.ifftn(1j * self.KX * u_hat, axes=self.spatial_axes))
        Uy = ops.real(ops.ifftn(1j * self.KY * u_hat, axes=self.spatial_axes))
        Vx = ops.real(ops.ifftn(1j * self.KX * v_hat, axes=self.spatial_axes))
        Vy = ops.real(ops.ifftn(1j * self.KY * v_hat, axes=self.spatial_axes))
        N_u = self._fft(-(U * Ux + V * Uy), ops)
        N_v = self._fft(-(U * Vx + V * Vy), ops)
        return ops.stack([self.dealias * N_u, self.dealias * N_v], axis=0)

    def generate_ic(self, generator="fourier", generator_params=None, seed=None):
        """Two smooth random components (independent Fourier fields)."""
        if generator_params is None:
            generator_params = {}
        if generator == "fourier":
            generator_params = {
                "n_modes": 6,
                "decay": 2.0,
                "amplitude": 0.5,
                **generator_params,
            }
        if isinstance(generator, str):
            gen = get_ic_generator(generator, **generator_params)
        else:
            gen = generator
        # independent seeds per component, derived deterministically
        s2 = None if seed is None else seed + 987654321
        U = gen.generate(shape=self.field_shape, seed=seed, grid=self.grids)
        V = gen.generate(shape=self.field_shape, seed=s2, grid=self.grids)
        return np.stack([U, V], axis=0)

    def validate_solution(self, ic, solution, tol=1e-6):
        is_valid = (
            not np.isnan(solution).any()
            and not np.isinf(solution).any()
            and np.abs(solution).max() < 1e6
        )
        return {"valid": is_valid, "max_value": float(np.abs(solution).max())}
