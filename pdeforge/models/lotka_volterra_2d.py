"""
2D diffusive Lotka-Volterra predator-prey system.

    u_t = Du Laplacian(u) + a u - b u v      (prey)
    v_t = Dv Laplacian(v) - c v + d u v      (predator)

Spatial diffusion turns the classic predator-prey oscillation into travelling
population waves and patchy dynamics.

Operator learning task: (u, v)(x, y, 0) -> (u, v)(x, y, T), components
stacked on a leading axis: shape (2, ny, nx).
"""

from typing import Callable, Dict, Tuple, Union

import numpy as np

from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model
from pdeforge.solvers.semilinear import SemiLinearSpectralModel


@register_model("lotka_volterra_2d")
class LotkaVolterra2D(SemiLinearSpectralModel):
    """
    Diffusive Lotka-Volterra on the periodic box. The diagonal linear parts
    (diffusion + linear growth/decay) ride in L; the u v coupling is stepped
    explicitly. With a spatially uniform IC the dynamics reduces exactly to
    the classic LV ODE — used as a validation invariant.
    """

    NDIM = 2
    INPUT_NAMES = ["u0", "v0"]
    OUTPUT_NAMES = ["u_T", "v_T"]

    USER_PARAMS = [
        ParamSpec(
            name="a",
            description="Prey growth rate",
            default=1.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.0, 10.0),
        ),
        ParamSpec(
            name="b",
            description="Predation rate",
            default=1.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.0, 10.0),
        ),
        ParamSpec(
            name="c",
            description="Predator death rate",
            default=1.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.0, 10.0),
        ),
        ParamSpec(
            name="d",
            description="Predator growth per prey",
            default=1.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.0, 10.0),
        ),
        ParamSpec(
            name="Du",
            description="Prey diffusivity",
            default=0.01,
            param_type=ParamType.PHYSICAL,
            bounds=(0.0, 1.0),
        ),
        ParamSpec(
            name="Dv",
            description="Predator diffusivity",
            default=0.005,
            param_type=ParamType.PHYSICAL,
            bounds=(0.0, 1.0),
        ),
        ParamSpec(
            name="time_end",
            description="Final time",
            default=5.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.1, 100.0),
        ),
    ]

    DEFAULT_PARAMS = {
        "a": 1.0,
        "b": 1.0,
        "c": 1.0,
        "d": 1.0,
        "Du": 0.01,
        "Dv": 0.005,
        "time_end": 5.0,
        "_n_time_steps": 101,
        "_dt": 0.01,
    }

    def __init__(self, resolution, domain=None, **params):
        super().__init__(resolution, domain, **params)

        p = self.params
        self.a, self.b, self.c, self.d = p["a"], p["b"], p["c"], p["d"]
        self.Du, self.Dv = p["Du"], p["Dv"]
        self.T = p.get("time_end", 5.0)
        self.n_t = p.get("_n_time_steps", 101)
        self.dt = p.get("_dt") or 0.01

        self._setup_spectral()

    def linear_symbol(self):
        L_u = -self.Du * self.K2 + self.a
        L_v = -self.Dv * self.K2 - self.c
        return np.stack([L_u, L_v], axis=0)

    def nonlinear_hat(self, v, u, ops):
        prey, pred = u[0], u[1]
        uv = prey * pred
        N_u = self._fft(-self.b * uv, ops)
        N_v = self._fft(self.d * uv, ops)
        return ops.stack([self.dealias * N_u, self.dealias * N_v], axis=0)

    def generate_ic(self, generator="default", generator_params=None, seed=None):
        """Random positive population fields around the coexistence point."""
        if generator_params is None:
            generator_params = {}
        rng = np.random.default_rng(seed)
        ny, nx = self.field_shape

        # Coexistence equilibrium (c/d, a/b) + smooth positive perturbations.
        u_star, v_star = self.c / self.d, self.a / self.b
        amp = generator_params.get("amplitude", 0.3)

        def smooth_field():
            noise = rng.standard_normal((ny, nx))
            noise_hat = np.fft.fft2(noise)
            cutoff = generator_params.get("cutoff", 4)
            ky = np.fft.fftfreq(ny) * ny
            kx = np.fft.fftfreq(nx) * nx
            KX, KY = np.meshgrid(kx, ky)
            mask = np.sqrt(KX**2 + KY**2) <= cutoff
            f = np.fft.ifft2(noise_hat * mask).real
            return f / (np.abs(f).max() + 1e-12)

        u0 = u_star * (1.0 + amp * smooth_field())
        v0 = v_star * (1.0 + amp * smooth_field())
        return np.stack([np.clip(u0, 1e-3, None), np.clip(v0, 1e-3, None)], axis=0)

    def validate_solution(self, ic, solution, tol=1e-6):
        is_valid = (
            not np.isnan(solution).any()
            and not np.isinf(solution).any()
            and np.abs(solution).max() < 1e4
        )
        return {"valid": is_valid, "max_value": float(np.abs(solution).max())}
