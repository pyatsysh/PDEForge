"""
2D Gray-Scott reaction-diffusion system (pattern formation).

    U_t = Du Laplacian(U) - U V^2 + F (1 - U)
    V_t = Dv Laplacian(V) + U V^2 - (F + k) V

The classic Pearson (1993) parameter plane: depending on feed F and kill k,
perturbations of the trivial state (U, V) = (1, 0) grow into spots, stripes,
labyrinths, or self-replicating patterns.

Operator learning task: (U, V)(x, y, 0) -> (U, V)(x, y, T). Fields are
stacked on a leading component axis: shape (2, ny, nx).
"""

from typing import Callable, Dict, Tuple, Union

import numpy as np

from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model
from pdeforge.solvers.semilinear import SemiLinearSpectralModel


@register_model("gray_scott_2d")
class GrayScott2D(SemiLinearSpectralModel):
    """
    Gray-Scott on the periodic box, Pearson's scaling (domain [0, 2.5]^2,
    Du = 2e-5, Dv = 1e-5). Two components ride the seam's leading component
    axis with a diagonal linear symbol; the U V^2 kinetics is stepped
    explicitly.
    """

    NDIM = 2
    INPUT_NAMES = ["U0", "V0"]
    OUTPUT_NAMES = ["U_T", "V_T"]

    USER_PARAMS = [
        ParamSpec(
            name="feed",
            description="Feed rate F",
            default=0.04,
            param_type=ParamType.PHYSICAL,
            bounds=(0.0, 0.12),
            affects="With kill, selects the Pearson pattern regime",
        ),
        ParamSpec(
            name="kill",
            description="Kill rate k",
            default=0.06,
            param_type=ParamType.PHYSICAL,
            bounds=(0.0, 0.08),
        ),
        ParamSpec(
            name="Du",
            description="Diffusivity of U",
            default=2e-5,
            param_type=ParamType.PHYSICAL,
            bounds=(1e-6, 1e-3),
        ),
        ParamSpec(
            name="Dv",
            description="Diffusivity of V",
            default=1e-5,
            param_type=ParamType.PHYSICAL,
            bounds=(1e-6, 1e-3),
        ),
        ParamSpec(
            name="time_end",
            description="Final time (patterns need t ~ 1000+)",
            default=2000.0,
            param_type=ParamType.PHYSICAL,
            bounds=(10.0, 20000.0),
        ),
    ]

    DEFAULT_PARAMS = {
        "feed": 0.04,
        "kill": 0.06,
        "Du": 2e-5,
        "Dv": 1e-5,
        "time_end": 2000.0,
        "_n_time_steps": 101,
        "_dt": 1.0,
    }

    def __init__(self, resolution, domain=None, **params):
        if domain is None:
            domain = {"x": (0.0, 2.5), "y": (0.0, 2.5)}
        super().__init__(resolution, domain, **params)

        self.F = self.params["feed"]
        self.k_rate = self.params["kill"]
        self.Du = self.params["Du"]
        self.Dv = self.params["Dv"]
        self.T = self.params.get("time_end", 2000.0)
        self.n_t = self.params.get("_n_time_steps", 101)
        self.dt = self.params.get("_dt") or 1.0

        self._setup_spectral()

    def linear_symbol(self):
        # Diagonal per component: linear decay terms ride in L.
        L_u = -self.Du * self.K2 - self.F
        L_v = -self.Dv * self.K2 - (self.F + self.k_rate)
        return np.stack([L_u, L_v], axis=0)

    def nonlinear_hat(self, v, u, ops):
        U, V = u[0], u[1]
        uvv = U * V * V
        # constant source F belongs to N (its transform is a k=0 spike)
        N_u = self._fft(-uvv + self.F, ops)
        N_v = self._fft(uvv, ops)
        return ops.stack([self.dealias * N_u, self.dealias * N_v], axis=0)

    def generate_ic(self, generator="default", generator_params=None, seed=None):
        """
        Pearson-style IC: the trivial state (U, V) = (1, 0) with a randomly
        placed square patch perturbed to (1/2, 1/4) plus small noise.
        """
        if generator_params is None:
            generator_params = {}
        rng = np.random.default_rng(seed)

        ny, nx = self.field_shape
        U = np.ones((ny, nx))
        V = np.zeros((ny, nx))

        n_patches = generator_params.get("n_patches", 3)
        patch = generator_params.get("patch_size", max(2, nx // 16))
        for _ in range(n_patches):
            cx = rng.integers(0, nx - patch)
            cy = rng.integers(0, ny - patch)
            U[cy : cy + patch, cx : cx + patch] = 0.5
            V[cy : cy + patch, cx : cx + patch] = 0.25

        noise = generator_params.get("noise", 0.02)
        U += noise * rng.standard_normal((ny, nx))
        V += noise * rng.standard_normal((ny, nx))

        return np.stack([U, V], axis=0)

    def validate_solution(self, ic, solution, tol=1e-6):
        is_valid = (
            not np.isnan(solution).any()
            and not np.isinf(solution).any()
            and np.abs(solution).max() < 10.0  # concentrations stay O(1)
        )
        return {"valid": is_valid, "max_value": float(np.abs(solution).max())}
