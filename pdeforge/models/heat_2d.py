"""
2D Heat Equation Solver

The heat equation models diffusion/conduction:

    ∂u/∂t = α ∇²u = α (∂²u/∂x² + ∂²u/∂y²)

with periodic boundary conditions.

Operator Learning Task:
    u(x, y, t=0) → u(x, y, t=T)
"""

from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np

from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model
from pdeforge.core.types import PDEDataset
from pdeforge.generators.initial_conditions import get_ic_generator
from pdeforge.solvers.semilinear import SemiLinearSpectralModel


@register_model("heat_2d")
class Heat2D(SemiLinearSpectralModel):
    """
    2D Heat equation for diffusion processes.

    ∂u/∂t = α ∇²u

    Solutions smooth out isotropically over time.

    Examples
    --------
    >>> dataset = generate_dataset(
    ...     model="heat_2d",
    ...     n_samples=100,
    ...     resolution={"x": 64, "y": 64},
    ...     params={"diffusivity": 0.01, "time_end": 1.0},
    ... )
    """

    NDIM = 2
    INPUT_NAMES = ["u0"]
    OUTPUT_NAMES = ["u_T"]

    USER_PARAMS = [
        ParamSpec(
            name="diffusivity",
            description="Thermal diffusivity α",
            default=0.01,
            param_type=ParamType.PHYSICAL,
            bounds=(1e-6, 1.0),
            units="m²/s",
            affects="Higher diffusivity → faster smoothing",
        ),
        ParamSpec(
            name="time_end",
            description="Final time for solution",
            default=1.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.01, 10.0),
            units="s",
            affects="Longer time → smoother solution",
        ),
    ]

    DEFAULT_PARAMS = {
        "diffusivity": 0.01,
        "time_end": 1.0,
        "_n_time_steps": 101,
        "_dt": None,
    }

    def __init__(
        self,
        resolution: Dict[str, int],
        domain: Dict[str, Tuple[float, float]] = None,
        **params,
    ):
        super().__init__(resolution, domain, **params)

        self.alpha = self.params["diffusivity"]
        self.T = self.params.get("time_end", 1.0)
        self.n_t = self.params.get("_n_time_steps", 101)

        self._setup_spectral()
        self.nx = resolution["x"]
        self.ny = resolution["y"]

        # Purely linear: the seam applies the exact propagator per substep
        # (û(k, t+dt) = û(k, t)·exp(-α|k|²dt)), so dt only sets frame cadence.
        self.dt = self.params.get("_dt") or self.T / max(1, self.n_t - 1)

    def linear_symbol(self):
        return -self.alpha * self.K2

    def generate_ic(
        self,
        generator: Union[str, Callable] = "fourier",
        generator_params: Dict = None,
        seed: int = None,
    ) -> np.ndarray:
        """Generate random 2D initial conditions."""
        if generator_params is None:
            generator_params = {}

        if generator == "fourier":
            default_params = {
                "n_modes": 8,
                "decay": 2.0,
                "amplitude": 1.0,
            }
            generator_params = {**default_params, **generator_params}

        if isinstance(generator, str):
            gen = get_ic_generator(generator, **generator_params)
        else:
            gen = generator

        return gen.generate(shape=(self.ny, self.nx), seed=seed, grid=self.grids)

    def validate_solution(
        self,
        ic: np.ndarray,
        solution: np.ndarray,
        tol: float = 1e-6,
    ) -> Dict:
        """Validate the solution."""
        is_valid = not np.isnan(solution).any() and not np.isinf(solution).any()
        return {"valid": is_valid, "max_value": np.abs(solution).max()}
