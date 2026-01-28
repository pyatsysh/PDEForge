"""
1D Heat Equation Solver

The heat equation models diffusion/conduction:

    ∂u/∂t = α ∂²u/∂x²

with periodic boundary conditions on [0, L].

Operator Learning Task:
    u(x, t=0) → u(x, t=T)
"""

import numpy as np
from scipy.integrate import odeint
from typing import Dict, Tuple, Optional, Union, Callable

from pdeforge.core.base import PDEModel
from pdeforge.core.registry import register_model
from pdeforge.core.types import PDEDataset
from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.generators.initial_conditions import get_ic_generator


@register_model("heat_1d")
class Heat1D(PDEModel):
    """
    1D Heat equation for diffusion processes.
    
    ∂u/∂t = α ∂²u/∂x²
    
    This is the simplest parabolic PDE - solutions smooth out over time.
    Use as a baseline for time-dependent operator learning.
    
    Examples
    --------
    >>> dataset = generate_dataset(
    ...     model="heat_1d",
    ...     n_samples=100,
    ...     resolution={"x": 256},
    ...     params={"diffusivity": 0.01, "time_end": 1.0},
    ... )
    """
    
    NDIM = 1
    INPUT_NAMES = ["u0"]
    OUTPUT_NAMES = ["u_T"]
    
    USER_PARAMS = [
        ParamSpec(
            name="diffusivity",
            description="Thermal diffusivity α (higher = faster smoothing)",
            default=0.01,
            param_type=ParamType.PHYSICAL,
            bounds=(1e-6, 1.0),
            units="m²/s",
            affects="Higher diffusivity → faster decay of high frequencies",
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
        "_n_time_steps": 201,
    }
    
    def __init__(
        self,
        resolution: Dict[str, int],
        domain: Dict[str, Tuple[float, float]] = None,
        **params
    ):
        super().__init__(resolution, domain, **params)
        
        self.alpha = self.params["diffusivity"]
        self.T = self.params.get("time_end", 1.0)
        self.n_t = self.params.get("_n_time_steps", 201)
        
        self.nx = resolution["x"]
        self.dx = self.grids["x"][1] - self.grids["x"][0]
        self.k = 2 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)
    
    def _rhs(self, u: np.ndarray, t: float) -> np.ndarray:
        """Compute RHS: α ∂²u/∂x²"""
        u_hat = np.fft.fft(u)
        u_hat_xx = -self.k**2 * u_hat
        u_xx = np.fft.ifft(u_hat_xx).real
        return self.alpha * u_xx
    
    def solve(self, ic: np.ndarray, return_full: bool = False) -> np.ndarray:
        """Solve the heat equation."""
        t = np.linspace(0, self.T, self.n_t)
        U = odeint(self._rhs, ic, t, mxstep=5000)
        
        if return_full:
            return U
        return U[-1]
    
    def generate_ic(
        self,
        generator: Union[str, Callable] = "fourier",
        generator_params: Dict = None,
        seed: int = None,
    ) -> np.ndarray:
        """Generate random initial conditions."""
        if generator_params is None:
            generator_params = {}
        
        default_params = {
            "n_modes": 10,
            "decay": 1.5,
            "amplitude": 1.0,
        }
        generator_params = {**default_params, **generator_params}
        
        if isinstance(generator, str):
            gen = get_ic_generator(generator, **generator_params)
        else:
            gen = generator
        
        return gen.generate(shape=(self.nx,), seed=seed, grid=self.grids)
    
    def validate_solution(
        self,
        ic: np.ndarray,
        solution: np.ndarray,
        tol: float = 1e-6,
    ) -> Dict:
        """Validate the solution."""
        is_valid = (
            not np.isnan(solution).any() and
            not np.isinf(solution).any()
        )
        return {'valid': is_valid, 'max_value': np.abs(solution).max()}
