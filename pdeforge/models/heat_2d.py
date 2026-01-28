"""
2D Heat Equation Solver

The heat equation models diffusion/conduction:

    ∂u/∂t = α ∇²u = α (∂²u/∂x² + ∂²u/∂y²)

with periodic boundary conditions.

Operator Learning Task:
    u(x, y, t=0) → u(x, y, t=T)
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union, Callable

from pdeforge.core.base import PDEModel
from pdeforge.core.registry import register_model
from pdeforge.core.types import PDEDataset
from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.generators.initial_conditions import get_ic_generator


@register_model("heat_2d")
class Heat2D(PDEModel):
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
        **params
    ):
        super().__init__(resolution, domain, **params)
        
        self.alpha = self.params["diffusivity"]
        self.T = self.params.get("time_end", 1.0)
        self.n_t = self.params.get("_n_time_steps", 101)
        
        self.nx = resolution["x"]
        self.ny = resolution["y"]
        
        # Wavenumbers
        dx = self.grids["x"][1] - self.grids["x"][0]
        dy = self.grids["y"][1] - self.grids["y"][0]
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ny, d=dy)
        self.KX, self.KY = np.meshgrid(kx, ky)
        self.K2 = self.KX**2 + self.KY**2
    
    def solve(self, ic: np.ndarray, return_full: bool = False) -> np.ndarray:
        """
        Solve using exact solution in Fourier space.
        
        For heat equation: û(k,t) = û(k,0) * exp(-α|k|²t)
        """
        t_array = np.linspace(0, self.T, self.n_t)
        
        # Transform IC
        u_hat_0 = np.fft.fft2(ic)
        
        if return_full:
            solutions = []
            for t in t_array:
                # Exact solution in Fourier space
                u_hat_t = u_hat_0 * np.exp(-self.alpha * self.K2 * t)
                u_t = np.fft.ifft2(u_hat_t).real
                solutions.append(u_t)
            return np.stack(solutions, axis=0)
        else:
            # Final time only
            u_hat_T = u_hat_0 * np.exp(-self.alpha * self.K2 * self.T)
            return np.fft.ifft2(u_hat_T).real
    
    def generate_ic(
        self,
        generator: Union[str, Callable] = "fourier",
        generator_params: Dict = None,
        seed: int = None,
    ) -> np.ndarray:
        """Generate random 2D initial conditions."""
        if generator_params is None:
            generator_params = {}
        
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
        is_valid = (
            not np.isnan(solution).any() and
            not np.isinf(solution).any()
        )
        return {'valid': is_valid, 'max_value': np.abs(solution).max()}
