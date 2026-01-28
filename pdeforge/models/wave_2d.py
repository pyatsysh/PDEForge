"""
2D Wave Equation Solver

The wave equation models oscillatory phenomena:

    ∂²u/∂t² = c² ∇²u = c² (∂²u/∂x² + ∂²u/∂y²)

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


@register_model("wave_2d")
class Wave2D(PDEModel):
    """
    2D Wave equation for oscillatory dynamics.
    
    ∂²u/∂t² = c² ∇²u
    
    Solutions are circular waves emanating from disturbances.
    
    Examples
    --------
    >>> dataset = generate_dataset(
    ...     model="wave_2d",
    ...     n_samples=100,
    ...     resolution={"x": 64, "y": 64},
    ...     params={"wave_speed": 1.0, "time_end": 2.0},
    ... )
    """
    
    NDIM = 2
    INPUT_NAMES = ["u0"]
    OUTPUT_NAMES = ["u_T"]
    
    USER_PARAMS = [
        ParamSpec(
            name="wave_speed",
            description="Wave propagation speed c",
            default=1.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.1, 10.0),
            units="m/s",
            affects="Higher speed → faster wave propagation",
        ),
        ParamSpec(
            name="time_end",
            description="Final time for solution",
            default=2.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.1, 20.0),
            units="s",
            affects="Controls how far waves travel",
        ),
    ]
    
    DEFAULT_PARAMS = {
        "wave_speed": 1.0,
        "time_end": 2.0,
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
        
        self.c = self.params["wave_speed"]
        self.T = self.params.get("time_end", 2.0)
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
        self.K = np.sqrt(self.K2)
        
        # Avoid division by zero at k=0
        self.K_safe = np.where(self.K == 0, 1, self.K)
    
    def solve(self, ic: np.ndarray, return_full: bool = False) -> np.ndarray:
        """
        Solve using exact solution in Fourier space.
        
        For wave equation with zero initial velocity:
        û(k,t) = û(k,0) * cos(c|k|t)
        """
        t_array = np.linspace(0, self.T, self.n_t)
        
        # Transform IC
        u_hat_0 = np.fft.fft2(ic)
        
        if return_full:
            solutions = []
            for t in t_array:
                # Exact solution: u(k,t) = u0(k) * cos(c|k|t)
                u_hat_t = u_hat_0 * np.cos(self.c * self.K * t)
                u_t = np.fft.ifft2(u_hat_t).real
                solutions.append(u_t)
            return np.stack(solutions, axis=0)
        else:
            u_hat_T = u_hat_0 * np.cos(self.c * self.K * self.T)
            return np.fft.ifft2(u_hat_T).real
    
    def generate_ic(
        self,
        generator: Union[str, Callable] = "fourier",
        generator_params: Dict = None,
        seed: int = None,
    ) -> np.ndarray:
        """Generate random 2D initial displacement."""
        if generator_params is None:
            generator_params = {}
        
        default_params = {
            "n_modes": 5,
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
