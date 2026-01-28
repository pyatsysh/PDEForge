"""
2D FitzHugh-Nagumo Equation Solver

The FitzHugh-Nagumo model describes excitable media (neurons, cardiac tissue):

    ∂u/∂t = D_u ∇²u + u - u³ - v
    ∂v/∂t = D_v ∇²v + ε(u - γv + β)

with periodic boundary conditions.

In 2D, this can produce spiral waves and target patterns.

Operator Learning Task:
    (u₀, v₀) → (u_T, v_T)
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union, Callable

from pdeforge.core.base import PDEModel
from pdeforge.core.registry import register_model
from pdeforge.core.types import PDEDataset
from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.generators.initial_conditions import get_ic_generator


@register_model("fitzhugh_nagumo_2d")
class FitzHughNagumo2D(PDEModel):
    """
    2D FitzHugh-Nagumo model for excitable media.
    
    ∂u/∂t = D_u ∇²u + u - u³ - v
    ∂v/∂t = D_v ∇²v + ε(u - γv + β)
    
    Can produce spiral waves and complex spatiotemporal patterns.
    
    Examples
    --------
    >>> dataset = generate_dataset(
    ...     model="fitzhugh_nagumo_2d",
    ...     n_samples=50,
    ...     resolution={"x": 64, "y": 64},
    ...     params={"epsilon": 0.08, "time_end": 50.0},
    ... )
    """
    
    NDIM = 2
    INPUT_NAMES = ["u0", "v0"]
    OUTPUT_NAMES = ["u_T", "v_T"]
    
    USER_PARAMS = [
        ParamSpec(
            name="epsilon",
            description="Time scale separation (ε << 1 for excitable dynamics)",
            default=0.08,
            param_type=ParamType.PHYSICAL,
            bounds=(0.01, 0.5),
            affects="Smaller ε → sharper waves, spiral patterns",
        ),
        ParamSpec(
            name="diffusivity_u",
            description="Diffusion coefficient for activator u",
            default=1.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.01, 10.0),
            units="m²/s",
            affects="Controls wave speed and pattern scale",
        ),
        ParamSpec(
            name="time_end",
            description="Final simulation time",
            default=50.0,
            param_type=ParamType.PHYSICAL,
            bounds=(1.0, 200.0),
            units="s",
            affects="Time for pattern development",
        ),
    ]
    
    DEFAULT_PARAMS = {
        "epsilon": 0.08,
        "diffusivity_u": 1.0,
        "diffusivity_v": 0.0,
        "gamma": 0.8,
        "beta": 0.7,
        "time_end": 50.0,
        "_n_time_steps": 101,
        "_dt": 0.1,
    }
    
    def __init__(
        self,
        resolution: Dict[str, int],
        domain: Dict[str, Tuple[float, float]] = None,
        **params
    ):
        if domain is None:
            domain = {"x": (0.0, 50.0), "y": (0.0, 50.0)}
        super().__init__(resolution, domain, **params)
        
        self.eps = self.params["epsilon"]
        self.D_u = self.params["diffusivity_u"]
        self.D_v = self.params.get("diffusivity_v", 0.0)
        self.gamma = self.params.get("gamma", 0.8)
        self.beta = self.params.get("beta", 0.7)
        self.T = self.params.get("time_end", 50.0)
        self.n_t = self.params.get("_n_time_steps", 101)
        self.dt = self.params.get("_dt", 0.1)
        
        self.nx = resolution["x"]
        self.ny = resolution["y"]
        
        # Wavenumbers for spectral Laplacian
        dx = self.grids["x"][1] - self.grids["x"][0]
        dy = self.grids["y"][1] - self.grids["y"][0]
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ny, d=dy)
        self.KX, self.KY = np.meshgrid(kx, ky)
        self.K2 = self.KX**2 + self.KY**2
    
    def _laplacian(self, u: np.ndarray) -> np.ndarray:
        """Compute 2D Laplacian spectrally."""
        u_hat = np.fft.fft2(u)
        return np.fft.ifft2(-self.K2 * u_hat).real
    
    def solve(
        self, 
        ic: np.ndarray, 
        ic_v: np.ndarray = None,
        return_full: bool = False
    ) -> np.ndarray:
        """
        Solve using semi-implicit time stepping.
        
        Parameters
        ----------
        ic : np.ndarray
            Initial condition for u, shape (ny, nx)
        ic_v : np.ndarray, optional
            Initial condition for v
        return_full : bool
            If True, return full trajectory
        """
        # Initialize
        u = ic.copy()
        if ic_v is None:
            v = (u + self.beta) / self.gamma
        else:
            v = ic_v.copy()
        
        t_output = np.linspace(0, self.T, self.n_t)
        dt = self.dt
        n_substeps = int(np.ceil(self.T / dt))
        output_interval = max(1, n_substeps // (self.n_t - 1))
        
        solutions = [(u.copy(), v.copy())]
        
        # Time stepping with operator splitting
        for step in range(n_substeps):
            # 1. Diffusion step (spectral, exact)
            u_hat = np.fft.fft2(u)
            v_hat = np.fft.fft2(v)
            
            exp_u = np.exp(-self.D_u * self.K2 * dt)
            exp_v = np.exp(-self.D_v * self.K2 * dt) if self.D_v > 0 else 1.0
            
            u = np.fft.ifft2(u_hat * exp_u).real
            v = np.fft.ifft2(v_hat * exp_v).real if self.D_v > 0 else v
            
            # 2. Reaction step (forward Euler)
            f_u = u - u**3 - v
            f_v = self.eps * (u - self.gamma * v + self.beta)
            
            u = u + dt * f_u
            v = v + dt * f_v
            
            # Store for output
            if (step + 1) % output_interval == 0 and len(solutions) < self.n_t:
                solutions.append((u.copy(), v.copy()))
        
        # Ensure we have n_t outputs
        while len(solutions) < self.n_t:
            solutions.append((u.copy(), v.copy()))
        solutions = solutions[:self.n_t]
        
        if return_full:
            # Shape: (n_t, ny, nx, 2)
            return np.stack([np.stack([s[0], s[1]], axis=-1) for s in solutions], axis=0)
        
        # Return final state as (ny, nx, 2)
        return np.stack([solutions[-1][0], solutions[-1][1]], axis=-1)
    
    def generate_ic(
        self,
        generator: Union[str, Callable] = "default",
        generator_params: Dict = None,
        seed: int = None,
    ) -> np.ndarray:
        """Generate initial condition with localized perturbation."""
        if generator_params is None:
            generator_params = {}
        
        if seed is not None:
            np.random.seed(seed)
        
        x = self.grids["x"]
        y = self.grids["y"]
        X, Y = np.meshgrid(x, y)
        
        Lx = x[-1] - x[0]
        Ly = y[-1] - y[0]
        
        # Random localized perturbation
        cx = np.random.uniform(0.2 * Lx, 0.5 * Lx) + x[0]
        cy = np.random.uniform(0.2 * Ly, 0.5 * Ly) + y[0]
        width = np.random.uniform(2.0, 5.0)
        amplitude = np.random.uniform(0.5, 1.5)
        
        u0 = amplitude * np.exp(-((X - cx)**2 + (Y - cy)**2) / width**2)
        
        return u0
    
    def generate_sample(
        self,
        generator: Union[str, Callable] = "default",
        generator_params: Dict = None,
        seed: int = None,
        validate: bool = True,
        max_attempts: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Generate a single sample."""
        if generator_params is None:
            generator_params = {}
        
        for attempt in range(max_attempts):
            current_seed = seed + attempt if seed is not None else None
            
            ic = self.generate_ic(generator, generator_params, current_seed)
            solution = self.solve(ic)
            
            if validate:
                validation = self.validate_solution(ic, solution)
                if validation['valid']:
                    return ic, solution, validation
            else:
                return ic, solution, {'valid': True}
        
        raise RuntimeError(f"Failed to generate valid sample after {max_attempts} attempts")
    
    def validate_solution(
        self,
        ic: np.ndarray,
        solution: np.ndarray,
        tol: float = 1e-6,
    ) -> Dict:
        """Validate the solution."""
        is_valid = (
            not np.isnan(solution).any() and
            not np.isinf(solution).any() and
            np.abs(solution).max() < 100
        )
        return {'valid': is_valid, 'max_value': np.abs(solution).max()}
