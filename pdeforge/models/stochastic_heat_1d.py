"""
1D Stochastic Heat Equation Solver

The stochastic heat equation with additive noise:

    ∂u/∂t = α ∂²u/∂x² + σ Ẇ

where Ẇ is space-time white noise (or colored noise).

For spatially smooth noise, we use:
    Ẇ(x,t) = Σ_k η_k(t) e_k(x)
    
where η_k are independent Brownian motions and e_k are Fourier modes.

Operator Learning Tasks:
    1. Multiple realizations: u₀ → {u_T^(1), u_T^(2), ...}
    2. Moment-based: u₀ → (E[u_T], Var[u_T])

with periodic boundary conditions.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union, Callable, List

from pdeforge.core.base import PDEModel
from pdeforge.core.registry import register_model
from pdeforge.core.types import PDEDataset
from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.generators.initial_conditions import get_ic_generator


@register_model("stochastic_heat_1d")
class StochasticHeat1D(PDEModel):
    """
    1D Stochastic Heat equation with additive noise.
    
    ∂u/∂t = α ∂²u/∂x² + σ Ẇ
    
    Produces multiple realizations per initial condition.
    Use for learning conditional distributions or uncertainty quantification.
    
    Examples
    --------
    >>> dataset = generate_dataset(
    ...     model="stochastic_heat_1d",
    ...     n_samples=100,
    ...     resolution={"x": 256},
    ...     params={"diffusivity": 0.01, "noise_intensity": 0.1, "n_realizations": 50},
    ... )
    >>> # dataset.outputs.shape = (100, 50, 256) - 50 realizations per IC
    """
    
    NDIM = 1
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
            name="noise_intensity",
            description="Noise amplitude σ",
            default=0.1,
            param_type=ParamType.PHYSICAL,
            bounds=(0.0, 1.0),
            affects="Higher σ → more variance in outputs",
        ),
        ParamSpec(
            name="n_realizations",
            description="Number of noise realizations per IC",
            default=20,
            param_type=ParamType.OUTPUT,
            bounds=(1, 200),
            affects="More realizations → better moment estimates",
        ),
        ParamSpec(
            name="time_end",
            description="Final simulation time",
            default=1.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.01, 10.0),
            units="s",
            affects="Longer time → more noise accumulation",
        ),
    ]
    
    DEFAULT_PARAMS = {
        "diffusivity": 0.01,
        "noise_intensity": 0.1,
        "n_realizations": 20,
        "time_end": 1.0,
        "_n_time_steps": 201,
        "_noise_modes": 20,  # Number of Fourier modes for colored noise
        "_noise_correlation_length": 0.05,  # Spatial correlation (0 = white)
    }
    
    def __init__(
        self,
        resolution: Dict[str, int],
        domain: Dict[str, Tuple[float, float]] = None,
        **params
    ):
        super().__init__(resolution, domain, **params)
        
        self.alpha = self.params["diffusivity"]
        self.sigma = self.params["noise_intensity"]
        self.n_realizations = self.params.get("n_realizations", 20)
        self.T = self.params.get("time_end", 1.0)
        self.n_t = self.params.get("_n_time_steps", 201)
        self.noise_modes = self.params.get("_noise_modes", 20)
        self.corr_length = self.params.get("_noise_correlation_length", 0.05)
        
        self.nx = resolution["x"]
        self.dx = self.grids["x"][1] - self.grids["x"][0]
        self.k = 2 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)
        self.dt = self.T / (self.n_t - 1)
    
    def _generate_noise_increment(self, seed: int = None) -> np.ndarray:
        """Generate spatially correlated noise increment."""
        if seed is not None:
            np.random.seed(seed)
        
        # Generate white noise in Fourier space, then apply correlation kernel
        xi = np.random.randn(self.nx) + 1j * np.random.randn(self.nx)
        
        # Correlation kernel: Q(k) ∝ exp(-|k|² * corr_length²)
        if self.corr_length > 0:
            kernel = np.exp(-self.k**2 * self.corr_length**2 / 2)
        else:
            kernel = np.ones(self.nx)
        
        # Normalize to have unit variance in physical space
        kernel = kernel / np.sqrt(np.sum(kernel**2) / self.nx)
        
        noise = np.fft.ifft(xi * kernel).real
        return noise * np.sqrt(self.dt)
    
    def solve_single_realization(
        self, 
        ic: np.ndarray, 
        seed: int = None,
        return_full: bool = False
    ) -> np.ndarray:
        """Solve one realization of the stochastic heat equation."""
        if seed is not None:
            np.random.seed(seed)
        
        u = ic.copy()
        solutions = [u.copy()] if return_full else None
        
        # Time stepping with Euler-Maruyama
        for step in range(1, self.n_t):
            # Diffusion (spectral, implicit would be better but explicit for simplicity)
            u_hat = np.fft.fft(u)
            u_hat = u_hat * np.exp(-self.alpha * self.k**2 * self.dt)
            u = np.fft.ifft(u_hat).real
            
            # Add noise increment
            dW = self._generate_noise_increment()
            u = u + self.sigma * dW
            
            if return_full:
                solutions.append(u.copy())
        
        if return_full:
            return np.stack(solutions, axis=0)
        return u
    
    def solve(
        self, 
        ic: np.ndarray, 
        seed: int = None,
        return_full: bool = False
    ) -> np.ndarray:
        """
        Solve the stochastic heat equation with multiple realizations.
        
        Returns
        -------
        np.ndarray
            Shape: (n_realizations, nx) or (n_realizations, n_t, nx) if return_full
        """
        realizations = []
        
        for r in range(self.n_realizations):
            realization_seed = seed + r if seed is not None else None
            u_r = self.solve_single_realization(ic, realization_seed, return_full)
            realizations.append(u_r)
        
        return np.stack(realizations, axis=0)
    
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
    
    def generate_sample(
        self,
        generator: Union[str, Callable] = "fourier",
        generator_params: Dict = None,
        seed: int = None,
        validate: bool = True,
        max_attempts: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Generate IC and multiple realizations."""
        if generator_params is None:
            generator_params = {}
        
        ic = self.generate_ic(generator, generator_params, seed)
        
        # Use different seeds for IC and noise
        noise_seed = seed * 1000 if seed is not None else None
        realizations = self.solve(ic, noise_seed)
        
        validation = self.validate_solution(ic, realizations)
        
        return ic, realizations, validation
    
    def validate_solution(
        self,
        ic: np.ndarray,
        solution: np.ndarray,
        tol: float = 1e-6,
    ) -> Dict:
        """Validate the stochastic solution."""
        is_valid = (
            not np.isnan(solution).any() and
            not np.isinf(solution).any()
        )
        
        # Compute moments
        mean = solution.mean(axis=0)
        var = solution.var(axis=0)
        
        return {
            'valid': is_valid,
            'n_realizations': solution.shape[0],
            'mean_max': np.abs(mean).max(),
            'var_max': var.max(),
        }
    
    def generate_dataset(
        self,
        n_samples: int,
        ic_generator: Union[str, Callable] = "fourier",
        ic_params: Dict = None,
        seed: int = None,
        validate: bool = True,
        n_jobs: int = 1,
        verbose: bool = True,
    ) -> PDEDataset:
        """
        Generate stochastic dataset.
        
        Returns dataset where outputs have shape (n_samples, n_realizations, nx).
        """
        from tqdm import tqdm
        
        if ic_params is None:
            ic_params = {}
        
        inputs_list = []
        outputs_list = []
        
        iterator = range(n_samples)
        if verbose:
            iterator = tqdm(iterator, desc="Generating stochastic samples")
        
        for i in iterator:
            sample_seed = seed + i if seed is not None else None
            ic, realizations, _ = self.generate_sample(
                ic_generator, ic_params, sample_seed, validate
            )
            inputs_list.append(ic)
            outputs_list.append(realizations)
        
        inputs = np.stack(inputs_list, axis=0)
        outputs = np.stack(outputs_list, axis=0)
        
        return PDEDataset(
            inputs=inputs,
            outputs=outputs,
            grid=self.grids,
            input_names=self.INPUT_NAMES,
            output_names=self.OUTPUT_NAMES,
            metadata={
                "model": "stochastic_heat_1d",
                "params": self.params,
                "n_realizations": self.n_realizations,
                "stochastic": True,
            },
        )
