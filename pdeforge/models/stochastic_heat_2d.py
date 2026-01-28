"""
2D Stochastic Heat Equation Solver

The stochastic heat equation with additive noise:

    ∂u/∂t = α ∇²u + σ Ẇ

where Ẇ is space-time noise (white or colored).

Operator Learning Tasks:
    1. Multiple realizations: u₀ → {u_T^(1), u_T^(2), ...}
    2. Moment-based: u₀ → (E[u_T], Var[u_T])

with periodic boundary conditions.
"""

from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np

from pdeforge.core.base import PDEModel
from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model
from pdeforge.core.types import PDEDataset
from pdeforge.generators.initial_conditions import get_ic_generator


@register_model("stochastic_heat_2d")
class StochasticHeat2D(PDEModel):
    """
    2D Stochastic Heat equation with additive noise.

    ∂u/∂t = α ∇²u + σ Ẇ

    Produces multiple realizations per initial condition.

    Examples
    --------
    >>> dataset = generate_dataset(
    ...     model="stochastic_heat_2d",
    ...     n_samples=50,
    ...     resolution={"x": 64, "y": 64},
    ...     params={"diffusivity": 0.01, "noise_intensity": 0.1, "n_realizations": 20},
    ... )
    >>> # dataset.outputs.shape = (50, 20, 64, 64)
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
        "_n_time_steps": 101,
        "_noise_correlation_length": 0.05,
    }

    def __init__(
        self,
        resolution: Dict[str, int],
        domain: Dict[str, Tuple[float, float]] = None,
        **params,
    ):
        super().__init__(resolution, domain, **params)

        self.alpha = self.params["diffusivity"]
        self.sigma = self.params["noise_intensity"]
        self.n_realizations = self.params.get("n_realizations", 20)
        self.T = self.params.get("time_end", 1.0)
        self.n_t = self.params.get("_n_time_steps", 101)
        self.corr_length = self.params.get("_noise_correlation_length", 0.05)

        self.nx = resolution["x"]
        self.ny = resolution["y"]

        dx = self.grids["x"][1] - self.grids["x"][0]
        dy = self.grids["y"][1] - self.grids["y"][0]
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ny, d=dy)
        self.KX, self.KY = np.meshgrid(kx, ky)
        self.K2 = self.KX**2 + self.KY**2

        self.dt = self.T / (self.n_t - 1)

        # Precompute noise correlation kernel
        if self.corr_length > 0:
            self.noise_kernel = np.exp(-self.K2 * self.corr_length**2 / 2)
            self.noise_kernel = self.noise_kernel / np.sqrt(
                np.sum(self.noise_kernel**2) / (self.nx * self.ny)
            )
        else:
            self.noise_kernel = np.ones((self.ny, self.nx))

    def _generate_noise_increment(self, seed: int = None) -> np.ndarray:
        """Generate 2D spatially correlated noise increment."""
        if seed is not None:
            np.random.seed(seed)

        xi = np.random.randn(self.ny, self.nx) + 1j * np.random.randn(self.ny, self.nx)
        noise = np.fft.ifft2(
            np.fft.fft2(np.random.randn(self.ny, self.nx)) * self.noise_kernel
        ).real

        return noise * np.sqrt(self.dt)

    def solve_single_realization(
        self, ic: np.ndarray, seed: int = None, return_full: bool = False
    ) -> np.ndarray:
        """Solve one realization."""
        if seed is not None:
            np.random.seed(seed)

        u = ic.copy()
        solutions = [u.copy()] if return_full else None

        # Precompute diffusion operator
        diffusion_factor = np.exp(-self.alpha * self.K2 * self.dt)

        for step in range(1, self.n_t):
            # Diffusion step (exact in Fourier space)
            u_hat = np.fft.fft2(u)
            u_hat = u_hat * diffusion_factor
            u = np.fft.ifft2(u_hat).real

            # Add noise
            dW = self._generate_noise_increment()
            u = u + self.sigma * dW

            if return_full:
                solutions.append(u.copy())

        if return_full:
            return np.stack(solutions, axis=0)
        return u

    def solve(
        self, ic: np.ndarray, seed: int = None, return_full: bool = False
    ) -> np.ndarray:
        """
        Solve with multiple realizations.

        Returns shape: (n_realizations, ny, nx) or (n_realizations, n_t, ny, nx)
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
        is_valid = not np.isnan(solution).any() and not np.isinf(solution).any()

        mean = solution.mean(axis=0)
        var = solution.var(axis=0)

        return {
            "valid": is_valid,
            "n_realizations": solution.shape[0],
            "mean_max": np.abs(mean).max(),
            "var_max": var.max(),
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
        """Generate stochastic dataset."""
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
                "model": "stochastic_heat_2d",
                "params": self.params,
                "n_realizations": self.n_realizations,
                "stochastic": True,
            },
        )
