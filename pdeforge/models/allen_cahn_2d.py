"""
2D Allen-Cahn Equation Solver

The Allen-Cahn equation models phase separation:

    ∂u/∂t = ε ∇²u + u - u³

In 2D, this produces droplet formation, coarsening, and interface motion.

with periodic boundary conditions.

Operator Learning Task:
    u(x, y, t=0) → u(x, y, t=T)
"""

from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np

from pdeforge.core.base import PDEModel
from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model
from pdeforge.core.types import PDEDataset
from pdeforge.generators.initial_conditions import get_ic_generator


@register_model("allen_cahn_2d")
class AllenCahn2D(PDEModel):
    """
    2D Allen-Cahn equation for phase separation.

    ∂u/∂t = ε ∇²u + u - u³

    Solutions evolve toward ±1 domains with sharp interfaces.
    Produces droplet formation and coarsening dynamics.

    Examples
    --------
    >>> dataset = generate_dataset(
    ...     model="allen_cahn_2d",
    ...     n_samples=50,
    ...     resolution={"x": 64, "y": 64},
    ...     params={"epsilon": 0.01, "time_end": 10.0},
    ... )
    """

    NDIM = 2
    INPUT_NAMES = ["u0"]
    OUTPUT_NAMES = ["u_T"]

    USER_PARAMS = [
        ParamSpec(
            name="epsilon",
            description="Interface width parameter",
            default=0.01,
            param_type=ParamType.PHYSICAL,
            bounds=(0.001, 0.5),
            affects="Smaller ε → sharper phase boundaries",
        ),
        ParamSpec(
            name="time_end",
            description="Final simulation time",
            default=10.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.1, 100.0),
            units="s",
            affects="Longer time → coarser domains (droplet coarsening)",
        ),
    ]

    DEFAULT_PARAMS = {
        "epsilon": 0.01,
        "time_end": 10.0,
        "_n_time_steps": 101,
        "_dt": 0.01,
    }

    def __init__(
        self,
        resolution: Dict[str, int],
        domain: Dict[str, Tuple[float, float]] = None,
        **params,
    ):
        super().__init__(resolution, domain, **params)

        self.eps = self.params["epsilon"]
        self.T = self.params.get("time_end", 10.0)
        self.n_t = self.params.get("_n_time_steps", 101)
        self.dt = self.params.get("_dt", 0.01)

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
        Solve using semi-implicit (IMEX) time stepping.

        Treats diffusion implicitly, reaction explicitly.
        """
        u = ic.copy()
        dt = self.dt
        n_substeps = int(np.ceil(self.T / dt))
        output_interval = max(1, n_substeps // (self.n_t - 1))

        # Precompute implicit diffusion factor
        # (1 - ε*dt*∇²)u_new = u_old + dt*(u - u³)
        # In Fourier: (1 + ε*dt*k²)û_new = ...
        implicit_factor = 1.0 / (1.0 + self.eps * dt * self.K2)

        solutions = [u.copy()]

        for step in range(n_substeps):
            # Explicit reaction step
            reaction = u - u**3
            u_star = u + dt * reaction

            # Implicit diffusion step
            u_hat = np.fft.fft2(u_star)
            u_hat = u_hat * implicit_factor
            u = np.fft.ifft2(u_hat).real

            # Store for output
            if (step + 1) % output_interval == 0 and len(solutions) < self.n_t:
                solutions.append(u.copy())

        # Ensure we have n_t outputs
        while len(solutions) < self.n_t:
            solutions.append(u.copy())
        solutions = solutions[: self.n_t]

        if return_full:
            return np.stack(solutions, axis=0)
        return solutions[-1]

    def generate_ic(
        self,
        generator: Union[str, Callable] = "default",
        generator_params: Dict = None,
        seed: int = None,
    ) -> np.ndarray:
        """Generate random initial condition with phase mixture."""
        if generator_params is None:
            generator_params = {}

        if seed is not None:
            np.random.seed(seed)

        x = self.grids["x"]
        y = self.grids["y"]
        X, Y = np.meshgrid(x, y)

        Lx = x[-1] - x[0]
        Ly = y[-1] - y[0]

        # Random Fourier modes
        n_modes = generator_params.get("n_modes", 5)

        u0 = np.zeros((self.ny, self.nx))
        for mx in range(1, n_modes + 1):
            for my in range(1, n_modes + 1):
                amp = np.random.randn() / (mx + my)
                phase = np.random.uniform(0, 2 * np.pi)
                u0 += amp * np.sin(
                    2 * np.pi * mx * X / Lx + 2 * np.pi * my * Y / Ly + phase
                )

        # Scale to have values roughly between -1 and 1
        u0 = np.tanh(u0 * 2)

        return u0

    def validate_solution(
        self,
        ic: np.ndarray,
        solution: np.ndarray,
        tol: float = 1e-6,
    ) -> Dict:
        """Validate the solution."""
        is_valid = (
            not np.isnan(solution).any()
            and not np.isinf(solution).any()
            and np.abs(solution).max() < 2.0
        )
        return {"valid": is_valid, "max_value": np.abs(solution).max()}
