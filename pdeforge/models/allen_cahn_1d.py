"""
1D Allen-Cahn Equation Solver

The Allen-Cahn equation models phase separation with non-conserved dynamics:

    ∂u/∂t = ε ∂²u/∂x² + u - u³

The double-well potential f(u) = (1-u²)²/4 has minima at u = ±1.
Solutions evolve toward piecewise constant states with sharp interfaces.

with periodic boundary conditions.

Operator Learning Task:
    u(x, t=0) → u(x, t=T)
"""

from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np

from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model
from pdeforge.core.types import PDEDataset
from pdeforge.generators.initial_conditions import get_ic_generator
from pdeforge.solvers.semilinear import SemiLinearSpectralModel


@register_model("allen_cahn_1d")
class AllenCahn1D(SemiLinearSpectralModel):
    """
    1D Allen-Cahn equation for phase separation.

    ∂u/∂t = ε ∂²u/∂x² + u - u³

    Solutions evolve toward ±1 with sharp interfaces.
    Important for phase field modeling and interface dynamics.

    Examples
    --------
    >>> dataset = generate_dataset(
    ...     model="allen_cahn_1d",
    ...     n_samples=100,
    ...     resolution={"x": 256},
    ...     params={"epsilon": 0.01, "time_end": 10.0},
    ... )
    """

    NDIM = 1
    INPUT_NAMES = ["u0"]
    OUTPUT_NAMES = ["u_T"]

    USER_PARAMS = [
        ParamSpec(
            name="epsilon",
            description="Interface width parameter (smaller = sharper interfaces)",
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
            affects="Longer time → more complete phase separation",
        ),
    ]

    DEFAULT_PARAMS = {
        "epsilon": 0.01,
        "time_end": 10.0,
        "_n_time_steps": 201,
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
        self.n_t = self.params.get("_n_time_steps", 201)

        self._setup_spectral()
        self.nx = resolution["x"]
        self.dx = self.grids["x"][1] - self.grids["x"][0]
        self.k = self.K[0]

        # ETDRK4 on the seam: the stiff -eps*k^2 term AND the linear +u
        # reaction sit in L (integrated exactly); only -u^3 is stepped.
        self.dt = self.params.get("_dt") or min(0.05, self.T / 200.0)

    def linear_symbol(self):
        return -self.eps * self.k**2 + 1.0

    def nonlinear_hat(self, v, u, ops):
        return -(self.dealias * self._fft(u**3, ops))

    def generate_ic(
        self,
        generator: Union[str, Callable] = "default",
        generator_params: Dict = None,
        seed: int = None,
    ) -> np.ndarray:
        """
        Generate initial condition with values near ±1 transition.

        Uses smooth random field that transitions between phases.
        """
        if generator_params is None:
            generator_params = {}

        if seed is not None:
            np.random.seed(seed)

        x = self.grids["x"]
        L = x[-1] - x[0]

        # Random Fourier modes with low frequency to get smooth transitions
        n_modes = generator_params.get("n_modes", 5)

        u0 = np.zeros(self.nx)
        for m in range(1, n_modes + 1):
            amp = np.random.randn() / m
            phase = np.random.uniform(0, 2 * np.pi)
            u0 += amp * np.sin(2 * np.pi * m * x / L + phase)

        # Scale to have values roughly between -1 and 1
        u0 = np.tanh(u0 * 2)

        return u0

    def validate_solution(
        self,
        ic: np.ndarray,
        solution: np.ndarray,
        tol: float = 1e-6,
    ) -> Dict:
        """Validate the solution (should stay bounded near ±1)."""
        is_valid = (
            not np.isnan(solution).any()
            and not np.isinf(solution).any()
            and np.abs(solution).max() < 2.0  # Should stay near ±1
        )
        return {"valid": is_valid, "max_value": np.abs(solution).max()}
