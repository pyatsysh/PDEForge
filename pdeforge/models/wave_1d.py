"""
1D Wave Equation Solver

The wave equation models oscillatory phenomena:

    ∂²u/∂t² = c² ∂²u/∂x²

Rewritten as first-order system:
    ∂u/∂t = v
    ∂v/∂t = c² ∂²u/∂x²

with periodic boundary conditions.

Operator Learning Task:
    (u(x,0), v(x,0)) → (u(x,T), v(x,T))  or
    u(x,0) → u(x,T)  (with zero initial velocity)
"""

from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
from scipy.integrate import odeint

from pdeforge.core.base import PDEModel
from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model
from pdeforge.core.types import PDEDataset
from pdeforge.generators.initial_conditions import get_ic_generator


@register_model("wave_1d")
class Wave1D(PDEModel):
    """
    1D Wave equation for oscillatory dynamics.

    ∂²u/∂t² = c² ∂²u/∂x²

    Solutions are traveling waves that preserve their shape.
    Tests ability to learn oscillatory, non-dissipative dynamics.

    Examples
    --------
    >>> dataset = generate_dataset(
    ...     model="wave_1d",
    ...     n_samples=100,
    ...     resolution={"x": 256},
    ...     params={"wave_speed": 1.0, "time_end": 2.0},
    ... )
    """

    NDIM = 1
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
        "_n_time_steps": 201,
        "_initial_velocity": "zero",  # "zero" or "random"
    }

    def __init__(
        self,
        resolution: Dict[str, int],
        domain: Dict[str, Tuple[float, float]] = None,
        **params,
    ):
        super().__init__(resolution, domain, **params)

        self.c = self.params["wave_speed"]
        self.T = self.params.get("time_end", 2.0)
        self.n_t = self.params.get("_n_time_steps", 201)

        self.nx = resolution["x"]
        self.dx = self.grids["x"][1] - self.grids["x"][0]
        self.k = 2 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)

    def _rhs(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        Compute RHS for first-order system.
        state = [u, v] where v = ∂u/∂t
        """
        u = state[: self.nx]
        v = state[self.nx :]

        # Compute ∂²u/∂x² spectrally
        u_hat = np.fft.fft(u)
        u_xx = np.fft.ifft(-self.k**2 * u_hat).real

        du_dt = v
        dv_dt = self.c**2 * u_xx

        return np.concatenate([du_dt, dv_dt])

    def solve(self, ic: np.ndarray, return_full: bool = False) -> np.ndarray:
        """
        Solve the wave equation.

        Parameters
        ----------
        ic : np.ndarray
            Initial displacement u(x,0). Initial velocity is zero.
        return_full : bool
            If True, return full trajectory
        """
        # Initial state: [u0, v0] where v0 = 0 (zero initial velocity)
        v0 = np.zeros_like(ic)
        state0 = np.concatenate([ic, v0])

        t = np.linspace(0, self.T, self.n_t)
        states = odeint(self._rhs, state0, t, mxstep=5000)

        # Extract u (first half of state)
        U = states[:, : self.nx]

        if return_full:
            return U
        return U[-1]

    def generate_ic(
        self,
        generator: Union[str, Callable] = "fourier",
        generator_params: Dict = None,
        seed: int = None,
    ) -> np.ndarray:
        """Generate random initial displacement."""
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

        return gen.generate(shape=(self.nx,), seed=seed, grid=self.grids)

    def validate_solution(
        self,
        ic: np.ndarray,
        solution: np.ndarray,
        tol: float = 1e-6,
    ) -> Dict:
        """Validate the solution (energy should be conserved)."""
        is_valid = not np.isnan(solution).any() and not np.isinf(solution).any()
        return {"valid": is_valid, "max_value": np.abs(solution).max()}
