"""
1D FitzHugh-Nagumo Equation Solver

The FitzHugh-Nagumo model describes excitable media (neurons, cardiac tissue):

    ∂u/∂t = D_u ∂²u/∂x² + u - u³ - v
    ∂v/∂t = D_v ∂²v/∂x² + ε(u - γv + β)

where:
    u: activator (fast variable, membrane potential)
    v: inhibitor (slow variable, recovery)
    ε: time scale separation (ε << 1)
    γ, β: recovery dynamics parameters

with periodic boundary conditions.

Operator Learning Task:
    (u₀, v₀) → (u_T, v_T)  or  u₀ → u_T
"""

from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
from scipy.integrate import odeint

from pdeforge.core.base import PDEModel
from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model
from pdeforge.core.types import PDEDataset
from pdeforge.generators.initial_conditions import get_ic_generator


@register_model("fitzhugh_nagumo_1d")
class FitzHughNagumo1D(PDEModel):
    """
    1D FitzHugh-Nagumo model for excitable media.

    ∂u/∂t = D_u ∂²u/∂x² + u - u³ - v
    ∂v/∂t = D_v ∂²v/∂x² + ε(u - γv + β)

    Models traveling waves, pulses, and spiral patterns in excitable systems.
    Used in neuroscience and cardiac modeling.

    Examples
    --------
    >>> dataset = generate_dataset(
    ...     model="fitzhugh_nagumo_1d",
    ...     n_samples=100,
    ...     resolution={"x": 256},
    ...     params={"epsilon": 0.08, "diffusivity_u": 1.0},
    ... )
    """

    NDIM = 1
    INPUT_NAMES = ["u0", "v0"]
    OUTPUT_NAMES = ["u_T", "v_T"]

    USER_PARAMS = [
        ParamSpec(
            name="epsilon",
            description="Time scale separation (ε << 1 for excitable dynamics)",
            default=0.08,
            param_type=ParamType.PHYSICAL,
            bounds=(0.01, 0.5),
            affects="Smaller ε → sharper pulses, slower recovery",
        ),
        ParamSpec(
            name="diffusivity_u",
            description="Diffusion coefficient for activator u",
            default=1.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.01, 10.0),
            units="m²/s",
            affects="Controls wave speed",
        ),
        ParamSpec(
            name="time_end",
            description="Final simulation time",
            default=50.0,
            param_type=ParamType.PHYSICAL,
            bounds=(1.0, 200.0),
            units="s",
            affects="Time for wave propagation",
        ),
    ]

    DEFAULT_PARAMS = {
        "epsilon": 0.08,
        "diffusivity_u": 1.0,
        "diffusivity_v": 0.0,  # Often v doesn't diffuse
        "gamma": 0.8,
        "beta": 0.7,
        "time_end": 50.0,
        "_n_time_steps": 201,
    }

    def __init__(
        self,
        resolution: Dict[str, int],
        domain: Dict[str, Tuple[float, float]] = None,
        **params,
    ):
        if domain is None:
            domain = {"x": (0.0, 100.0)}  # Larger domain for wave propagation
        super().__init__(resolution, domain, **params)

        self.eps = self.params["epsilon"]
        self.D_u = self.params["diffusivity_u"]
        self.D_v = self.params.get("diffusivity_v", 0.0)
        self.gamma = self.params.get("gamma", 0.8)
        self.beta = self.params.get("beta", 0.7)
        self.T = self.params.get("time_end", 50.0)
        self.n_t = self.params.get("_n_time_steps", 201)

        self.nx = resolution["x"]
        self.dx = self.grids["x"][1] - self.grids["x"][0]
        self.k = 2 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)

    def _laplacian(self, u: np.ndarray) -> np.ndarray:
        """Compute Laplacian spectrally."""
        u_hat = np.fft.fft(u)
        return np.fft.ifft(-self.k**2 * u_hat).real

    def _rhs(self, state: np.ndarray, t: float) -> np.ndarray:
        """Compute RHS of FitzHugh-Nagumo system."""
        u = state[: self.nx]
        v = state[self.nx :]

        # Reaction terms
        f_u = u - u**3 - v
        f_v = self.eps * (u - self.gamma * v + self.beta)

        # Diffusion
        du_dt = self.D_u * self._laplacian(u) + f_u
        dv_dt = self.D_v * self._laplacian(v) + f_v

        return np.concatenate([du_dt, dv_dt])

    def solve(
        self, ic: np.ndarray, ic_v: np.ndarray = None, return_full: bool = False
    ) -> np.ndarray:
        """
        Solve the FitzHugh-Nagumo equations.

        Parameters
        ----------
        ic : np.ndarray
            Initial condition for u
        ic_v : np.ndarray, optional
            Initial condition for v. If None, use steady state v = (u + β)/γ
        return_full : bool
            If True, return full trajectory
        """
        # Default v initial condition
        if ic_v is None:
            ic_v = (ic + self.beta) / self.gamma

        state0 = np.concatenate([ic, ic_v])

        t = np.linspace(0, self.T, self.n_t)
        states = odeint(self._rhs, state0, t, mxstep=10000)

        # Extract u and v
        U = states[:, : self.nx]
        V = states[:, self.nx :]

        if return_full:
            return np.stack([U, V], axis=-1)  # Shape: (n_t, nx, 2)

        # Return final state as (nx, 2)
        return np.stack([U[-1], V[-1]], axis=-1)

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
        L = x[-1] - x[0]

        # Steady state with localized perturbation (stimulus)
        u0 = np.zeros(self.nx)

        # Random perturbation location and width
        center = np.random.uniform(0.1 * L, 0.3 * L) + x[0]
        width = np.random.uniform(1.0, 5.0)
        amplitude = np.random.uniform(0.5, 1.5)

        u0 = amplitude * np.exp(-(((x - center) / width) ** 2))

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
                if validation["valid"]:
                    # Return u0 as input, (u_T, v_T) as output
                    return ic, solution, validation
            else:
                return ic, solution, {"valid": True}

        raise RuntimeError(
            f"Failed to generate valid sample after {max_attempts} attempts"
        )

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
            and np.abs(solution).max() < 100
        )
        return {"valid": is_valid, "max_value": np.abs(solution).max()}
