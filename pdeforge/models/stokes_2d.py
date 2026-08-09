"""
2D Stokes Flow Solver

The Stokes equations describe creeping (low Reynolds number) viscous flow:

    -μ ∇²u + ∇p = f       (momentum)
    ∇·u = 0               (incompressibility)

where u = (u, v) is velocity, p is pressure, and f = (fx, fy) is body force.

Operator Learning Task:
    (fx, fy) → (u, v, p)
"""

from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np

from pdeforge.core.base import PDEModel
from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model
from pdeforge.core.types import PDEDataset
from pdeforge.generators.forcing import FourierForcingGenerator


@register_model("stokes_2d")
class Stokes2D(PDEModel):
    """
    2D Stokes flow on a periodic domain.

    -μ ∇²u + ∇p = f,  ∇·u = 0

    This model maps body force fields to velocity and pressure:
    (fx, fy) → (u, v, p)

    Use this for learning incompressible flow operators. The spectral solver
    enforces the divergence-free constraint exactly via Leray projection.

    Examples
    --------
    >>> dataset = generate_dataset(
    ...     model="stokes_2d",
    ...     n_samples=100,
    ...     resolution={"x": 64, "y": 64},
    ...     params={"viscosity": 1.0},
    ... )
    """

    NDIM = 2
    TIME_DEPENDENT = False  # steady elliptic solve
    INPUT_NAMES = ["fx", "fy"]
    OUTPUT_NAMES = ["u", "v", "p"]

    # User-facing parameters
    USER_PARAMS = [
        ParamSpec(
            name="viscosity",
            description="Dynamic viscosity of the fluid",
            default=1.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.01, 100.0),
            units="Pa·s",
            affects="Higher viscosity → smoother velocity fields",
        ),
        ParamSpec(
            name="force_complexity",
            description="Complexity of random forcing (number of Fourier modes)",
            default=5,
            param_type=ParamType.INPUT,
            bounds=(1, 20),
            affects="More modes → more complex, multi-scale forcing",
        ),
    ]

    # Internal defaults
    DEFAULT_PARAMS = {
        "viscosity": 1.0,
        "force_complexity": 5,
        "_force_decay": 1.5,  # Internal: Fourier coefficient decay rate
    }

    def __init__(
        self,
        resolution: Dict[str, int],
        domain: Dict[str, Tuple[float, float]] = None,
        **params,
    ):
        super().__init__(resolution, domain, **params)

        # Grid parameters
        self.nx = resolution.get("x", 64)
        self.ny = resolution.get("y", 64)
        self.dx = self.grids["x"][1] - self.grids["x"][0]
        self.dy = self.grids["y"][1] - self.grids["y"][0]

        self.mu = self.params["viscosity"]

        # Create meshgrid
        self.X, self.Y = np.meshgrid(self.grids["x"], self.grids["y"])

        # Wavenumbers
        self.kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)
        self.ky = 2 * np.pi * np.fft.fftfreq(self.ny, d=self.dy)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky)
        self.K2 = self.KX**2 + self.KY**2

        # Force generator (use user-facing param name)
        n_modes = self.params.get(
            "force_complexity", self.params.get("n_force_modes", 5)
        )
        self.force_generator = FourierForcingGenerator(
            n_modes=n_modes,
            decay=self.params.get("_force_decay", 1.5),
            amplitude=1.0,
            n_components=2,
            normalize=True,
        )

    def solve(self, force: np.ndarray, return_components: bool = True) -> np.ndarray:
        """
        Solve the Stokes equations using spectral method with Leray projection.

        Parameters
        ----------
        force : np.ndarray
            Body force, either shape (ny, nx, 2) or tuple (fx, fy)
        return_components : bool
            If True, return (u, v, p) as separate arrays

        Returns
        -------
        np.ndarray or Tuple
            Velocity and pressure fields
        """
        # Handle different input formats
        if isinstance(force, tuple):
            fx, fy = force
        else:
            fx = force[:, :, 0] if force.ndim == 3 else force[0]
            fy = force[:, :, 1] if force.ndim == 3 else force[1]

        # FFT of force
        fx_hat = np.fft.fft2(fx)
        fy_hat = np.fft.fft2(fy)

        # Initialize solution
        u_hat = np.zeros_like(fx_hat, dtype=complex)
        v_hat = np.zeros_like(fy_hat, dtype=complex)
        p_hat = np.zeros_like(fx_hat, dtype=complex)

        # Mask for non-zero wavenumbers
        mask = self.K2 > 1e-14

        # k · f
        k_dot_f = self.KX * fx_hat + self.KY * fy_hat

        # Leray projection: project force onto divergence-free subspace
        # u_hat = P f_hat / (μ |k|²)  where P = I - k⊗k/|k|²
        u_hat[mask] = (fx_hat[mask] - self.KX[mask] * k_dot_f[mask] / self.K2[mask]) / (
            self.mu * self.K2[mask]
        )
        v_hat[mask] = (fy_hat[mask] - self.KY[mask] * k_dot_f[mask] / self.K2[mask]) / (
            self.mu * self.K2[mask]
        )

        # Pressure from: ∇p = f + μ∇²u
        p_hat[mask] = -1j * k_dot_f[mask] / self.K2[mask]

        # Transform back to physical space
        u = np.fft.ifft2(u_hat).real
        v = np.fft.ifft2(v_hat).real
        p = np.fft.ifft2(p_hat).real
        p = p - p.mean()  # Zero mean pressure

        if return_components:
            return u, v, p
        else:
            # Stack into single array
            return np.stack([u, v, p], axis=-1)

    def generate_ic(
        self,
        generator: Union[str, Callable] = "fourier",
        generator_params: Dict = None,
        seed: int = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate random body force field.

        Parameters
        ----------
        generator : str or Callable
            Generator type or custom function
        generator_params : Dict, optional
            Generator parameters
        seed : int, optional
            Random seed

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Force components (fx, fy)
        """
        if generator_params is None:
            generator_params = {}

        # Use the force generator
        fx, fy = self.force_generator.generate(
            shape=(self.ny, self.nx),
            seed=seed,
            grid=self.grids,
        )

        return np.stack([fx, fy], axis=-1)

    def generate_sample(
        self,
        generator: Union[str, Callable] = "fourier",
        generator_params: Dict = None,
        seed: int = None,
        validate: bool = True,
        max_attempts: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate a single (force, solution) sample.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, Dict]
            (force, solution, info) where force has shape (ny, nx, 2)
            and solution has shape (ny, nx, 3) for (u, v, p)
        """
        if generator_params is None:
            generator_params = {}

        for attempt in range(max_attempts):
            current_seed = seed + attempt if seed is not None else None

            # Generate force
            force = self.generate_ic(
                generator=generator,
                generator_params=generator_params,
                seed=current_seed,
            )

            # Solve
            u, v, p = self.solve(force, return_components=True)
            solution = np.stack([u, v, p], axis=-1)

            # Validate
            if validate:
                validation = self.validate_solution(force, solution)
                if validation["valid"]:
                    return force, solution, validation
            else:
                return force, solution, {"valid": True}

        raise RuntimeError(
            f"Failed to generate valid sample after {max_attempts} attempts"
        )

    def validate_solution(
        self,
        force: np.ndarray,
        solution: np.ndarray,
        tol: float = 1e-8,
    ) -> Dict:
        """
        Validate the Stokes solution.

        Checks:
        1. Divergence-free condition: ∇·u ≈ 0
        2. Momentum residual: -μ∇²u + ∇p ≈ f
        """
        if solution.ndim == 3:
            u = solution[:, :, 0]
            v = solution[:, :, 1]
            p = solution[:, :, 2]
        else:
            u, v, p = solution

        if force.ndim == 3:
            fx = force[:, :, 0]
            fy = force[:, :, 1]
        else:
            fx, fy = force[:, :, 0], force[:, :, 1]

        # Check divergence
        u_hat = np.fft.fft2(u)
        v_hat = np.fft.fft2(v)
        div = np.fft.ifft2(1j * self.KX * u_hat + 1j * self.KY * v_hat).real
        div_norm = np.abs(div).max()

        # Check momentum residual
        p_hat = np.fft.fft2(p)

        lap_u = np.fft.ifft2(-self.K2 * u_hat).real
        lap_v = np.fft.ifft2(-self.K2 * v_hat).real
        dp_dx = np.fft.ifft2(1j * self.KX * p_hat).real
        dp_dy = np.fft.ifft2(1j * self.KY * p_hat).real

        R_x = -self.mu * lap_u + dp_dx - fx
        R_y = -self.mu * lap_v + dp_dy - fy

        momentum_x = float(np.linalg.norm(R_x)) / max(float(np.linalg.norm(fx)), 1e-10)
        momentum_y = float(np.linalg.norm(R_y)) / max(float(np.linalg.norm(fy)), 1e-10)

        is_valid = (
            not np.isnan(solution).any()
            and not np.isinf(solution).any()
            and div_norm < tol
            and momentum_x < tol
            and momentum_y < tol
        )

        return {
            "valid": is_valid,
            "divergence": div_norm,
            "momentum_x": momentum_x,
            "momentum_y": momentum_y,
        }
