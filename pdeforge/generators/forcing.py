"""
Forcing term generators for PDEForge.

This module provides methods for generating random forcing terms
for PDEs like Stokes flow, Navier-Stokes, etc.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np


class ForcingGenerator(ABC):
    """Abstract base class for forcing term generators."""

    @abstractmethod
    def generate(
        self,
        shape: Tuple[int, ...],
        seed: int = None,
        grid: Dict[str, np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate random forcing terms.

        Parameters
        ----------
        shape : Tuple[int, ...]
            Shape of the output array
        seed : int, optional
            Random seed
        grid : Dict[str, np.ndarray], optional
            Grid coordinates

        Returns
        -------
        np.ndarray
            Generated forcing field (may have multiple components)
        """
        pass


class FourierForcingGenerator(ForcingGenerator):
    """
    Generate forcing terms using Fourier series.

    For vector fields (like body force in Stokes), generates
    each component independently.

    Parameters
    ----------
    n_modes : int
        Number of Fourier modes
    decay : float
        Power law decay exponent for coefficients
    amplitude : float
        Overall amplitude scaling
    n_components : int
        Number of force components (e.g., 2 for 2D vector field)
    normalize : bool
        Whether to normalize the RMS of the force
    """

    def __init__(
        self,
        n_modes: int = 5,
        decay: float = 1.5,
        amplitude: float = 1.0,
        n_components: int = 2,
        normalize: bool = True,
    ):
        self.n_modes = n_modes
        self.decay = decay
        self.amplitude = amplitude
        self.n_components = n_components
        self.normalize = normalize

    def generate(
        self,
        shape: Tuple[int, ...],
        seed: int = None,
        grid: Dict[str, np.ndarray] = None,
    ) -> Tuple[np.ndarray, ...]:
        """
        Generate Fourier-based forcing.

        Returns a tuple of arrays, one for each component.
        """
        if seed is not None:
            np.random.seed(seed)

        if len(shape) != 2:
            raise ValueError("FourierForcingGenerator only supports 2D fields")

        return self._generate_2d(shape, grid)

    def _generate_2d(
        self,
        shape: Tuple[int, int],
        grid: Dict[str, np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate 2D force field with components (fx, fy)."""
        ny, nx = shape

        if grid is None:
            x = np.linspace(0, 1, nx, endpoint=False)
            y = np.linspace(0, 1, ny, endpoint=False)
            Lx, Ly = 1.0, 1.0
        else:
            x = grid.get("x", np.linspace(0, 1, nx, endpoint=False))
            y = grid.get("y", np.linspace(0, 1, ny, endpoint=False))
            Lx = x[-1] - x[0] + (x[1] - x[0])
            Ly = y[-1] - y[0] + (y[1] - y[0])

        X, Y = np.meshgrid(x, y)

        fx = np.zeros((ny, nx))
        fy = np.zeros((ny, nx))

        for kx in range(1, self.n_modes + 1):
            for ky in range(1, self.n_modes + 1):
                kx_val = 2 * np.pi * kx / Lx
                ky_val = 2 * np.pi * ky / Ly

                # Decay coefficient
                decay = 1.0 / (kx**self.decay * ky**self.decay)

                # fx component
                a_x = np.random.randn() * decay
                b_x = np.random.randn() * decay
                fx += a_x * np.sin(kx_val * X) * np.sin(ky_val * Y)
                fx += b_x * np.cos(kx_val * X) * np.cos(ky_val * Y)

                # fy component
                a_y = np.random.randn() * decay
                b_y = np.random.randn() * decay
                fy += a_y * np.sin(kx_val * X) * np.cos(ky_val * Y)
                fy += b_y * np.cos(kx_val * X) * np.sin(ky_val * Y)

        # Normalize
        if self.normalize:
            f_rms = np.sqrt(np.mean(fx**2 + fy**2))
            if f_rms > 1e-10:
                fx = fx / f_rms * self.amplitude
                fy = fy / f_rms * self.amplitude
        else:
            fx = fx * self.amplitude
            fy = fy * self.amplitude

        return fx, fy
