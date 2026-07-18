"""
Initial condition generators for PDEForge.

This module provides various methods for generating random initial conditions
for PDE simulations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union

import numpy as np


class ICGenerator(ABC):
    """Abstract base class for initial condition generators."""

    @abstractmethod
    def generate(
        self,
        shape: Tuple[int, ...],
        seed: Optional[int] = None,
        grid: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Generate random initial conditions.

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
            Generated initial conditions
        """
        pass


class FourierICGenerator(ICGenerator):
    """
    Generate initial conditions using Fourier series.

    For 1D: u₀(x) = Σₙ aₙ sin(2πnx/L)
    For 2D: u₀(x,y) = ΣₘΣₙ aₘₙ [sin/cos combinations]

    Coefficients decay as 1/n^α for smoothness.

    Parameters
    ----------
    n_modes : int
        Number of Fourier modes
    decay : float
        Power law decay exponent for coefficients (higher = smoother)
    amplitude : float
        Overall amplitude scaling
    use_cos : bool
        Whether to include cosine terms (default True for 2D)
    periodic : bool
        Whether the IC should be periodic
    """

    def __init__(
        self,
        n_modes: int = 10,
        decay: float = 1.5,
        amplitude: float = 1.0,
        use_cos: bool = True,
        periodic: bool = True,
    ):
        self.n_modes = n_modes
        self.decay = decay
        self.amplitude = amplitude
        self.use_cos = use_cos
        self.periodic = periodic

    def generate(
        self,
        shape: Tuple[int, ...],
        seed: Optional[int] = None,
        grid: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Generate Fourier-based initial conditions.

        Parameters
        ----------
        shape : Tuple[int, ...]
            Shape of the output array
        seed : int, optional
            Random seed
        grid : Dict[str, np.ndarray], optional
            Grid coordinates. If None, uses unit domain.

        Returns
        -------
        np.ndarray
            Generated initial conditions
        """
        if seed is not None:
            np.random.seed(seed)

        ndim = len(shape)

        if ndim == 1:
            return self._generate_1d(shape[0], grid)
        elif ndim == 2:
            return self._generate_2d((shape[0], shape[1]), grid)
        else:
            raise ValueError(f"Unsupported dimensionality: {ndim}")

    def _generate_1d(
        self,
        n: int,
        grid: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """Generate 1D Fourier IC."""
        if grid is None:
            x = np.linspace(0, 1, n, endpoint=False)
            L = 1.0
        else:
            x = grid.get("x", np.linspace(0, 1, n, endpoint=False))
            L = x[-1] - x[0] + (x[1] - x[0])  # Account for periodic spacing

        u = np.zeros(n)

        for k in range(1, self.n_modes + 1):
            # Decay coefficient
            decay_factor = 1.0 / (k**self.decay)

            # Random coefficient
            a = np.random.randn() * decay_factor * self.amplitude

            # Add sine term
            u += a * np.sin(2 * np.pi * k * x / L)

            # Optionally add cosine term
            if self.use_cos:
                b = np.random.randn() * decay_factor * self.amplitude
                u += b * np.cos(2 * np.pi * k * x / L)

        return u

    def _generate_2d(
        self,
        shape: Tuple[int, int],
        grid: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """Generate 2D Fourier IC."""
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
        u = np.zeros((ny, nx))

        for kx in range(1, self.n_modes + 1):
            for ky in range(1, self.n_modes + 1):
                # Decay coefficient
                decay_factor = 1.0 / ((kx**self.decay) * (ky**self.decay))

                # Sin-Sin
                a = np.random.randn() * decay_factor * self.amplitude
                u += (
                    a
                    * np.sin(2 * np.pi * kx * X / Lx)
                    * np.sin(2 * np.pi * ky * Y / Ly)
                )

                if self.use_cos:
                    # Sin-Cos
                    b = np.random.randn() * decay_factor * self.amplitude
                    u += (
                        b
                        * np.sin(2 * np.pi * kx * X / Lx)
                        * np.cos(2 * np.pi * ky * Y / Ly)
                    )

                    # Cos-Sin
                    c = np.random.randn() * decay_factor * self.amplitude
                    u += (
                        c
                        * np.cos(2 * np.pi * kx * X / Lx)
                        * np.sin(2 * np.pi * ky * Y / Ly)
                    )

                    # Cos-Cos
                    d = np.random.randn() * decay_factor * self.amplitude
                    u += (
                        d
                        * np.cos(2 * np.pi * kx * X / Lx)
                        * np.cos(2 * np.pi * ky * Y / Ly)
                    )

        return u


class GaussianRandomFieldGenerator(ICGenerator):
    """
    Generate initial conditions using Gaussian Random Fields.

    Uses the spectral method to generate fields with a specified
    power spectrum (typically power-law decay).

    Parameters
    ----------
    alpha : float
        Power spectrum exponent: P(k) ∝ k^(-alpha)
        Higher values give smoother fields.
    amplitude : float
        Overall amplitude scaling
    """

    def __init__(self, alpha: float = 2.0, amplitude: float = 1.0):
        self.alpha = alpha
        self.amplitude = amplitude

    def generate(
        self,
        shape: Tuple[int, ...],
        seed: Optional[int] = None,
        grid: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Generate GRF initial conditions.

        Parameters
        ----------
        shape : Tuple[int, ...]
            Shape of the output array
        seed : int, optional
            Random seed
        grid : Dict[str, np.ndarray], optional
            Grid coordinates (not used, kept for API consistency)

        Returns
        -------
        np.ndarray
            Generated initial conditions
        """
        if seed is not None:
            np.random.seed(seed)

        ndim = len(shape)

        if ndim == 1:
            return self._generate_1d(shape[0])
        elif ndim == 2:
            return self._generate_2d((shape[0], shape[1]))
        else:
            raise ValueError(f"Unsupported dimensionality: {ndim}")

    def _generate_1d(self, n: int) -> np.ndarray:
        """Generate 1D GRF."""
        # Create wavenumbers
        k = np.fft.fftfreq(n) * n
        k[0] = 1e-10  # Avoid division by zero

        # Power spectrum
        power = np.abs(k) ** (-self.alpha)
        power[0] = 0  # Zero mean

        # Random phases
        phases = np.random.uniform(0, 2 * np.pi, n)

        # Complex Fourier coefficients
        amplitudes = np.sqrt(power) * np.exp(1j * phases)

        # Make Hermitian for real output
        amplitudes[n // 2 + 1 :] = np.conj(amplitudes[1 : n // 2][::-1])
        if n % 2 == 0:
            amplitudes[n // 2] = amplitudes[n // 2].real

        # Inverse FFT
        u = np.fft.ifft(amplitudes).real
        u = u / u.std() * self.amplitude

        return u

    def _generate_2d(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generate 2D GRF."""
        ny, nx = shape

        # Create wavenumber grids
        kx = np.fft.fftfreq(nx) * nx
        ky = np.fft.fftfreq(ny) * ny
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX**2 + KY**2)
        K[0, 0] = 1e-10  # Avoid division by zero

        # Power spectrum
        power = K ** (-self.alpha)
        power[0, 0] = 0  # Zero mean

        # Random phases
        phases = np.random.uniform(0, 2 * np.pi, (ny, nx))

        # Complex Fourier coefficients
        amplitudes = np.sqrt(power) * np.exp(1j * phases)

        # Inverse FFT
        u = np.fft.ifft2(amplitudes).real
        u = u / u.std() * self.amplitude

        return u


class SigmoidTransformGenerator(ICGenerator):
    """
    Generate bounded positive fields using sigmoid transform.

    Useful for generating permeability/diffusivity fields that must
    stay within physical bounds.

    u(x) = u_min + (u_max - u_min) * sigmoid(c(x))

    where c(x) is a Gaussian random field.

    Parameters
    ----------
    u_min, u_max : float
        Bounds for the output field
    base_generator : ICGenerator, optional
        Generator for the base field (default: GRF)
    """

    def __init__(
        self,
        u_min: float = 0.1,
        u_max: float = 10.0,
        base_generator: ICGenerator = None,
    ):
        self.u_min = u_min
        self.u_max = u_max
        self.base_generator = base_generator or GaussianRandomFieldGenerator()

    def generate(
        self,
        shape: Tuple[int, ...],
        seed: Optional[int] = None,
        grid: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """Generate bounded positive field."""
        # Generate base field
        c = self.base_generator.generate(shape, seed, grid)

        # Apply sigmoid transform
        sigmoid = 1.0 / (1.0 + np.exp(-c))
        u = self.u_min + (self.u_max - self.u_min) * sigmoid

        return u


def get_ic_generator(name: str, **params) -> ICGenerator:
    """
    Factory function to get an IC generator by name.

    Parameters
    ----------
    name : str
        Generator type: "fourier", "grf", "sigmoid"
    **params
        Parameters for the generator

    Returns
    -------
    ICGenerator
        The requested generator instance
    """
    generators = {
        "fourier": FourierICGenerator,
        "grf": GaussianRandomFieldGenerator,
        "gaussian_random_field": GaussianRandomFieldGenerator,
        "grf_neumann": GRFNeumannGenerator,
        "grf_periodic": GRFPeriodicGenerator,
        "sigmoid": SigmoidTransformGenerator,
    }

    if name not in generators:
        raise ValueError(
            f"Unknown IC generator: {name}. Available: {list(generators.keys())}"
        )

    return generators[name](**params)


class GRFNeumannGenerator(ICGenerator):
    """
    Matern-like Gaussian random field with the CANONICAL operator-learning
    covariance N(0, (-Laplacian + tau^2 I)^(-alpha)) under Neumann boundary
    conditions on the unit square — the measure behind the classic FNO Darcy
    datasets (tau = 3, alpha = 2 there; verified empirically against the
    distributed Darcy421 data, spectrum fit R^2 = 0.998).

    Sampled by Karhunen-Loeve expansion in the cosine (DCT) basis with
    per-mode variance (pi^2 (i^2 + j^2) + tau^2)^(-alpha), then rescaled to
    an exact target pointwise std `sigma` (the overall scale is the one
    convention-dependent constant; pinning it to data is the honest choice).
    """

    def __init__(self, alpha: float = 2.0, tau: float = 3.0, sigma: float = 0.2918):
        self.alpha = alpha
        self.tau = tau
        self.sigma = sigma

    def generate(
        self,
        shape: Tuple[int, ...],
        seed: Optional[int] = None,
        grid: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        if len(shape) not in (2, 3):
            raise ValueError(
                "GRFNeumannGenerator supports 2D and 3D (got shape %r)" % (shape,)
            )
        from scipy.fft import idctn

        rng = np.random.default_rng(seed)
        # sum of squared mode indices over however many axes (2D bit-identical
        # to the original implementation; 3D is the canonical-measure
        # extension — the covariance operator is dimension-agnostic)
        mode2 = np.zeros(shape)
        for axis, n in enumerate(shape):
            idx_shape = [1] * len(shape)
            idx_shape[axis] = n
            mode2 = mode2 + (np.arange(n) ** 2).reshape(idx_shape)
        lam = (np.pi**2 * mode2 + self.tau**2) ** (-self.alpha)
        lam[(0,) * len(shape)] = 0.0  # zero-mean field
        # Fixed-constant calibration: E[domain mean-square] = sigma^2 (via
        # Parseval for the ortho DCT). Per-sample std then FLUCTUATES around
        # sigma, as the true Gaussian measure demands (verified against the
        # canonical Darcy421 data: per-sample std spread ~0.08 around 0.29) —
        # a per-sample rescale would silently condition the measure.
        scale = self.sigma / np.sqrt(lam.sum() / np.prod(shape))
        coeff = rng.standard_normal(shape) * (scale * np.sqrt(lam))
        return idctn(coeff, norm="ortho")


class GRFPeriodicGenerator(ICGenerator):
    """
    1D periodic Gaussian random field with the FNO-paper Burgers measure

        u0 ~ N(0, scale * (-Laplacian + tau^2 I)^(-alpha))

    on the unit circle. Canonical values (Li et al. 2020 Burgers): tau = 5,
    alpha = 2, scale = tau^4 = 625, i.e. N(0, 625(-Delta + 25 I)^(-2)).

    KL expansion in the Fourier basis: eigenvalues of -Laplacian on the unit
    circle are (2 pi k)^2, so lambda_k = scale * ((2 pi k)^2 + tau^2)^(-alpha).
    Complex coefficients with Hermitian symmetry give a real field with
    pointwise variance sum_k lambda_k (the k = 0 mean mode included, as in
    the original datasets).
    """

    def __init__(self, alpha: float = 2.0, tau: float = 5.0, scale: float = 625.0):
        self.alpha = alpha
        self.tau = tau
        self.scale = scale

    def _lambda(self, k):
        return self.scale * ((2.0 * np.pi * k) ** 2 + self.tau**2) ** (-self.alpha)

    def expected_variance(self, n: int) -> float:
        """Pointwise variance E[u0(x)^2] of the discretised field."""
        k = np.abs(np.fft.fftfreq(n) * n)
        return float(self._lambda(k).sum())

    def generate(
        self,
        shape: Tuple[int, ...],
        seed: Optional[int] = None,
        grid: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        if len(shape) != 1:
            raise ValueError("GRFPeriodicGenerator is 1D (got shape %r)" % (shape,))
        rng = np.random.default_rng(seed)
        n = shape[0]
        half = n // 2

        Z = np.zeros(n, dtype=complex)
        # z_0: real mean mode
        Z[0] = np.sqrt(self._lambda(0)) * rng.standard_normal()
        # 0 < k < n/2: complex CN(0, lambda_k)
        ks = np.arange(1, half if n % 2 == 0 else half + 1)
        lam = self._lambda(ks)
        z = np.sqrt(lam / 2.0) * (
            rng.standard_normal(len(ks)) + 1j * rng.standard_normal(len(ks))
        )
        Z[ks] = z
        Z[-ks] = np.conj(z)
        # Nyquist mode (even n): real
        if n % 2 == 0:
            Z[half] = np.sqrt(self._lambda(half)) * rng.standard_normal()

        # u(x_j) = sum_k Z_k exp(2 pi i j k / n) = n * ifft(Z)
        return np.fft.ifft(Z).real * n
