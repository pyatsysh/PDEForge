"""Numerical solvers for PDEForge."""

from pdeforge.solvers.spectral import (
    SpectralSolver1D,
    SpectralSolver2D,
    compute_fft_derivative_1d,
    compute_fft_derivative_2d,
    compute_fft_laplacian_2d,
    get_wavenumbers_1d,
    get_wavenumbers_2d,
)

__all__ = [
    "SpectralSolver1D",
    "SpectralSolver2D",
    "compute_fft_derivative_1d",
    "compute_fft_derivative_2d",
    "compute_fft_laplacian_2d",
    "get_wavenumbers_1d",
    "get_wavenumbers_2d",
]

# FEniCSx utilities (optional)
try:
    from pdeforge.solvers.fenics_utils import (
        create_rectangle_with_hole,
        create_simple_rectangle,
        mark_boundaries_rectangle,
    )
    __all__.extend([
        "create_rectangle_with_hole",
        "create_simple_rectangle",
        "mark_boundaries_rectangle",
    ])
except ImportError:
    # FEniCSx not available
    pass
