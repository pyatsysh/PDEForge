"""
Spectral (FFT-based) solvers for PDEForge.

This module provides utilities for solving PDEs using spectral methods
with Fast Fourier Transform (FFT) for spatial derivatives.
"""

import numpy as np
from scipy.integrate import odeint, solve_ivp
from typing import Callable, Tuple, Dict, Optional


def get_wavenumbers_1d(n: int, dx: float) -> np.ndarray:
    """
    Compute wavenumbers for 1D FFT.
    
    Parameters
    ----------
    n : int
        Number of grid points
    dx : float
        Grid spacing
        
    Returns
    -------
    np.ndarray
        Wavenumber array
    """
    return 2 * np.pi * np.fft.fftfreq(n, d=dx)


def get_wavenumbers_2d(
    ny: int, nx: int, dy: float, dx: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute wavenumbers for 2D FFT.
    
    Parameters
    ----------
    ny, nx : int
        Number of grid points in y and x directions
    dy, dx : float
        Grid spacing in y and x directions
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (KY, KX) meshgrids of wavenumbers
    """
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    return np.meshgrid(ky, kx, indexing='ij')


def compute_fft_derivative_1d(u: np.ndarray, k: np.ndarray, order: int = 1) -> np.ndarray:
    """
    Compute derivative using FFT.
    
    Parameters
    ----------
    u : np.ndarray
        Field to differentiate
    k : np.ndarray
        Wavenumber array
    order : int
        Order of derivative (1, 2, etc.)
        
    Returns
    -------
    np.ndarray
        Derivative of u
    """
    u_hat = np.fft.fft(u)
    if order % 2 == 0:
        # Even order derivative (real result)
        du_hat = (1j * k) ** order * u_hat
    else:
        # Odd order derivative
        du_hat = (1j * k) ** order * u_hat
    return np.fft.ifft(du_hat).real


def compute_fft_derivative_2d(
    u: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    dim: str,
    order: int = 1,
) -> np.ndarray:
    """
    Compute partial derivative in 2D using FFT.
    
    Parameters
    ----------
    u : np.ndarray
        2D field to differentiate
    kx, ky : np.ndarray
        Wavenumber meshgrids
    dim : str
        Dimension to differentiate ('x' or 'y')
    order : int
        Order of derivative
        
    Returns
    -------
    np.ndarray
        Partial derivative of u
    """
    u_hat = np.fft.fft2(u)
    k = kx if dim == 'x' else ky
    du_hat = (1j * k) ** order * u_hat
    return np.fft.ifft2(du_hat).real


def compute_fft_laplacian_2d(
    u: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
) -> np.ndarray:
    """
    Compute Laplacian in 2D using FFT.
    
    Parameters
    ----------
    u : np.ndarray
        2D field
    kx, ky : np.ndarray
        Wavenumber meshgrids
        
    Returns
    -------
    np.ndarray
        Laplacian of u
    """
    u_hat = np.fft.fft2(u)
    k2 = kx**2 + ky**2
    lap_u_hat = -k2 * u_hat
    return np.fft.ifft2(lap_u_hat).real


class SpectralSolver1D:
    """
    1D spectral solver for time-dependent PDEs.
    
    Uses FFT for spatial derivatives and scipy's ODE integrators
    for time stepping.
    
    Parameters
    ----------
    n : int
        Number of grid points
    L : float
        Domain length
    """
    
    def __init__(self, n: int, L: float = 1.0):
        self.n = n
        self.L = L
        self.dx = L / n
        self.x = np.linspace(0, L, n, endpoint=False)
        self.k = get_wavenumbers_1d(n, self.dx)
    
    def solve(
        self,
        rhs_func: Callable,
        u0: np.ndarray,
        t_span: Tuple[float, float],
        n_t: int = 101,
        method: str = 'RK45',
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the PDE.
        
        Parameters
        ----------
        rhs_func : Callable
            Right-hand side function: rhs(t, u, k) -> du/dt
        u0 : np.ndarray
            Initial condition
        t_span : Tuple[float, float]
            Time interval (t0, tf)
        n_t : int
            Number of time points to output
        method : str
            ODE solver method
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (t, u) where t is time array and u is solution array of shape (n_t, n)
        """
        t_eval = np.linspace(t_span[0], t_span[1], n_t)
        
        def rhs_wrapper(t, u):
            return rhs_func(t, u, self.k)
        
        # Try solve_ivp first, fall back to odeint if needed
        try:
            sol = solve_ivp(
                rhs_wrapper,
                t_span,
                u0,
                method=method,
                t_eval=t_eval,
                vectorized=False,
            )
            return sol.t, sol.y.T
        except Exception:
            # Fall back to odeint
            def rhs_odeint(u, t):
                return rhs_func(t, u, self.k)
            
            u = odeint(rhs_odeint, u0, t_eval, mxstep=5000)
            return t_eval, u


class SpectralSolver2D:
    """
    2D spectral solver for steady-state or time-dependent PDEs.
    
    Uses FFT for spatial derivatives.
    
    Parameters
    ----------
    ny, nx : int
        Number of grid points in y and x directions
    Ly, Lx : float
        Domain size in y and x directions
    """
    
    def __init__(
        self,
        ny: int,
        nx: int,
        Ly: float = 1.0,
        Lx: float = 1.0,
    ):
        self.ny = ny
        self.nx = nx
        self.Ly = Ly
        self.Lx = Lx
        self.dy = Ly / ny
        self.dx = Lx / nx
        
        # Create grids
        self.x = np.linspace(0, Lx, nx, endpoint=False)
        self.y = np.linspace(0, Ly, ny, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Wavenumbers
        self.kx = 2 * np.pi * np.fft.fftfreq(nx, d=self.dx)
        self.ky = 2 * np.pi * np.fft.fftfreq(ny, d=self.dy)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky)
        self.K2 = self.KX**2 + self.KY**2
    
    def gradient(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradient of u."""
        u_hat = np.fft.fft2(u)
        du_dx = np.fft.ifft2(1j * self.KX * u_hat).real
        du_dy = np.fft.ifft2(1j * self.KY * u_hat).real
        return du_dx, du_dy
    
    def laplacian(self, u: np.ndarray) -> np.ndarray:
        """Compute Laplacian of u."""
        u_hat = np.fft.fft2(u)
        lap_u = np.fft.ifft2(-self.K2 * u_hat).real
        return lap_u
    
    def divergence(self, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
        """Compute divergence of vector field (ux, uy)."""
        ux_hat = np.fft.fft2(ux)
        uy_hat = np.fft.fft2(uy)
        div = np.fft.ifft2(1j * self.KX * ux_hat + 1j * self.KY * uy_hat).real
        return div
    
    def solve_poisson(
        self,
        f: np.ndarray,
        regularization: float = 1e-14,
    ) -> np.ndarray:
        """
        Solve Poisson equation: -∇²u = f.
        
        Parameters
        ----------
        f : np.ndarray
            Right-hand side
        regularization : float
            Small value to avoid division by zero at k=0
            
        Returns
        -------
        np.ndarray
            Solution u
        """
        f_hat = np.fft.fft2(f)
        
        # Avoid division by zero at k=0
        K2_reg = self.K2.copy()
        K2_reg[0, 0] = regularization
        
        u_hat = f_hat / K2_reg
        u_hat[0, 0] = 0  # Set mean to zero
        
        return np.fft.ifft2(u_hat).real
    
    def solve_stokes(
        self,
        fx: np.ndarray,
        fy: np.ndarray,
        mu: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve 2D Stokes equations using Leray projection.
        
        -μ∇²u + ∇p = f
        ∇·u = 0
        
        Parameters
        ----------
        fx, fy : np.ndarray
            Body force components
        mu : float
            Viscosity
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (u, v, p) - velocity components and pressure
        """
        fx_hat = np.fft.fft2(fx)
        fy_hat = np.fft.fft2(fy)
        
        # Initialize
        u_hat = np.zeros_like(fx_hat, dtype=complex)
        v_hat = np.zeros_like(fy_hat, dtype=complex)
        p_hat = np.zeros_like(fx_hat, dtype=complex)
        
        # Mask for non-zero wavenumbers
        mask = self.K2 > 1e-14
        
        # k · f
        k_dot_f = self.KX * fx_hat + self.KY * fy_hat
        
        # Leray projection and solve
        u_hat[mask] = (fx_hat[mask] - self.KX[mask] * k_dot_f[mask] / self.K2[mask]) / (mu * self.K2[mask])
        v_hat[mask] = (fy_hat[mask] - self.KY[mask] * k_dot_f[mask] / self.K2[mask]) / (mu * self.K2[mask])
        
        # Pressure
        p_hat[mask] = -1j * k_dot_f[mask] / self.K2[mask]
        
        u = np.fft.ifft2(u_hat).real
        v = np.fft.ifft2(v_hat).real
        p = np.fft.ifft2(p_hat).real
        p = p - p.mean()  # Zero mean pressure
        
        return u, v, p
