"""
2D incompressible Navier-Stokes, vorticity-streamfunction formulation.

    dw/dt + u . grad(w) = nu Laplacian(w) + f,   u = (psi_y, -psi_x),
    Laplacian(psi) = -w

on the periodic box. This is THE canonical operator-learning benchmark
(Li et al. 2020): the map w(x, 0) -> w(x, T).

Solved on the spectral seam with ETDRK4: viscous diffusion exact, the
dealiased advection term explicit. An optional steady forcing term covers the
FNO-paper setup (f = amp*(sin(2 pi (x+y)) + cos(2 pi (x+y)))).
"""

from typing import Callable, Dict, Tuple, Union

import numpy as np

from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model
from pdeforge.generators.initial_conditions import get_ic_generator
from pdeforge.solvers.semilinear import SemiLinearSpectralModel


@register_model("ns_vorticity_2d")
class NSVorticity2D(SemiLinearSpectralModel):
    """
    2D incompressible Navier-Stokes (vorticity form) on a periodic box.

    Operator task: w(x, y, 0) -> w(x, y, T). Lower viscosity = richer,
    finer-scale dynamics. With forcing="fno" the setup matches the classic
    FNO Navier-Stokes benchmark family.
    """

    NDIM = 2
    INPUT_NAMES = ["w0"]
    OUTPUT_NAMES = ["w_T"]

    USER_PARAMS = [
        ParamSpec(
            name="viscosity",
            description="Kinematic viscosity nu (lower = more turbulent)",
            default=1e-3,
            param_type=ParamType.PHYSICAL,
            bounds=(1e-5, 1.0),
        ),
        ParamSpec(
            name="time_horizon",
            description="Final time T",
            default=5.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.1, 100.0),
        ),
        ParamSpec(
            name="forcing",
            description="Forcing type",
            default="none",
            param_type=ParamType.PHYSICAL,
            choices=["none", "fno"],
            affects="'fno' adds the Li et al. 2020 steady forcing",
        ),
        ParamSpec(
            name="forcing_amplitude",
            description="Forcing amplitude (used when forcing != 'none')",
            default=0.1,
            param_type=ParamType.PHYSICAL,
            bounds=(0.0, 10.0),
        ),
    ]

    DEFAULT_PARAMS = {
        "viscosity": 1e-3,
        "time_horizon": 5.0,
        "forcing": "none",
        "forcing_amplitude": 0.1,
        "_n_time_steps": 101,
        "_dt": None,
    }

    def __init__(self, resolution, domain=None, **params):
        super().__init__(resolution, domain, **params)

        self.nu = self.params["viscosity"]
        self.T = self.params.get("time_horizon", 5.0)
        self.n_t = self.params.get("_n_time_steps", 101)

        self._setup_spectral()
        # dim_order is (y, x): K[0] = KY, K[1] = KX
        self.KY, self.KX = self.K[0], self.K[1]
        # Inverse Laplacian with the zero mode pinned to zero.
        K2 = self.K2.copy()
        K2[K2 == 0.0] = 1.0
        self._inv_K2 = 1.0 / K2
        self._inv_K2.flat[0] = 0.0

        # Advective CFL for the explicit term; diffusion is exact in L.
        dx = self.grids["x"][1] - self.grids["x"][0]
        dt = self.params.get("_dt")
        if dt is None:
            dt = min(0.5 * dx, self.T / 500.0)
        self.dt = dt

        # Steady forcing (in vorticity form: curl of the velocity forcing).
        self._forcing_hat = None
        if self.params.get("forcing") == "fno":
            amp = self.params.get("forcing_amplitude", 0.1)
            X, Y = np.meshgrid(self.grids["x"], self.grids["y"])
            # Li et al. 2020: velocity forcing amp*(sin+cos)(2 pi (x+y));
            # its curl is amp*2*pi*(cos - (-sin)) ... applied directly as a
            # vorticity source with the same functional form.
            f = amp * (np.sin(2 * np.pi * (X + Y)) + np.cos(2 * np.pi * (X + Y)))
            self._forcing_hat = np.fft.fft2(f)

    def linear_symbol(self):
        return -self.nu * self.K2

    def velocity_from_vorticity_hat(self, w_hat, ops):
        """u = (psi_y, -psi_x) with Laplacian(psi) = -w, spectrally."""
        psi_hat = w_hat * self._inv_K2  # psi_hat = w_hat / k^2
        u_hat = 1j * self.KY * psi_hat
        v_hat = -1j * self.KX * psi_hat
        return u_hat, v_hat

    def nonlinear_hat(self, v, u, ops):
        w_hat = v
        u_hat, v_hat = self.velocity_from_vorticity_hat(w_hat, ops)
        ux = ops.real(ops.ifftn(u_hat, axes=self.spatial_axes))
        uy = ops.real(ops.ifftn(v_hat, axes=self.spatial_axes))
        wx = ops.real(ops.ifftn(1j * self.KX * w_hat, axes=self.spatial_axes))
        wy = ops.real(ops.ifftn(1j * self.KY * w_hat, axes=self.spatial_axes))
        adv_hat = self._fft(ux * wx + uy * wy, ops)
        N = -(self.dealias * adv_hat)
        if self._forcing_hat is not None:
            N = N + ops.asarray(self._forcing_hat)
        return N

    def generate_ic(self, generator="grf", generator_params=None, seed=None):
        """Random initial vorticity (GRF by default, FNO-style smoothness)."""
        if generator_params is None:
            generator_params = {}
        if generator == "grf":
            generator_params = {"alpha": 2.5, "amplitude": 1.0, **generator_params}
        if isinstance(generator, str):
            gen = get_ic_generator(generator, **generator_params)
        else:
            gen = generator
        return gen.generate(shape=self.field_shape, seed=seed, grid=self.grids)

    def validate_solution(self, ic, solution, tol=1e-6):
        is_valid = (
            not np.isnan(solution).any()
            and not np.isinf(solution).any()
            and np.abs(solution).max() < 1e6
        )
        return {"valid": is_valid, "max_value": float(np.abs(solution).max())}
