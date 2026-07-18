"""
2D Kolmogorov flow: incompressible Navier-Stokes with steady sinusoidal
band forcing f = (f0 sin(n y), 0). In vorticity form the forcing contributes
curl(f) = -f0 * n * cos(n y).

The classic forced-turbulence benchmark: energy injected at wavenumber n is
cascaded and dissipated, giving a statistically steady turbulent state at low
viscosity.

Operator learning task: w(x, y, 0) -> w(x, y, T).
"""

import numpy as np

from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model
from pdeforge.models.ns_vorticity_2d import NSVorticity2D


@register_model("kolmogorov_flow_2d")
class KolmogorovFlow2D(NSVorticity2D):
    """
    Kolmogorov flow on the 2 pi-periodic box: NS vorticity dynamics driven by
    a steady sinusoidal band force at wavenumber n.
    """

    USER_PARAMS = [
        ParamSpec(
            name="viscosity",
            description="Kinematic viscosity (1/Re); lower = more turbulent",
            default=1.0 / 40.0,
            param_type=ParamType.PHYSICAL,
            bounds=(1e-4, 1.0),
        ),
        ParamSpec(
            name="forcing_wavenumber",
            description="Band-forcing wavenumber n",
            default=4,
            param_type=ParamType.PHYSICAL,
            bounds=(1, 16),
        ),
        ParamSpec(
            name="forcing_amplitude",
            description="Forcing amplitude f0",
            default=1.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.0, 10.0),
        ),
        ParamSpec(
            name="time_horizon",
            description="Final time T",
            default=10.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.1, 200.0),
        ),
    ]

    DEFAULT_PARAMS = {
        "viscosity": 1.0 / 40.0,
        "forcing_wavenumber": 4,
        "forcing_amplitude": 1.0,
        "time_horizon": 10.0,
        "forcing": "kolmogorov",
        "_n_time_steps": 101,
        "_dt": None,
    }

    def __init__(self, resolution, domain=None, **params):
        # The canonical setting is the 2 pi-periodic box.
        if domain is None:
            domain = {
                "x": (0.0, 2.0 * np.pi),
                "y": (0.0, 2.0 * np.pi),
            }
        super().__init__(resolution, domain, **params)

        n = int(self.params.get("forcing_wavenumber", 4))
        f0 = self.params.get("forcing_amplitude", 1.0)
        Y = np.meshgrid(self.grids["x"], self.grids["y"])[1]
        # curl of (f0 sin(n y), 0) = -f0 n cos(n y)
        self._forcing_hat = np.fft.fft2(-f0 * n * np.cos(n * Y))
