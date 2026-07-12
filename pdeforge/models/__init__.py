"""PDE models for PDEForge."""

# Import spectral models to trigger registration
# Stochastic: Heat
# Reaction-diffusion: Allen-Cahn
# Reaction-diffusion: FitzHugh-Nagumo
# Wave equation
# Heat equation
from pdeforge.models import (
    allen_cahn_1d,
    allen_cahn_2d,
    burgers_1d,
    cahn_hilliard,
    darcy_2d,
    fitzhugh_nagumo_1d,
    fitzhugh_nagumo_2d,
    heat_1d,
    heat_2d,
    stochastic_heat_1d,
    stochastic_heat_2d,
    stokes_2d,
    wave_1d,
    wave_2d,
)
from pdeforge.models.allen_cahn_1d import AllenCahn1D
from pdeforge.models.allen_cahn_2d import AllenCahn2D

# Make model classes available
from pdeforge.models.burgers_1d import Burgers1D
from pdeforge.models.cahn_hilliard import CahnHilliard
from pdeforge.models.darcy_2d import Darcy2D
from pdeforge.models.fitzhugh_nagumo_1d import FitzHughNagumo1D
from pdeforge.models.fitzhugh_nagumo_2d import FitzHughNagumo2D
from pdeforge.models.heat_1d import Heat1D
from pdeforge.models.heat_2d import Heat2D
from pdeforge.models.stochastic_heat_1d import StochasticHeat1D
from pdeforge.models.stochastic_heat_2d import StochasticHeat2D
from pdeforge.models.stokes_2d import Stokes2D
from pdeforge.models.wave_1d import Wave1D
from pdeforge.models.wave_2d import Wave2D

__all__ = [
    # Original models
    "Burgers1D",
    "Darcy2D",
    "Stokes2D",
    # Heat
    "Heat1D",
    "Heat2D",
    # Wave
    "Wave1D",
    "Wave2D",
    # FitzHugh-Nagumo
    "FitzHughNagumo1D",
    "FitzHughNagumo2D",
    # Allen-Cahn
    "AllenCahn1D",
    "AllenCahn2D",
    # Cahn-Hilliard (spinodal decomposition, 2D/3D)
    "CahnHilliard",
    # Stochastic Heat
    "StochasticHeat1D",
    "StochasticHeat2D",
]

# Import FEniCSx models (optional - only if FEniCSx is available)
try:
    from pdeforge.models import cylinder_flow_2d
    from pdeforge.models.cylinder_flow_2d import CylinderFlow2D

    __all__.append("CylinderFlow2D")

    from pdeforge.models import cylinder_flow_2d_unsteady
    from pdeforge.models.cylinder_flow_2d_unsteady import CylinderFlow2DUnsteady

    __all__.append("CylinderFlow2DUnsteady")

    from pdeforge.models import cylinder_flow_2d_parameterized
    from pdeforge.models.cylinder_flow_2d_parameterized import (
        CylinderFlow2DParameterized,
    )

    __all__.append("CylinderFlow2DParameterized")

    from pdeforge.models import cylinder_flow_2d_turbulent
    from pdeforge.models.cylinder_flow_2d_turbulent import CylinderFlow2DTurbulent

    __all__.append("CylinderFlow2DTurbulent")
except ImportError:
    # FEniCSx not available - skip
    pass
