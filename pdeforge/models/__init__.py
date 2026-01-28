"""PDE models for PDEForge."""

# Import spectral models to trigger registration
from pdeforge.models import burgers_1d
from pdeforge.models import darcy_2d
from pdeforge.models import stokes_2d

# Heat equation
from pdeforge.models import heat_1d
from pdeforge.models import heat_2d

# Wave equation
from pdeforge.models import wave_1d
from pdeforge.models import wave_2d

# Reaction-diffusion: FitzHugh-Nagumo
from pdeforge.models import fitzhugh_nagumo_1d
from pdeforge.models import fitzhugh_nagumo_2d

# Reaction-diffusion: Allen-Cahn
from pdeforge.models import allen_cahn_1d
from pdeforge.models import allen_cahn_2d

# Stochastic: Heat
from pdeforge.models import stochastic_heat_1d
from pdeforge.models import stochastic_heat_2d

# Make model classes available
from pdeforge.models.burgers_1d import Burgers1D
from pdeforge.models.darcy_2d import Darcy2D
from pdeforge.models.stokes_2d import Stokes2D
from pdeforge.models.heat_1d import Heat1D
from pdeforge.models.heat_2d import Heat2D
from pdeforge.models.wave_1d import Wave1D
from pdeforge.models.wave_2d import Wave2D
from pdeforge.models.fitzhugh_nagumo_1d import FitzHughNagumo1D
from pdeforge.models.fitzhugh_nagumo_2d import FitzHughNagumo2D
from pdeforge.models.allen_cahn_1d import AllenCahn1D
from pdeforge.models.allen_cahn_2d import AllenCahn2D
from pdeforge.models.stochastic_heat_1d import StochasticHeat1D
from pdeforge.models.stochastic_heat_2d import StochasticHeat2D

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
    from pdeforge.models.cylinder_flow_2d_parameterized import CylinderFlow2DParameterized
    __all__.append("CylinderFlow2DParameterized")

    from pdeforge.models import cylinder_flow_2d_turbulent
    from pdeforge.models.cylinder_flow_2d_turbulent import CylinderFlow2DTurbulent
    __all__.append("CylinderFlow2DTurbulent")
except ImportError:
    # FEniCSx not available - skip
    pass
