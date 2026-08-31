"""PDE models for PDEForge."""

# Import spectral models to trigger registration
# Stochastic: Heat
# Reaction-diffusion: Allen-Cahn
# Reaction-diffusion: FitzHugh-Nagumo
# Wave equation
# Heat equation
from pdeforge.models import (
    advection_1d,
    airfoil_euler_2d,
    allen_cahn_1d,
    allen_cahn_2d,
    allen_cahn_3d,
    burgers_1d,
    burgers_2d,
    cahn_hilliard,
    darcy_2d,
    darcy_fno_2d,
    darcy_fno_3d,
    eggshell_droplets_3d,
    fitzhugh_nagumo_1d,
    fitzhugh_nagumo_2d,
    gray_scott_2d,
    heat_1d,
    heat_2d,
    heat_3d,
    helmholtz_2d,
    heterogeneous_wave_2d,
    kdv_1d,
    kolmogorov_flow_2d,
    ks_1d,
    lotka_volterra_2d,
    ns_vorticity_2d,
    schrodinger_1d,
    shallow_water_2d,
    stochastic_allen_cahn_2d,
    stochastic_burgers_1d,
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
from pdeforge.models.eggshell_droplets_3d import EggshellDroplets3D
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
    # Cahn-Hilliard (two droplets in an egg-shell pellet, 3D)
    "EggshellDroplets3D",
    # Stochastic Heat
    "StochasticHeat1D",
    "StochasticHeat2D",
    # 2026 catalogue expansion (roadmap Phase 2)
    "NSVorticity2D",
    "KolmogorovFlow2D",
    "KuramotoSivashinsky1D",
    "GrayScott2D",
    "Advection1D",
    "DarcyFNO2D",
    "DarcyFNO3D",
    "AirfoilEuler2D",
    "KdV1D",
    "LotkaVolterra2D",
    "Burgers2D",
    "ShallowWater2D",
    "Schrodinger1D",
    "HeterogeneousWave2D",
    "Helmholtz2D",
    "Heat3D",
    "AllenCahn3D",
    "StochasticBurgers1D",
    "StochasticAllenCahn2D",
]

from pdeforge.models.advection_1d import Advection1D
from pdeforge.models.airfoil_euler_2d import AirfoilEuler2D
from pdeforge.models.allen_cahn_3d import AllenCahn3D
from pdeforge.models.burgers_2d import Burgers2D
from pdeforge.models.darcy_fno_2d import DarcyFNO2D
from pdeforge.models.darcy_fno_3d import DarcyFNO3D
from pdeforge.models.gray_scott_2d import GrayScott2D
from pdeforge.models.heat_3d import Heat3D
from pdeforge.models.helmholtz_2d import Helmholtz2D
from pdeforge.models.heterogeneous_wave_2d import HeterogeneousWave2D
from pdeforge.models.kdv_1d import KdV1D
from pdeforge.models.kolmogorov_flow_2d import KolmogorovFlow2D
from pdeforge.models.ks_1d import KuramotoSivashinsky1D
from pdeforge.models.lotka_volterra_2d import LotkaVolterra2D
from pdeforge.models.ns_vorticity_2d import NSVorticity2D
from pdeforge.models.schrodinger_1d import Schrodinger1D
from pdeforge.models.shallow_water_2d import ShallowWater2D
from pdeforge.models.stochastic_allen_cahn_2d import StochasticAllenCahn2D
from pdeforge.models.stochastic_burgers_1d import StochasticBurgers1D

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

    from pdeforge.models import naca_flow_2d
    from pdeforge.models.naca_flow_2d import NACAFlow2D

    __all__.append("NACAFlow2D")

    from pdeforge.models import elasticity_2d
    from pdeforge.models.elasticity_2d import Elasticity2D

    __all__.append("Elasticity2D")

    from pdeforge.models import porous_darcy_fem
    from pdeforge.models.porous_darcy_fem import PorousDarcyFEM

    __all__.append("PorousDarcyFEM")

    from pdeforge.models import rayleigh_benard_2d
    from pdeforge.models.rayleigh_benard_2d import RayleighBenard2D

    __all__.append("RayleighBenard2D")
except ImportError:
    # FEniCSx not available - skip
    pass
