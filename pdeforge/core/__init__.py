"""Core infrastructure for PDEForge."""

from pdeforge.core.base import PDEModel
from pdeforge.core.registry import (
    register_model, 
    get_model, 
    list_models,
    get_model_info,
    describe_model,
    describe_all_models,
)
from pdeforge.core.types import PDEDataset, Domain, GridSpec
from pdeforge.core.params import ParamSpec, ParamType

# FEniCSx base (optional)
try:
    from pdeforge.core.fenics_base import FEniCSModel
    HAS_FENICSX = True
except ImportError:
    FEniCSModel = None
    HAS_FENICSX = False

__all__ = [
    "PDEModel",
    "FEniCSModel",
    "register_model",
    "get_model", 
    "list_models",
    "get_model_info",
    "describe_model",
    "describe_all_models",
    "PDEDataset",
    "Domain",
    "GridSpec",
    "ParamSpec",
    "ParamType",
    "HAS_FENICSX",
]
