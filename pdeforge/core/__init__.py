"""Core infrastructure for PDEForge."""

from typing import TYPE_CHECKING, Any, Optional

from pdeforge.core.base import PDEModel
from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import (
    describe_all_models,
    describe_model,
    get_model,
    get_model_info,
    list_models,
    register_model,
)
from pdeforge.core.types import Domain, GridSpec, PDEDataset

# FEniCSx base (optional)
FEniCSModel: Optional[Any]
try:
    from pdeforge.core.fenics_base import FEniCSModel as _FEniCSModel

    FEniCSModel = _FEniCSModel
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
