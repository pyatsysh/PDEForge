"""
Parameter specification for PDEForge models.

This module provides utilities for declaring and documenting
user-facing parameters for PDE models.

Philosophy:
    PDEForge is for generating ML training datasets, not a general PDE solver.
    Model contributors should expose only parameters that affect data characteristics:

    EXPOSE:
        - Physical parameters (viscosity, Reynolds number, etc.)
        - Domain/geometry parameters that define the problem
        - Initial condition characteristics

    HIDE:
        - Solver tolerances and convergence criteria
        - Mesh internals and discretization details
        - Time stepping schemes and internal parameters
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class ParamType(Enum):
    """Types of parameters for documentation."""

    PHYSICAL = "physical"  # Physical constants (viscosity, density, etc.)
    GEOMETRY = "geometry"  # Domain/geometry parameters
    INPUT = "input"  # Input field characteristics
    OUTPUT = "output"  # Output characteristics


@dataclass
class ParamSpec:
    """
    Specification for a user-facing parameter.

    Attributes
    ----------
    name : str
        Parameter name (as passed to generate_dataset)
    description : str
        Human-readable description
    default : Any
        Default value
    param_type : ParamType
        Category of parameter
    bounds : Tuple[float, float], optional
        Valid range for numeric parameters
    choices : List[Any], optional
        Valid choices for discrete parameters
    units : str, optional
        Physical units (if applicable)
    affects : str, optional
        What aspect of the data this parameter affects
    """

    name: str
    description: str
    default: Any
    param_type: ParamType = ParamType.PHYSICAL
    bounds: Optional[Tuple[float, float]] = None
    choices: Optional[List[Any]] = None
    units: Optional[str] = None
    affects: Optional[str] = None

    def validate(self, value: Any) -> bool:
        """Check if a value is valid for this parameter."""
        if self.bounds is not None:
            if not (self.bounds[0] <= value <= self.bounds[1]):
                return False
        if self.choices is not None:
            if value not in self.choices:
                return False
        return True

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "default": self.default,
            "type": self.param_type.value,
            "bounds": self.bounds,
            "choices": self.choices,
            "units": self.units,
            "affects": self.affects,
        }


def describe_params(param_specs: List[ParamSpec]) -> str:
    """
    Generate a formatted description of parameters.

    Parameters
    ----------
    param_specs : List[ParamSpec]
        List of parameter specifications

    Returns
    -------
    str
        Formatted parameter documentation
    """
    lines = []

    # Group by type
    by_type = {}
    for spec in param_specs:
        type_name = spec.param_type.value.title()
        if type_name not in by_type:
            by_type[type_name] = []
        by_type[type_name].append(spec)

    for type_name, specs in by_type.items():
        lines.append(f"\n{type_name} Parameters:")
        lines.append("-" * (len(type_name) + 12))

        for spec in specs:
            # Name and default
            line = f"  {spec.name}: {spec.description}"
            if spec.units:
                line += f" [{spec.units}]"
            lines.append(line)

            # Default and constraints
            constraints = [f"default={spec.default}"]
            if spec.bounds:
                constraints.append(f"range=[{spec.bounds[0]}, {spec.bounds[1]}]")
            if spec.choices:
                constraints.append(f"choices={spec.choices}")
            if spec.affects:
                constraints.append(f"affects: {spec.affects}")

            lines.append(f"      ({', '.join(constraints)})")

    return "\n".join(lines)
