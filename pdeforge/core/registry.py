"""
Model registry for PDEForge.

This module implements a factory pattern for registering and retrieving PDE models.
New models can be added by using the @register_model decorator.
"""

from typing import Dict, List, Optional, Type

# Global registry of PDE models
_MODEL_REGISTRY: Dict[str, Type] = {}


def register_model(name: str):
    """
    Decorator to register a PDE model class.

    Parameters
    ----------
    name : str
        Name to register the model under (e.g., "burgers_1d")

    Returns
    -------
    callable
        Decorator function

    Examples
    --------
    >>> @register_model("my_pde")
    ... class MyPDE(PDEModel):
    ...     def solve(self, ic):
    ...         ...
    """

    def decorator(cls):
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered")
        _MODEL_REGISTRY[name] = cls
        cls._registered_name = name
        return cls

    return decorator


def get_model(name: str) -> Type:
    """
    Get a registered PDE model class by name.

    Parameters
    ----------
    name : str
        Name of the registered model

    Returns
    -------
    Type
        The PDE model class

    Raises
    ------
    ValueError
        If the model is not registered

    Examples
    --------
    >>> model_cls = get_model("burgers_1d")
    >>> model = model_cls(resolution={"x": 256})
    """
    if name not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{name}' not found. Available models: {available}")
    return _MODEL_REGISTRY[name]


def list_models() -> List[str]:
    """
    List all registered model names.

    Returns
    -------
    List[str]
        List of registered model names

    Examples
    --------
    >>> print(list_models())
    ['burgers_1d', 'darcy_2d', 'stokes_2d']
    """
    return sorted(_MODEL_REGISTRY.keys())


def get_model_info(name: str) -> Dict:
    """
    Get information about a registered model.

    Parameters
    ----------
    name : str
        Name of the registered model

    Returns
    -------
    Dict
        Dictionary with model information
    """
    model_cls = get_model(name)

    # Get user-facing parameters
    user_params = {}
    for param in getattr(model_cls, "USER_PARAMS", []):
        user_params[param.name] = param.to_dict()

    return {
        "name": name,
        "class": model_cls.__name__,
        "docstring": model_cls.__doc__,
        "ndim": getattr(model_cls, "NDIM", None),
        "backend": getattr(model_cls, "BACKEND", "spectral"),
        "inputs": getattr(model_cls, "INPUT_NAMES", []),
        "outputs": getattr(model_cls, "OUTPUT_NAMES", []),
        "user_params": user_params,
    }


def describe_model(name: str) -> str:
    """
    Get a formatted description of a model for display.

    Parameters
    ----------
    name : str
        Name of the registered model

    Returns
    -------
    str
        Human-readable model description

    Examples
    --------
    >>> print(describe_model("burgers_1d"))
    Model: burgers_1d
    ...
    """
    model_cls = get_model(name)
    return model_cls.describe()


def describe_all_models(verbose: bool = False) -> str:
    """
    Get a formatted description of all registered models.

    Parameters
    ----------
    verbose : bool
        If True, include full parameter descriptions.
        If False, show compact summary (default).

    Returns
    -------
    str
        Human-readable summary of all models

    Examples
    --------
    >>> print(describe_all_models())
    PDEForge Models
    ===============
    ...
    """
    lines = []
    lines.append("=" * 60)
    lines.append("PDEForge Models")
    lines.append("=" * 60)
    lines.append("")

    for name in list_models():
        model_cls = get_model(name)

        # Basic info
        ndim = getattr(model_cls, "NDIM", None)
        ndim_str = f"{ndim}D" if ndim else "2D/3D"
        backend = getattr(model_cls, "BACKEND", "spectral")
        inputs = getattr(model_cls, "INPUT_NAMES", [])
        outputs = getattr(model_cls, "OUTPUT_NAMES", [])

        lines.append(f"[{name}]")
        lines.append(f"  Dimensions: {ndim_str}  |  Backend: {backend}")
        lines.append(f"  Task: {', '.join(inputs)} → {', '.join(outputs)}")

        # User parameters
        user_params = getattr(model_cls, "USER_PARAMS", [])
        if user_params:
            lines.append(f"  Parameters:")
            for param in user_params:
                if verbose:
                    bounds = (
                        f"[{param.bounds[0]}, {param.bounds[1]}]"
                        if param.bounds
                        else ""
                    )
                    units = f" ({param.units})" if param.units else ""
                    lines.append(f"    - {param.name}: {param.description}")
                    lines.append(
                        f"        default={param.default}{units}, range={bounds}"
                    )
                else:
                    default = param.default
                    if isinstance(default, float) and abs(default) < 0.01:
                        default_str = f"{default:.2e}"
                    else:
                        default_str = f"{default}"
                    lines.append(f"    - {param.name} (default: {default_str})")
        else:
            lines.append(f"  Parameters: (none exposed)")

        lines.append("")

    lines.append("=" * 60)
    lines.append(f"Total: {len(list_models())} models")
    lines.append("")
    lines.append("Use describe_model('name') for detailed info on a specific model.")

    return "\n".join(lines)
