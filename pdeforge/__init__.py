"""
PDEForge: generate PDE datasets for operator learning.
"""

from pdeforge.core.registry import register_model, get_model, list_models, describe_model, describe_all_models
from pdeforge.core.base import PDEModel
from pdeforge.core.types import PDEDataset, Domain, GridSpec
from pdeforge.generators.initial_conditions import (
    FourierICGenerator,
    GaussianRandomFieldGenerator,
)
from pdeforge.exploration import (
    explore_parameter,
    explore_parameter_grid,
    explore_model,
    visualize_parameter_effect,
)

# import models to trigger registration
from pdeforge import models as _models

_LARGE_DATASET_THRESHOLD = 5000


def generate_dataset(model, n_samples, resolution, domain=None, params=None,
                     ic_generator="fourier", ic_params=None, seed=None,
                     validate=True, n_jobs=1, verbose=True):
    """
    Generate dataset from a PDE model.

    model: name of PDE model (e.g. "burgers_1d", "darcy_2d")
    n_samples: how many samples
    resolution: grid resolution, e.g. {"x": 256} or {"x": 64, "y": 64}
    domain: bounds per dim, defaults to unit
    params: model-specific params
    ic_generator: type of IC generator
    """
    # tip for large datasets
    if verbose and n_samples >= _LARGE_DATASET_THRESHOLD:
        import sys
        print(
            f"Generating {n_samples:,} samples... This may take a while.\n"
            f"Tip: save with dataset.save('./my_data') for reuse.",
            file=sys.stderr
        )

    model_cls = get_model(model)

    if domain is None:
        domain = {k: (0.0, 1.0) for k in resolution.keys()}
    if params is None:
        params = {}
    if ic_params is None:
        ic_params = {}

    pde_model = model_cls(
        resolution=resolution,
        domain=domain,
        **params
    )

    return pde_model.generate_dataset(
        n_samples=n_samples,
        ic_generator=ic_generator,
        ic_params=ic_params,
        seed=seed,
        validate=validate,
        n_jobs=n_jobs,
        verbose=verbose,
    )


__version__ = "0.1.0"
__all__ = [
    "generate_dataset",
    "list_models",
    "describe_model",
    "describe_all_models",
    "get_model",
    "explore_parameter",
    "explore_parameter_grid",
    "explore_model",
    "visualize_parameter_effect",
    "register_model",
    "PDEModel",
    "PDEDataset",
    "Domain",
    "GridSpec",
    "FourierICGenerator",
    "GaussianRandomFieldGenerator",
]
