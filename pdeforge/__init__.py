"""
PDEForge: generate PDE datasets for operator learning.
"""

# import models to trigger registration
from pdeforge import models as _models
from pdeforge.core.base import PDEModel
from pdeforge.core.registry import (
    describe_all_models,
    describe_model,
    get_model,
    list_models,
    register_model,
)
from pdeforge.core.types import Domain, GridSpec, PDEDataset
from pdeforge.exploration import (
    explore_model,
    explore_parameter,
    explore_parameter_grid,
    visualize_parameter_effect,
)
from pdeforge.generators.initial_conditions import (
    FourierICGenerator,
    GaussianRandomFieldGenerator,
)

_LARGE_DATASET_THRESHOLD = 5000


def generate_dataset(
    model=None,
    n_samples=None,
    resolution=None,
    domain=None,
    params=None,
    ic_generator="fourier",
    ic_params=None,
    seed=None,
    validate=True,
    n_jobs=1,
    verbose=True,
    backend="auto",
    outputs="final",
    preset=None,
    to=None,
    chunk_size=256,
):
    """
    Generate dataset from a PDE model.

    model: name of PDE model (e.g. "burgers_1d", "darcy_2d")
    n_samples: how many samples
    resolution: grid resolution, e.g. {"x": 256} or {"x": 64, "y": 64}
    domain: bounds per dim, defaults to unit
    params: model-specific params
    ic_generator: type of IC generator
    backend: solver backend — "auto" (default; resolves to numpy for spectral
        models, fenicsx for FEM models) or explicitly "numpy" / "jax".
        JAX is opt-in: install with pip install pdeforge[jax].
    outputs: "final" (default) or "trajectory" for full rollouts shaped
        (n_samples, n_t, *spatial) with a "t" coordinate in the grid.
    preset: name of a classic-benchmark preset (see pdeforge.list_presets());
        fills model/params/ic settings, which explicit arguments override.
    to: optional output path for chunked-to-disk generation (directory for
        memmap-lazy .npy, or .h5). Removes the RAM ceiling: samples are
        written in chunks of chunk_size and never all held in memory.
    """
    # tip for large datasets
    if verbose and n_samples >= _LARGE_DATASET_THRESHOLD:
        import sys

        print(
            f"Generating {n_samples:,} samples... This may take a while.\n"
            f"Tip: save with dataset.save('./my_data') for reuse.",
            file=sys.stderr,
        )

    if preset is not None:
        from pdeforge.presets import get_preset

        cfg = get_preset(preset)
        model = model or cfg["model"]
        params = {**cfg.get("params", {}), **(params or {})}
        if ic_generator == "fourier":  # not explicitly overridden
            ic_generator = cfg.get("ic_generator", "fourier")
        ic_params = {**cfg.get("ic_params", {}), **(ic_params or {})}
        # A preset may pin the domain and a native resolution (some setups are
        # only themselves on their own box); explicit arguments still win.
        if domain is None:
            domain = cfg.get("domain")
        if resolution is None:
            resolution = cfg.get("resolution")
        if outputs == "final" and "outputs" in cfg:
            outputs = cfg["outputs"]

    if model is None or n_samples is None or resolution is None:
        raise TypeError(
            "generate_dataset requires model, n_samples, and resolution "
            "(or a preset supplying the model and a native resolution)."
        )

    model_cls = get_model(model)

    # domain=None passes through: PDEModel defaults to the unit box, and
    # models with a natural non-unit domain (e.g. Kuramoto-Sivashinsky)
    # supply their own default.
    if params is None:
        params = {}
    if ic_params is None:
        ic_params = {}

    from pdeforge.solvers.ops import resolve_backend

    resolved = resolve_backend(model_cls, backend)

    pde_model = model_cls(
        resolution=resolution, domain=domain, backend=resolved, **params
    )

    # Chunked-to-disk path: write incrementally, return a lazy handle.
    if to is not None:
        from pathlib import Path

        from pdeforge.io.datasets import load_dataset
        from pdeforge.io.streaming import generate_streaming

        out_path = generate_streaming(
            pde_model,
            n_samples,
            to,
            ic_generator=ic_generator,
            ic_params=ic_params,
            seed=seed,
            validate=validate,
            n_jobs=n_jobs,
            verbose=verbose,
            outputs=outputs,
            chunk_size=chunk_size,
        )
        if Path(out_path).is_dir():
            return PDEDataset.load(out_path, mmap=True)
        return load_dataset(out_path)

    # Forward outputs= only when non-default: models that override
    # generate_dataset (stochastic, FEM) predate the kwarg.
    extra = {} if outputs == "final" else {"outputs": outputs}

    return pde_model.generate_dataset(
        n_samples=n_samples,
        ic_generator=ic_generator,
        ic_params=ic_params,
        seed=seed,
        validate=validate,
        n_jobs=n_jobs,
        verbose=verbose,
        **extra,
    )


def reproduce(source, verbose=True, n_jobs=1):
    """
    Regenerate a dataset from its own metadata — "dataset as code".

    source: a metadata dict, a metadata.json path, or a saved-dataset path
        (directory / .npz / .h5).

    The metadata written by generate_dataset() records model, resolution,
    domain, params, IC generator settings, and seed; this function replays
    them. Datasets produced by split() cannot be reproduced directly —
    reproduce the parent (full) dataset, then re-split with the recorded
    split_seed and split_fractions.
    """
    import json
    from pathlib import Path

    if isinstance(source, dict):
        meta = dict(source)
    else:
        path = Path(source)
        if path.is_file() and path.suffix == ".json":
            with open(path) as f:
                meta = json.load(f)
        else:
            from pdeforge.io.datasets import load_dataset

            meta = dict(load_dataset(path).metadata)

    if "split" in meta:
        raise ValueError(
            f"This metadata describes a '{meta['split']}' split, not a full "
            "dataset. Reproduce the parent dataset, then call "
            f".split(seed={meta.get('split_seed')}, "
            f"**{meta.get('split_fractions')})."
        )

    if meta.get("ic_generator") == "custom":
        raise ValueError(
            "Dataset was generated with a custom (callable) IC generator, "
            "which metadata cannot capture. Re-run with the original callable."
        )

    if meta.get("seed") is None:
        raise ValueError(
            "Dataset was generated without a seed and cannot be reproduced "
            "bit-for-bit. Always pass seed= for reproducible datasets."
        )

    domain = {k: tuple(v) for k, v in meta["domain"].items()}

    return generate_dataset(
        model=meta["model"],
        n_samples=meta["n_samples"],
        resolution=meta["resolution"],
        domain=domain,
        params=meta.get("params") or {},
        ic_generator=meta.get("ic_generator", "fourier"),
        ic_params=meta.get("ic_params") or {},
        seed=meta["seed"],
        verbose=verbose,
        n_jobs=n_jobs,
        backend=meta.get("backend", "auto"),
        outputs=meta.get("outputs", "final"),
    )


from pdeforge._version import __version__
from pdeforge.io.airfrans import load_airfrans
from pdeforge.io.darcy_fno import load_darcy_fno
from pdeforge.io.datasets import export_pdebench, load_dataset, save_dataset
from pdeforge.io.torch_pt import read_torch_pt
from pdeforge.io.vtk_xml import read_vtk_xml
from pdeforge.presets import list_presets

__all__ = [
    "generate_dataset",
    "reproduce",
    "load_dataset",
    "save_dataset",
    "export_pdebench",
    "load_airfrans",
    "load_darcy_fno",
    "read_vtk_xml",
    "read_torch_pt",
    "list_presets",
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
