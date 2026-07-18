"""
Dataset I/O utilities for PDEForge.

This module provides functions for saving and loading datasets
in various formats.
"""

import json
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from pdeforge.core.types import PDEDataset


def save_dataset(
    dataset: PDEDataset,
    path: Union[str, Path],
    format: str = "auto",
) -> None:
    """
    Save a dataset to disk.

    Parameters
    ----------
    dataset : PDEDataset
        Dataset to save
    path : str or Path
        Output path
    format : str
        Format: "auto", "directory", "npz", or "hdf5"
        "auto" chooses based on path extension
    """
    path = Path(path)

    if format == "auto":
        if path.suffix == ".npz":
            format = "npz"
        elif path.suffix in [".h5", ".hdf5"]:
            format = "hdf5"
        elif path.suffix == ".zarr":
            format = "zarr"
        else:
            format = "directory"

    if format == "directory":
        dataset.save(path)
    elif format == "npz":
        export_to_npz(dataset, path)
    elif format == "hdf5":
        export_to_hdf5(dataset, path)
    elif format == "zarr":
        export_to_zarr(dataset, path)
    else:
        raise ValueError(f"Unknown format: {format}")


def load_dataset(path: Union[str, Path]) -> PDEDataset:
    """
    Load a dataset from disk.

    Parameters
    ----------
    path : str or Path
        Path to saved dataset

    Returns
    -------
    PDEDataset
        Loaded dataset
    """
    path = Path(path)

    if path.suffix == ".zarr":
        return _load_from_zarr(path)
    elif path.is_dir():
        return PDEDataset.load(path)
    elif path.suffix == ".npz":
        return _load_from_npz(path)
    elif path.suffix in [".h5", ".hdf5"]:
        return _load_from_hdf5(path)
    else:
        raise ValueError(f"Unknown file format: {path.suffix}")


def export_to_npz(dataset: PDEDataset, path: Union[str, Path]) -> None:
    """
    Export dataset to a single .npz file.

    Parameters
    ----------
    dataset : PDEDataset
        Dataset to export
    path : str or Path
        Output path (.npz)
    """
    path = Path(path)

    # Prepare data dict
    data = {
        "inputs": dataset.inputs,
        "outputs": dataset.outputs,
    }

    # Add grid arrays
    for dim, coords in dataset.grid.items():
        data[f"grid_{dim}"] = coords

    # Save metadata as JSON string in array
    metadata = {
        **dataset.metadata,
        "input_names": dataset.input_names,
        "output_names": dataset.output_names,
        "grid_dims": list(dataset.grid.keys()),
    }
    data["metadata"] = np.array([json.dumps(metadata, default=str)])

    np.savez_compressed(path, **data)
    print(f"Dataset exported to {path}")


def _load_from_npz(path: Union[str, Path]) -> PDEDataset:
    """Load dataset from .npz file."""
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    inputs = data["inputs"]
    outputs = data["outputs"]

    # Load metadata
    metadata = json.loads(data["metadata"][0])
    grid_dims = metadata.pop("grid_dims")
    input_names = metadata.pop("input_names", ["input"])
    output_names = metadata.pop("output_names", ["output"])

    # Load grid
    grid = {dim: data[f"grid_{dim}"] for dim in grid_dims}

    return PDEDataset(
        inputs=inputs,
        outputs=outputs,
        grid=grid,
        metadata=metadata,
        input_names=input_names,
        output_names=output_names,
    )


def export_to_hdf5(dataset: PDEDataset, path: Union[str, Path]) -> None:
    """
    Export dataset to HDF5 format.

    Parameters
    ----------
    dataset : PDEDataset
        Dataset to export
    path : str or Path
        Output path (.h5 or .hdf5)
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for HDF5 export. Install with: pip install h5py"
        )

    path = Path(path)

    with h5py.File(path, "w") as f:
        # Save arrays
        f.create_dataset("inputs", data=dataset.inputs, compression="gzip")
        f.create_dataset("outputs", data=dataset.outputs, compression="gzip")

        # Save grid
        grid_grp = f.create_group("grid")
        for dim, coords in dataset.grid.items():
            grid_grp.create_dataset(dim, data=coords)

        # Save metadata as attributes
        f.attrs["input_names"] = json.dumps(dataset.input_names)
        f.attrs["output_names"] = json.dumps(dataset.output_names)
        f.attrs["metadata"] = json.dumps(dataset.metadata, default=str)

    print(f"Dataset exported to {path}")


def _load_from_hdf5(path: Union[str, Path]) -> PDEDataset:
    """Load dataset from HDF5 file."""
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for HDF5 loading. Install with: pip install h5py"
        )

    path = Path(path)

    with h5py.File(path, "r") as f:
        inputs = f["inputs"][:]
        outputs = f["outputs"][:]

        # Load grid
        grid = {dim: f["grid"][dim][:] for dim in f["grid"].keys()}

        # Load metadata
        input_names = json.loads(f.attrs["input_names"])
        output_names = json.loads(f.attrs["output_names"])
        metadata = json.loads(f.attrs["metadata"])

    return PDEDataset(
        inputs=inputs,
        outputs=outputs,
        grid=grid,
        metadata=metadata,
        input_names=input_names,
        output_names=output_names,
    )


def export_to_zarr(dataset: PDEDataset, path: Union[str, Path]) -> None:
    """
    Export dataset to a zarr store (cloud/streaming-friendly chunked format).

    Requires the optional zarr dependency: pip install pdeforge[zarr]
    """
    try:
        import zarr
    except ImportError:
        raise ImportError(
            "zarr is required for zarr export. Install with: pip install pdeforge[zarr]"
        )

    path = Path(path)
    root = zarr.open_group(str(path), mode="w")
    root.create_array("inputs", data=dataset.inputs)
    root.create_array("outputs", data=dataset.outputs)
    grid = root.create_group("grid")
    for dim, coords in dataset.grid.items():
        grid.create_array(dim, data=coords)
    root.attrs["input_names"] = dataset.input_names
    root.attrs["output_names"] = dataset.output_names
    root.attrs["metadata"] = json.loads(json.dumps(dataset.metadata, default=str))
    print(f"Dataset exported to {path}")


def _load_from_zarr(path: Union[str, Path]) -> PDEDataset:
    """Load dataset from a zarr store."""
    try:
        import zarr
    except ImportError:
        raise ImportError(
            "zarr is required for zarr loading. Install with: pip install pdeforge[zarr]"
        )

    root = zarr.open_group(str(Path(path)), mode="r")
    # zarr's Group.__getitem__ stubs narrow to Array; annotate around it.
    grid_grp: Any = root["grid"]
    grid = {str(dim): np.asarray(grid_grp[str(dim)]) for dim in grid_grp.keys()}
    return PDEDataset(
        inputs=np.asarray(root["inputs"]),
        outputs=np.asarray(root["outputs"]),
        grid=grid,
        metadata=dict(root.attrs.get("metadata", {})),
        input_names=list(root.attrs.get("input_names", ["input"])),
        output_names=list(root.attrs.get("output_names", ["output"])),
    )


def export_pdebench(dataset: PDEDataset, path: Union[str, Path]) -> None:
    """
    Export in the PDEBench HDF5 layout: a "tensor" dataset shaped
    (n_samples, n_t, *spatial) plus "x-coordinate" / "y-coordinate" /
    "t-coordinate" datasets, so PDEBench-style readers and training
    pipelines can consume PDEForge data directly.

    Final-state datasets export with n_t = 1; trajectory datasets
    (outputs="trajectory") export their full rollout. The OUTPUT field maps
    to "tensor"; inputs and full provenance ride along under "pdeforge/".
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for PDEBench export. Install with: pip install pdeforge[hdf5]"
        )

    path = Path(path)
    out = dataset.outputs
    is_traj = dataset.metadata.get("outputs") == "trajectory"
    if not is_traj:
        out = out[:, None, ...]  # (N, 1, *spatial)

    with h5py.File(path, "w") as f:
        f.create_dataset("tensor", data=out, compression="gzip")
        if "x" in dataset.grid:
            f.create_dataset("x-coordinate", data=dataset.grid["x"])
        if "y" in dataset.grid:
            f.create_dataset("y-coordinate", data=dataset.grid["y"])
        if "t" in dataset.grid:
            f.create_dataset("t-coordinate", data=dataset.grid["t"])
        else:
            f.create_dataset("t-coordinate", data=np.array([0.0]))
        # Provenance + inputs preserved under a namespaced group.
        g = f.create_group("pdeforge")
        g.create_dataset("inputs", data=dataset.inputs, compression="gzip")
        g.attrs["metadata"] = json.dumps(dataset.metadata, default=str)
        g.attrs["input_names"] = json.dumps(dataset.input_names)
        g.attrs["output_names"] = json.dumps(dataset.output_names)
    print(f"Dataset exported (PDEBench layout) to {path}")
