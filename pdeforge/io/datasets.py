"""
Dataset I/O utilities for PDEForge.

This module provides functions for saving and loading datasets
in various formats.
"""

import numpy as np
from pathlib import Path
import json
from typing import Union, Dict, Any

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
        else:
            format = "directory"
    
    if format == "directory":
        dataset.save(path)
    elif format == "npz":
        export_to_npz(dataset, path)
    elif format == "hdf5":
        export_to_hdf5(dataset, path)
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
    
    if path.is_dir():
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
        'inputs': dataset.inputs,
        'outputs': dataset.outputs,
    }
    
    # Add grid arrays
    for dim, coords in dataset.grid.items():
        data[f'grid_{dim}'] = coords
    
    # Save metadata as JSON string in array
    metadata = {
        **dataset.metadata,
        'input_names': dataset.input_names,
        'output_names': dataset.output_names,
        'grid_dims': list(dataset.grid.keys()),
    }
    data['metadata'] = np.array([json.dumps(metadata, default=str)])
    
    np.savez_compressed(path, **data)
    print(f"Dataset exported to {path}")


def _load_from_npz(path: Union[str, Path]) -> PDEDataset:
    """Load dataset from .npz file."""
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    
    inputs = data['inputs']
    outputs = data['outputs']
    
    # Load metadata
    metadata = json.loads(data['metadata'][0])
    grid_dims = metadata.pop('grid_dims')
    input_names = metadata.pop('input_names', ["input"])
    output_names = metadata.pop('output_names', ["output"])
    
    # Load grid
    grid = {dim: data[f'grid_{dim}'] for dim in grid_dims}
    
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
        raise ImportError("h5py is required for HDF5 export. Install with: pip install h5py")
    
    path = Path(path)
    
    with h5py.File(path, 'w') as f:
        # Save arrays
        f.create_dataset('inputs', data=dataset.inputs, compression='gzip')
        f.create_dataset('outputs', data=dataset.outputs, compression='gzip')
        
        # Save grid
        grid_grp = f.create_group('grid')
        for dim, coords in dataset.grid.items():
            grid_grp.create_dataset(dim, data=coords)
        
        # Save metadata as attributes
        f.attrs['input_names'] = json.dumps(dataset.input_names)
        f.attrs['output_names'] = json.dumps(dataset.output_names)
        f.attrs['metadata'] = json.dumps(dataset.metadata, default=str)
    
    print(f"Dataset exported to {path}")


def _load_from_hdf5(path: Union[str, Path]) -> PDEDataset:
    """Load dataset from HDF5 file."""
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required for HDF5 loading. Install with: pip install h5py")
    
    path = Path(path)
    
    with h5py.File(path, 'r') as f:
        inputs = f['inputs'][:]
        outputs = f['outputs'][:]
        
        # Load grid
        grid = {dim: f['grid'][dim][:] for dim in f['grid'].keys()}
        
        # Load metadata
        input_names = json.loads(f.attrs['input_names'])
        output_names = json.loads(f.attrs['output_names'])
        metadata = json.loads(f.attrs['metadata'])
    
    return PDEDataset(
        inputs=inputs,
        outputs=outputs,
        grid=grid,
        metadata=metadata,
        input_names=input_names,
        output_names=output_names,
    )
