"""
Type definitions and data structures for PDEForge.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class Domain:
    """
    Represents a computational domain.

    Attributes
    ----------
    bounds : Dict[str, Tuple[float, float]]
        Dictionary mapping dimension names to (min, max) bounds
        e.g., {"x": (0, 1), "y": (0, 1)}
    periodic : Dict[str, bool]
        Dictionary mapping dimension names to periodicity flags
    """

    bounds: Dict[str, Tuple[float, float]]
    periodic: Dict[str, bool] = field(default_factory=dict)

    def __post_init__(self):
        # Default to periodic in all dimensions if not specified
        for dim in self.bounds:
            if dim not in self.periodic:
                self.periodic[dim] = True

    @property
    def ndim(self) -> int:
        return len(self.bounds)

    @property
    def dims(self) -> List[str]:
        return list(self.bounds.keys())

    def size(self, dim: str) -> float:
        """Get the size of the domain in a given dimension."""
        low, high = self.bounds[dim]
        return high - low


@dataclass
class GridSpec:
    """
    Specification for a computational grid.

    Attributes
    ----------
    resolution : Dict[str, int]
        Number of grid points in each dimension
    domain : Domain
        The computational domain
    """

    resolution: Dict[str, int]
    domain: Domain

    def __post_init__(self):
        # Validate that resolution and domain have same dimensions
        if set(self.resolution.keys()) != set(self.domain.dims):
            raise ValueError(
                f"Resolution keys {set(self.resolution.keys())} must match "
                f"domain dimensions {set(self.domain.dims)}"
            )

    def get_grid(self, dim: str, endpoint: bool = False) -> np.ndarray:
        """
        Get grid points for a given dimension.

        Parameters
        ----------
        dim : str
            Dimension name
        endpoint : bool
            Whether to include the endpoint (for periodic domains, usually False)

        Returns
        -------
        np.ndarray
            1D array of grid points
        """
        low, high = self.domain.bounds[dim]
        n = self.resolution[dim]
        return np.linspace(low, high, n, endpoint=endpoint)

    def get_meshgrid(self, indexing: str = "ij") -> Tuple[np.ndarray, ...]:
        """
        Get meshgrid for all dimensions.

        Parameters
        ----------
        indexing : str
            'ij' for matrix indexing, 'xy' for Cartesian indexing

        Returns
        -------
        Tuple of np.ndarray
            Meshgrid arrays for each dimension
        """
        grids = [self.get_grid(dim) for dim in sorted(self.domain.dims)]
        return np.meshgrid(*grids, indexing=indexing)

    def get_spacing(self, dim: str) -> float:
        """Get grid spacing in a given dimension."""
        return self.domain.size(dim) / self.resolution[dim]

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the grid."""
        return tuple(self.resolution[dim] for dim in sorted(self.resolution.keys()))


@dataclass
class PDEDataset:
    """
    A dataset of PDE solutions for operator learning.

    Attributes
    ----------
    inputs : np.ndarray
        Input fields, shape (n_samples, *spatial_dims, n_input_channels)
    outputs : np.ndarray
        Output fields, shape (n_samples, *spatial_dims, n_output_channels)
    grid : Dict[str, np.ndarray]
        Grid coordinates for each dimension
    metadata : Dict
        Metadata including model parameters, generation info, etc.
    input_names : List[str]
        Names of input channels
    output_names : List[str]
        Names of output channels
    """

    inputs: np.ndarray
    outputs: np.ndarray
    grid: Dict[str, np.ndarray]
    metadata: Dict[str, Any]
    input_names: List[str] = field(default_factory=lambda: ["input"])
    output_names: List[str] = field(default_factory=lambda: ["output"])

    def __post_init__(self):
        # Ensure arrays are numpy arrays
        self.inputs = np.asarray(self.inputs)
        self.outputs = np.asarray(self.outputs)

    @property
    def n_samples(self) -> int:
        return self.inputs.shape[0]

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return self.inputs.shape[1:]

    @property
    def output_shape(self) -> Tuple[int, ...]:
        return self.outputs.shape[1:]

    def split(
        self,
        train: float = 0.6,
        val: float = 0.15,
        cal: float = 0.15,
        test: float = 0.1,
        seed: int = None,
    ) -> Dict[str, "PDEDataset"]:
        """
        Split the dataset into train/val/calibration/test sets.

        Parameters
        ----------
        train, val, cal, test : float
            Fractions for each split (should sum to 1)
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        Dict[str, PDEDataset]
            Dictionary with keys 'train', 'val', 'cal', 'test'
        """
        assert abs(train + val + cal + test - 1.0) < 1e-6, "Fractions must sum to 1"

        rng = np.random.default_rng(seed)
        indices = rng.permutation(self.n_samples)

        n_train = int(train * self.n_samples)
        n_val = int(val * self.n_samples)
        n_cal = int(cal * self.n_samples)

        splits = {
            "train": indices[:n_train],
            "val": indices[n_train : n_train + n_val],
            "cal": indices[n_train + n_val : n_train + n_val + n_cal],
            "test": indices[n_train + n_val + n_cal :],
        }

        result = {}
        for name, idx in splits.items():
            if len(idx) > 0:
                result[name] = PDEDataset(
                    inputs=self.inputs[idx],
                    outputs=self.outputs[idx],
                    grid=self.grid.copy(),
                    metadata={**self.metadata, "split": name},
                    input_names=self.input_names,
                    output_names=self.output_names,
                )

        return result

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the dataset to disk.

        Parameters
        ----------
        path : str or Path
            Path to save directory (will be created if doesn't exist)
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save arrays
        np.save(path / "inputs.npy", self.inputs)
        np.save(path / "outputs.npy", self.outputs)

        # Save grid
        for dim, coords in self.grid.items():
            np.save(path / f"grid_{dim}.npy", coords)

        # Save metadata as JSON
        metadata = {
            **self.metadata,
            "input_names": self.input_names,
            "output_names": self.output_names,
            "grid_dims": list(self.grid.keys()),
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"Dataset saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PDEDataset":
        """
        Load a dataset from disk.

        Parameters
        ----------
        path : str or Path
            Path to saved dataset directory

        Returns
        -------
        PDEDataset
            Loaded dataset
        """
        path = Path(path)

        # Load arrays
        inputs = np.load(path / "inputs.npy")
        outputs = np.load(path / "outputs.npy")

        # Load metadata
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)

        # Load grid
        grid_dims = metadata.pop("grid_dims")
        grid = {dim: np.load(path / f"grid_{dim}.npy") for dim in grid_dims}

        input_names = metadata.pop("input_names", ["input"])
        output_names = metadata.pop("output_names", ["output"])

        return cls(
            inputs=inputs,
            outputs=outputs,
            grid=grid,
            metadata=metadata,
            input_names=input_names,
            output_names=output_names,
        )

    def visualize(self):
        """
        Launch interactive visualization.

        This opens an ipywidgets-based viewer for exploring the dataset.
        """
        from pdeforge.visualization.interactive import DatasetExplorer

        explorer = DatasetExplorer(self)
        return explorer.show()

    def __repr__(self) -> str:
        return (
            f"PDEDataset(\n"
            f"  n_samples={self.n_samples},\n"
            f"  input_shape={self.input_shape},\n"
            f"  output_shape={self.output_shape},\n"
            f"  input_names={self.input_names},\n"
            f"  output_names={self.output_names},\n"
            f"  model={self.metadata.get('model', 'unknown')}\n"
            f")"
        )
