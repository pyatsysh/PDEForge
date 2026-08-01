"""
Read the distributed canonical Darcy files into a PDEDataset.

The FNO Darcy benchmark ships as ``torch.save`` archives (Zenodo record
12784353: ``darcy_train_421.pt`` / ``darcy_test_421.pt``, dicts of ``x`` =
coefficient a and ``y`` = solution u). They are read here without PyTorch,
memory-mapped, via :mod:`pdeforge.io.torch_pt`.

The point of loading them is comparison: ``darcy_fno_2d`` regenerates this
family from scratch, and solving these stored coefficients with it returns
the stored solutions to float32 round-off (tests/test_canonical.py). Use the
published file when you need those exact arrays; use the model when you need
a different resolution, more samples, or different measure parameters.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from pdeforge.core.types import PDEDataset
from pdeforge.io.torch_pt import read_torch_pt

__all__ = ["load_darcy_fno"]


def load_darcy_fno(
    path,
    n_samples: Optional[int] = None,
    resolution: Optional[int] = None,
    mmap: bool = True,
    verbose: bool = True,
) -> PDEDataset:
    """
    Load a distributed Darcy ``.pt`` file.

    path : a ``darcy_{train,test}_<res>.pt`` file.
    n_samples : take only the first n samples (None = all).
    resolution : subsample the grid to this many points per side. Only the
        strides that hit the stored grid exactly are allowed -- the published
        low-resolution files are strided views of the 421 master grid
        (421, 211, 141, 106, 85 for strides 1..5), and anything else would be
        an interpolation dressed up as a resolution.
    mmap : keep the arrays memory-mapped (the 421 files are ~7 GB each).

    The stored fields are sampled at CELL CENTRES, (2i+1)/(2K) -- see
    :mod:`pdeforge.models.darcy_fno_2d` -- so that is the grid returned.
    """
    path = Path(path)
    d = read_torch_pt(path, mmap=mmap, keys=["x", "y"])
    a, u = d["x"], d["y"]
    if a.shape != u.shape or a.ndim != 3:
        raise ValueError(
            f"{path}: expected matching (N, K, K) arrays, got {a.shape} / {u.shape}"
        )

    if n_samples is not None:
        a, u = a[:n_samples], u[:n_samples]
    k_full = a.shape[-1]

    stride = 1
    if resolution is not None and resolution != k_full:
        stride, rem = divmod(k_full - 1, resolution - 1)
        if rem or stride < 1 or (k_full - 1) // stride + 1 != resolution:
            raise ValueError(
                f"resolution {resolution} is not a stride of the stored "
                f"{k_full} grid; available: "
                + ", ".join(str((k_full - 1) // s + 1) for s in range(1, 6))
            )
        a, u = a[:, ::stride, ::stride], u[:, ::stride, ::stride]

    k = a.shape[-1]
    centres = (2 * np.arange(k_full) + 1) / (2 * k_full)
    grid = {"x": centres[::stride], "y": centres[::stride]}
    if verbose:
        print(f"Loaded {a.shape[0]} samples at {k}x{k} from {path.name}")

    return PDEDataset(
        inputs=a,
        outputs=u,
        grid=grid,
        metadata={
            "source": "darcy_fno",
            "file": path.name,
            "n_samples": int(a.shape[0]),
            "resolution": k,
            "stored_resolution": k_full,
            "stride": stride,
            "grid_convention": "cell centres (2i+1)/(2K)",
            "regenerate_with": {
                "model": "darcy_fno_2d",
                "params": {"coeff": "lognormal", "alpha": 2.0, "tau": 3.0},
            },
            "reference": (
                "Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart, "
                "Anandkumar (2021), Fourier Neural Operator for Parametric "
                "PDEs, ICLR; data Zenodo 12784353"
            ),
        },
        input_names=["a"],
        output_names=["u"],
    )
