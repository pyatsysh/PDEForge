# Data Formats

PDEForge datasets can be saved in multiple formats depending on your needs.

## PDEDataset Structure

A `PDEDataset` contains:

| Attribute | Type | Description |
|-----------|------|-------------|
| `inputs` | ndarray | Input fields, shape `(n_samples, *spatial, n_in)` |
| `outputs` | ndarray | Output fields, shape `(n_samples, *spatial, n_out)` |
| `grid` | dict | Spatial coordinates per dimension |
| `metadata` | dict | Model name, parameters, generation info |

## Save Formats

### Directory Format (Default)

```python
dataset.save("./my_dataset")
```

Creates a directory with:

```
my_dataset/
  inputs.npy
  outputs.npy
  grid.npz
  metadata.json
```

Human-readable metadata, easy to inspect with standard tools.

### Compressed NPZ

```python
dataset.save("./my_dataset.npz")
```

Single compressed file containing all arrays. Good for archiving and transfer.

### HDF5

```python
dataset.save("./my_dataset.h5")
```

Requires `h5py`. Supports partial loading for large datasets:

```python
import h5py
with h5py.File("my_dataset.h5", "r") as f:
    batch = f["inputs"][0:100]  # Load only first 100 samples
```

## Loading

```python
from pdeforge import load_dataset

dataset = load_dataset("./my_dataset")      # Directory
dataset = load_dataset("./my_dataset.npz")  # NPZ
dataset = load_dataset("./my_dataset.h5")   # HDF5
```

## Integration with PyTorch

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

dataset = load_dataset("./my_dataset")

torch_dataset = TensorDataset(
    torch.from_numpy(dataset.inputs).float(),
    torch.from_numpy(dataset.outputs).float(),
)

loader = DataLoader(torch_dataset, batch_size=32, shuffle=True)
```

## Data Shapes

### 1D Problems (e.g., Burgers)

```python
inputs.shape   # (n_samples, nx)
outputs.shape  # (n_samples, nx)
```

### 2D Scalar Problems (e.g., Darcy)

```python
inputs.shape   # (n_samples, nx, ny)
outputs.shape  # (n_samples, nx, ny)
```

### 2D Vector Problems (e.g., Stokes)

```python
inputs.shape   # (n_samples, nx, ny, 2)  # (fx, fy)
outputs.shape  # (n_samples, nx, ny, 3)  # (u, v, p)
```

### Time-Dependent Problems

```python
inputs.shape   # (n_samples, nx, ny, n_in)
outputs.shape  # (n_samples, nt, nx, ny, n_out)
```

## Reading foreign datasets (interop)

Some published datasets are worth *reading* rather than regenerating. PDEForge
regenerates a setup only when it can state a measured error against the
original; where a faithful recreation would mean shipping a whole different
solver, the honest move is interop — bring the data onto the same
`PDEDataset` surface so the splits, calibration and observation-operator
machinery applies to it, and cite the authors.

### AirfRANS

[AirfRANS](https://airfrans.readthedocs.io) (Bonnet et al., NeurIPS 2022
Datasets & Benchmarks) is 1000 RANS solutions over 2D airfoils: NACA 4- and
5-digit shapes with continuously sampled digits, $Re \in [2, 6] \times 10^6$,
angle of attack in $[-5°, 15°]$, each a ~180k-node unstructured point cloud
from OpenFOAM.

```python
from pdeforge import load_airfrans

d = load_airfrans("/path/to/AirfRANS/Dataset", split="full_train",
                  n_samples=64, n_points=16384, seed=0)

d.inputs.shape   # (64, 16384, 8)  x, y, u_inf_x, u_inf_y, sdf, n_x, n_y, surface
d.outputs.shape  # (64, 16384, 4)  u, v, p, nu_t
```

`p` and `nu_t` are kinematic (per unit density), matching the source files.
The first seven input channels are the canonical AirfRANS features; the eighth
is their boolean wall flag, a mask rather than a physical feature — slice
`inputs[..., :7]` for the canonical setup.

**Splits are theirs, not ours.** `split=` takes a manifest key
(`full_train`, `full_test`, `scarce_train`, `reynolds_train`, `reynolds_test`,
`aoa_train`, `aoa_test`), so published comparisons stay comparable. Use
`split="all"` for every case on disk.

**Node count.** The meshes differ in size per case, so a common count is
required to stack them; `n_points` subsamples. This is a subsample of the real
solution, not an interpolation of it. `keep_surface=True` (the default) keeps
*every* airfoil wall node and subsamples only the interior — the wall is ~0.6%
of the cloud, so a uniform draw would keep roughly 90 of ~994 wall nodes and
gut the quantity most aerodynamic targets depend on.

**Per-case parameters** are decoded from the case names into
`metadata["case_params"]` (inlet velocity, angle of attack, NACA digits,
series, derived Reynolds number).

A physical check that the conventions are wired correctly:

```python
from pdeforge.io.airfrans import surface_pressure

cp = surface_pressure(d, 0)      # C_p = (p/rho) / (0.5 |U_inf|^2)
cp["cp"].max()                   # 1.00 at the stagnation point
```

!!! note "No VTK dependency"
    AirfRANS ships as VTK XML (`.vtu` / `.vtp`). PDEForge reads it with a
    small built-in parser (`pdeforge.read_vtk_xml`) rather than depending on
    `vtk`, `pyvista` or `meshio` — one interop loader should not drag a
    visualisation stack into the install. It handles inline base64 data,
    zlib-compressed or not, and raises on appended-raw files instead of
    returning something subtly wrong.

For airfoil data **with knobs** rather than a fixed download, see
`naca_flow_2d` (laminar incompressible FEM, geometry as a distribution) and
the transonic `airfoil_euler_2d` model.
