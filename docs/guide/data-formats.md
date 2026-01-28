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
