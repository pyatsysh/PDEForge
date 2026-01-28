# Unified API

PDEForge provides a single interface for all PDE models. This consistency simplifies experimentation across different physical systems.

## The generate_dataset Function

```python
from pdeforge import generate_dataset

dataset = generate_dataset(
    model: str,              # Model identifier
    n_samples: int,          # Number of samples to generate
    resolution: dict,        # Grid resolution per dimension
    domain: dict = None,     # Domain bounds (default: unit domain)
    params: dict = None,     # Model-specific parameters
    ic_generator: str = "fourier",  # Initial condition type
    ic_params: dict = None,  # IC generator settings
    seed: int = None,        # Random seed
    validate: bool = True,   # Check solution validity
    n_jobs: int = 1,         # Parallel workers
    verbose: bool = True,    # Progress output
)
```

### Parameters

**model**: Name of the registered PDE model. Use `list_models()` to see available options.

**n_samples**: How many input-output pairs to generate.

**resolution**: Dictionary mapping dimension names to grid points:
```python
resolution={"x": 256}           # 1D
resolution={"x": 64, "y": 64}   # 2D
```

**domain**: Dictionary mapping dimension names to (min, max) tuples:
```python
domain={"x": (0, 2*np.pi)}      # Custom domain
```

**params**: Model-specific physical parameters. Use `describe_model()` to see what each model accepts.

**seed**: Integer for reproducible dataset generation.

## The PDEDataset Object

`generate_dataset` returns a `PDEDataset` with:

```python
dataset.inputs      # ndarray: (n_samples, *spatial_dims, n_input_channels)
dataset.outputs     # ndarray: (n_samples, *spatial_dims, n_output_channels)
dataset.grid        # dict: {"x": array, "y": array, ...}
dataset.metadata    # dict: model name, params, generation info
```

### Splitting

```python
splits = dataset.split(train=0.6, val=0.15, cal=0.15, test=0.1)

# Access individual splits
train_data = splits['train']
cal_data = splits['cal']
```

The calibration split (`cal`) is intended for uncertainty quantification methods like conformal prediction.

### Saving and Loading

```python
# Directory format (human-readable)
dataset.save("./data/my_dataset")

# Compressed NPZ
dataset.save("./data/my_dataset.npz")

# HDF5 (requires h5py)
dataset.save("./data/my_dataset.h5")

# Loading
from pdeforge import load_dataset
dataset = load_dataset("./data/my_dataset")
```

## Discovering Models

```python
from pdeforge import list_models, describe_model

# All available models
print(list_models())

# Detailed info about a model
info = describe_model("darcy_2d")
print(info)
```

## Working with Model Classes Directly

For advanced use cases:

```python
from pdeforge import get_model

# Get the model class
Burgers = get_model("burgers_1d")

# Instantiate with custom settings
model = Burgers(
    resolution={"x": 256},
    domain={"x": (0, 2*np.pi)},
    viscosity=0.005,
)

# Generate individual samples
ic = model.generate_ic(seed=42)
solution = model.solve(ic)
```
