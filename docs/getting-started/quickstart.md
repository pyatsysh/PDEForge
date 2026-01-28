# Quick Start

This guide walks through the basic workflow of generating PDE datasets with PDEForge.

## Discovering Available Models

```python
from pdeforge import list_models, describe_model

# List all registered models
print(list_models())
# ['burgers_1d', 'darcy_2d', 'stokes_2d', 'cylinder_flow_2d', ...]

# Get detailed information about a specific model
print(describe_model("burgers_1d"))
```

## Generating a Dataset

All models use the same `generate_dataset` function:

```python
from pdeforge import generate_dataset

dataset = generate_dataset(
    model="burgers_1d",     # Model name
    n_samples=1000,          # Number of samples
    resolution={"x": 256},   # Grid resolution
    params={"viscosity": 0.01, "time_horizon": 1.0},
    seed=42,                 # For reproducibility
)
```

## Exploring the Dataset

```python
# Basic info
print(dataset)
# PDEDataset(
#   n_samples=1000,
#   input_shape=(256,),
#   output_shape=(256,),
#   model=burgers_1d
# )

# Access arrays
inputs = dataset.inputs    # Shape: (1000, 256)
outputs = dataset.outputs  # Shape: (1000, 256)
grid = dataset.grid        # {"x": array([0, ..., 1])}

# Metadata
print(dataset.metadata)
```

## Splitting for Machine Learning

PDEForge includes a dedicated calibration split for uncertainty quantification:

```python
splits = dataset.split(
    train=0.6,   # 60% training
    val=0.15,    # 15% validation
    cal=0.15,    # 15% calibration (for UQ)
    test=0.1,    # 10% testing
)

X_train, y_train = splits['train'].inputs, splits['train'].outputs
X_cal, y_cal = splits['cal'].inputs, splits['cal'].outputs
```

## Saving and Loading

```python
# Save to directory (includes metadata)
dataset.save("./my_dataset")

# Save as compressed file
dataset.save("./my_dataset.npz")

# Save as HDF5 (requires h5py)
dataset.save("./my_dataset.h5")

# Load later
from pdeforge import load_dataset
dataset = load_dataset("./my_dataset")
```

## Visualization

In Jupyter notebooks:

```python
# Interactive exploration
dataset.visualize()

# Parameter exploration
from pdeforge import explore_parameter, visualize_parameter_effect

results = explore_parameter(
    model="burgers_1d",
    param_name="viscosity",
    param_values=[0.001, 0.01, 0.1],
    resolution={"x": 128},
)
visualize_parameter_effect(results)
```

## Next Steps

- See [Available Models](../guide/models.md) for all supported PDEs
- Learn about [Parameter Exploration](../guide/exploration.md)
- Check [Performance Tips](../advanced/performance.md) for large datasets
