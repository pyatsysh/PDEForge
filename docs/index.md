# PDEForge

**A unified framework for generating PDE datasets for operator learning.**

PDEForge provides a consistent interface to generate training data for neural operators from various partial differential equations. Generate datasets at any resolution, explore parameter effects interactively, and use built-in support for uncertainty quantification workflows.

## Key Features

- **Unified API**: Same interface for all PDE models, from 1D Burgers to 2D flow around obstacles
- **Configurable Resolution**: Generate data at any spatial resolution you need
- **Parameter Exploration**: Visualize how physical parameters affect solutions before committing to large datasets
- **UQ-Ready**: Dedicated calibration splits for conformal prediction and other uncertainty methods
- **Extensible**: Add custom models through a simple registry pattern

## Quick Example

```python
from pdeforge import generate_dataset, list_models

# See available models
print(list_models())
# ['burgers_1d', 'darcy_2d', 'stokes_2d', 'cylinder_flow_2d', ...]

# Generate a dataset
dataset = generate_dataset(
    model="burgers_1d",
    n_samples=1000,
    resolution={"x": 256},
    params={"viscosity": 0.01},
    seed=42,
)

# Split for ML workflow with calibration set
splits = dataset.split(train=0.6, val=0.15, cal=0.15, test=0.1)
```

## Comparison with Existing Solutions

| Feature | Static Datasets | PDEForge |
|---------|-----------------|----------|
| Resolution | Fixed | Configurable |
| API | Different per dataset | Unified |
| Parameters | Pre-set | Configurable |
| Exploration | Download first | Interactive preview |
| UQ support | No calibration split | Built-in |

## Installation

```bash
pip install pdeforge
```

For FEniCSx-based models (complex geometries):

```bash
conda install -c conda-forge fenics-dolfinx
pip install pdeforge
```

## Documentation

- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quickstart.md)
- [Available Models](guide/models.md)
- [API Reference](api/reference.md)

## License

MIT License. See [LICENSE](https://github.com/pyatsysh/pdeforge/blob/main/LICENSE) for details.
