# Parameter Exploration

Understanding how physical parameters affect PDE solutions is important before generating large training datasets. PDEForge provides tools for systematic parameter exploration.

## Single Parameter Exploration

Vary one parameter while keeping others fixed:

```python
from pdeforge import explore_parameter, visualize_parameter_effect

results = explore_parameter(
    model="burgers_1d",
    param_name="viscosity",
    param_values=[0.001, 0.01, 0.1],
    resolution={"x": 128},
    n_samples_per_value=5,
    seed=42,
)

# Visualize in Jupyter
visualize_parameter_effect(results)
```

The same initial condition is used across parameter values, making it easy to see the isolated effect of changing that parameter.

## Multi-Parameter Grid

Explore combinations of parameters:

```python
from pdeforge import explore_parameter_grid

results = explore_parameter_grid(
    model="darcy_2d",
    param_grid={
        "kappa_min": [0.01, 0.1, 1.0],
        "kappa_max": [5.0, 10.0, 50.0],
    },
    resolution={"x": 32, "y": 32},
    n_samples_per_combo=3,
)
```

## Exploring All Parameters

Get a quick overview of all configurable parameters:

```python
from pdeforge import explore_model

explorations = explore_model(
    model="burgers_1d",
    resolution={"x": 128},
)

for param_name, dataset in explorations.items():
    print(f"Effect of {param_name}:")
    visualize_parameter_effect(dataset)
```

## Use Cases

### Choosing Training Ranges

Before generating 10,000 samples, explore to understand:

- Which parameter ranges produce interesting behavior
- Which regimes are numerically stable
- Where shocks or discontinuities appear

### Building Physical Intuition

See how viscosity affects shock formation in Burgers, or how permeability contrast affects pressure fields in Darcy flow.

### Identifying Difficult Regimes

Parameters that produce sharp gradients, oscillations, or near-singular behavior may be harder for neural operators to learn. Identifying these regimes early helps in designing training curricula.

## Output Format

`explore_parameter` returns a `PDEDataset` with additional metadata:

```python
results.metadata['explored_param']     # Parameter name
results.metadata['param_values']       # Values tested
results.metadata['samples_per_value']  # Samples at each value
```

The samples are ordered: first `n_samples_per_value` samples use the first parameter value, next batch uses the second value, and so on.
