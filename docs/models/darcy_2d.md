# Darcy 2D

The Darcy equation models steady-state flow through porous media, relating permeability to pressure.

## Equation

$$-\nabla \cdot (\kappa(x,y) \nabla u) = f$$

with periodic boundary conditions.

## Operator Learning Task

Map permeability field to pressure field:

$$\kappa(x,y) \mapsto u(x,y)$$

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `kappa_min` | 0.1 | (1e-3, 10.0) | Minimum permeability |
| `kappa_max` | 10.0 | (1.0, 100.0) | Maximum permeability |
| `source_type` | "sine" | "sine", "constant" | Type of forcing function f |

## Usage

```python
from pdeforge import generate_dataset

dataset = generate_dataset(
    model="darcy_2d",
    n_samples=1000,
    resolution={"x": 64, "y": 64},
    params={
        "kappa_min": 0.1,
        "kappa_max": 10.0,
    },
    seed=42,
)
```

## Solver

FFT-based spectral method with conjugate gradient iteration for the variable-coefficient problem.

## Input Generation

Permeability fields are generated using:

1. Gaussian random field as base
2. Sigmoid transform to map to $[\kappa_{\min}, \kappa_{\max}]$

This produces physically plausible heterogeneous media.

## Physical Behavior

- **High contrast** ($\kappa_{\max}/\kappa_{\min}$ large): Sharp pressure gradients at interfaces
- **Smooth permeability**: Smooth pressure fields
- **Channelized structures**: Flow concentrates in high-permeability regions

## Data Shapes

```python
dataset.inputs.shape   # (n_samples, nx, ny)
dataset.outputs.shape  # (n_samples, nx, ny)
```
