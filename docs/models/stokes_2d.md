# Stokes 2D

The Stokes equations model incompressible viscous flow at low Reynolds number.

## Equations

$$-\mu \nabla^2 \mathbf{u} + \nabla p = \mathbf{f}$$
$$\nabla \cdot \mathbf{u} = 0$$

with periodic boundary conditions.

## Operator Learning Task

Map body force to velocity and pressure:

$$(f_x, f_y) \mapsto (u, v, p)$$

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `viscosity` | 1.0 | (0.01, 100.0) | Dynamic viscosity μ |
| `n_force_modes` | 5 | (1, 20) | Fourier modes in forcing |

## Usage

```python
from pdeforge import generate_dataset

dataset = generate_dataset(
    model="stokes_2d",
    n_samples=1000,
    resolution={"x": 64, "y": 64},
    params={
        "viscosity": 1.0,
        "n_force_modes": 5,
    },
    seed=42,
)
```

## Solver

FFT-based spectral method. The incompressibility constraint is enforced through projection in Fourier space.

## Input Generation

Random divergence-free body forces generated via:

$$\mathbf{f} = \nabla \times \psi$$

where $\psi$ is a scalar stream function composed of random Fourier modes.

## Physical Behavior

- **Higher viscosity**: Smoother velocity fields
- **More force modes**: More complex flow patterns
- **Incompressibility**: Velocity field is divergence-free

## Data Shapes

```python
dataset.inputs.shape   # (n_samples, nx, ny, 2)  # fx, fy
dataset.outputs.shape  # (n_samples, nx, ny, 3)  # u, v, p
```
