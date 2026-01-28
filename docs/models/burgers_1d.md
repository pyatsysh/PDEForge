# Burgers 1D

The viscous Burgers equation models nonlinear advection with diffusion, commonly used as a simplified model for shock formation.

## Equation

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$

with periodic boundary conditions on $[0, 2\pi]$.

## Operator Learning Task

Map initial condition to solution at final time:

$$u(x, t=0) \mapsto u(x, t=T)$$

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `viscosity` | 0.01/π | (1e-6, 1.0) | Diffusion coefficient ν |
| `time_horizon` | 1.0 | (0.1, 10.0) | Final time T |

Lower viscosity produces sharper shocks. The advection coefficient is fixed at 1.0.

## Usage

```python
from pdeforge import generate_dataset

dataset = generate_dataset(
    model="burgers_1d",
    n_samples=1000,
    resolution={"x": 256},
    params={
        "viscosity": 0.01,
        "time_horizon": 1.0,
    },
    seed=42,
)
```

## Solver

FFT-based pseudo-spectral method with `scipy.integrate.odeint` for time stepping.

## Initial Conditions

Default generator: Fourier series with random coefficients

$$u_0(x) = \sum_{k=1}^{N} a_k \sin(kx)$$

where coefficients $a_k$ decay as $k^{-\alpha}$.

## Physical Behavior

- **Low viscosity**: Solutions develop sharp shock fronts
- **High viscosity**: Solutions remain smooth
- **Long time horizon**: More nonlinear evolution, potential for shocks to interact

## Data Shapes

```python
dataset.inputs.shape   # (n_samples, nx)
dataset.outputs.shape  # (n_samples, nx)
```
