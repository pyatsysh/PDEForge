# Available Models

PDEForge provides implementations of several PDE problems commonly used in operator learning research.

## Spectral Models

These models use FFT-based solvers and work with the base installation.

### Burgers 1D (`burgers_1d`)

The viscous Burgers equation models advection with diffusion:

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$

**Operator learning task**: Initial condition to solution at final time

$$u(x, 0) \mapsto u(x, T)$$

```python
dataset = generate_dataset(
    model="burgers_1d",
    n_samples=1000,
    resolution={"x": 256},
    params={
        "viscosity": 0.01,      # Diffusion coefficient
        "time_horizon": 1.0,    # Final time T
    },
)
```

### Darcy 2D (`darcy_2d`)

Steady-state flow through porous media:

$$-\nabla \cdot (\kappa(x,y) \nabla u) = f$$

**Operator learning task**: Permeability field to pressure field

$$\kappa(x,y) \mapsto u(x,y)$$

```python
dataset = generate_dataset(
    model="darcy_2d",
    n_samples=1000,
    resolution={"x": 64, "y": 64},
    params={
        "kappa_min": 0.1,    # Minimum permeability
        "kappa_max": 10.0,   # Maximum permeability
    },
)
```

### Stokes 2D (`stokes_2d`)

Incompressible viscous flow at low Reynolds number:

$$-\mu \nabla^2 \mathbf{u} + \nabla p = \mathbf{f}, \quad \nabla \cdot \mathbf{u} = 0$$

**Operator learning task**: Body force to velocity and pressure fields

$$(f_x, f_y) \mapsto (u, v, p)$$

```python
dataset = generate_dataset(
    model="stokes_2d",
    n_samples=1000,
    resolution={"x": 64, "y": 64},
    params={
        "viscosity": 1.0,
        "n_force_modes": 5,   # Fourier modes in forcing
    },
)
```

## FEniCSx Models

These require FEniCSx installation. See [FEniCSx Setup](../getting-started/fenicsx.md).

### Cylinder Flow 2D (`cylinder_flow_2d`)

Steady Navier-Stokes flow around a circular cylinder:

$$\rho (\mathbf{u} \cdot \nabla) \mathbf{u} - \mu \nabla^2 \mathbf{u} + \nabla p = 0$$
$$\nabla \cdot \mathbf{u} = 0$$

**Operator learning task**: Inlet velocity scale to flow field

$$\text{inlet scale} \mapsto (u, v, p)$$

```python
dataset = generate_dataset(
    model="cylinder_flow_2d",
    n_samples=100,
    resolution={"x": 128, "y": 64},
    params={
        "viscosity": 0.001,
        "inlet_velocity": 0.3,
        "cylinder_radius": 0.05,
    },
)
```

### Cylinder Flow 2D Unsteady (`cylinder_flow_2d_unsteady`)

Time-dependent flow exhibiting vortex shedding (von Karman vortex street):

**Operator learning task**: Inlet velocity to time-dependent flow trajectory

```python
dataset = generate_dataset(
    model="cylinder_flow_2d_unsteady",
    n_samples=10,
    resolution={"x": 110, "y": 41},
    params={
        "inlet_velocity": 1.0,
        "time_end": 8.0,
        "_n_time_steps": 81,
    },
)
# Output shape: (10, 81, 41, 110, 3)
```

### Elasticity 2D (`elasticity_2d`)

Plane-strain linear elasticity with random stiff inclusions; the operator
task maps the Young's-modulus field to displacement and von Mises stress.
Validated by the Clapeyron energy balance at solver precision.

```python
dataset = generate_dataset(
    model="elasticity_2d",
    n_samples=100,
    resolution={"x": 64, "y": 64},
    params={"e_inclusion": 10.0, "traction_y": -1.0},
)
```

### Rayleigh-Benard 2D (`rayleigh_benard_2d`)

Boussinesq convection in the closed unit cavity (hot bottom, cold top,
no-slip walls). Below Ra_c the solver returns the conduction state with
Nu = 1; above it, convection rolls whose selection depends on the seeded
perturbation. Plate-averaged Nusselt numbers are stored per solve.

```python
dataset = generate_dataset(
    model="rayleigh_benard_2d",
    n_samples=20,
    resolution={"x": 64, "y": 64},
    params={"rayleigh": 1e4, "prandtl": 0.71},
)
```

### Porous Darcy FEM (`porous_darcy_fem`)

The cross-model pipeline: a seeded Cahn-Hilliard run grows a two-phase
morphology, binarized into a permeability field; steady Darcy flow crosses
it under a unit pressure drop. Effective permeability is stored per
sample; flux balance and the pressure maximum principle are validated.

```python
dataset = generate_dataset(
    model="porous_darcy_fem",
    n_samples=100,
    resolution={"x": 64, "y": 64},
    params={"ch_time": 8.0, "permeability_contrast": 1e3},
)
```

## Model Information

Use `describe_model` to see configurable parameters:

```python
from pdeforge import describe_model
print(describe_model("burgers_1d"))
```

This shows:

- Physical parameters you can modify
- Default values and valid ranges
- Input/output field names
- Backend used (spectral or FEniCSx)
