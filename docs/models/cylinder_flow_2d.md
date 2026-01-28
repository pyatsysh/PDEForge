# Cylinder Flow 2D

Steady-state flow around a circular cylinder, a classic problem in computational fluid dynamics.

## Equations

$$\rho (\mathbf{u} \cdot \nabla) \mathbf{u} - \mu \nabla^2 \mathbf{u} + \nabla p = 0$$
$$\nabla \cdot \mathbf{u} = 0$$

## Boundary Conditions

- **Inlet**: Parabolic velocity profile
- **Outlet**: Zero-stress (do-nothing)
- **Walls**: No-slip
- **Cylinder surface**: No-slip

## Operator Learning Task

Map inlet velocity scale to flow field:

$$\text{inlet scale} \mapsto (u, v, p)$$

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `viscosity` | 0.001 | (1e-5, 0.1) | Dynamic viscosity μ |
| `inlet_velocity` | 0.3 | (0.01, 2.0) | Mean inlet velocity |
| `cylinder_radius` | 0.05 | (0.01, 0.1) | Cylinder radius |

## Usage

Requires FEniCSx. See [FEniCSx Setup](../getting-started/fenicsx.md).

```python
from pdeforge import generate_dataset

dataset = generate_dataset(
    model="cylinder_flow_2d",
    n_samples=50,
    resolution={"x": 128, "y": 64},
    params={
        "viscosity": 0.001,
        "inlet_velocity": 0.3,
    },
    seed=42,
)
```

## Solver

Finite element method using FEniCSx with:

- Taylor-Hood elements (P2-P1 for velocity-pressure)
- Newton solver for nonlinear Navier-Stokes
- gmsh for mesh generation with cylinder hole

## Domain

Default channel: 2.2 x 0.41 with cylinder centered at (0.2, 0.2).

## Physical Behavior

- **Low Reynolds number**: Steady symmetric wake
- **Higher Re**: Asymmetric wake, eventually vortex shedding (requires unsteady model)

## Data Shapes

```python
dataset.inputs.shape   # (n_samples, 1)  # inlet scale
dataset.outputs.shape  # (n_samples, nx, ny, 3)  # u, v, p
```

## Notes

Solutions are interpolated from the unstructured FEM mesh to a regular grid. Points inside the cylinder are filled with zeros. The `domain_mask` in metadata indicates valid points.
