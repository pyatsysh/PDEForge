# Cylinder Flow 2D unsteady (`cylinder_flow_2d_unsteady`)

The von Kármán vortex street. Above a critical Reynolds number the steady
symmetric wake of [`cylinder_flow_2d`](cylinder_flow_2d.md) loses stability and
the cylinder starts shedding vortices alternately from each side, at a
well-defined frequency. This model returns the whole trajectory rather than an
endpoint, which makes it the catalogue's reference for autoregressive rollout
work on a wall-bounded flow.

<figure class="pf-model-fig" markdown>
![Cylinder flow unsteady](../figures/cylinder_flow_unsteady_vorticity.png)
<figcaption>Wake vorticity during shedding (<code>cylinder_flow_2d_unsteady</code>): vortices leave the cylinder alternately from each side.</figcaption>
</figure>

## Equations

$$\rho\left(\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u}\cdot\nabla)\mathbf{u}\right)
- \mu\,\nabla^2 \mathbf{u} + \nabla p = 0,
\qquad \nabla \cdot \mathbf{u} = 0$$

with a parabolic inlet profile, optionally ramped, zero stress at the outlet,
and no slip on the walls and the cylinder.

## Operator learning task

$$(\text{inlet velocity},\ \text{initial state})
\mapsto \mathbf{u}(x, t) \ \text{for}\ t \in [0, T]$$

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `inlet_velocity` | 1.0 | (0.1, 3.0) | Mean inlet velocity, which sets Reynolds |
| `viscosity` | 0.001 | (1e-5, 0.01) | Dynamic viscosity; lower gives stronger vortices |
| `time_end` | 8.0 | (1.0, 20.0) | Final time; longer gives more shedding cycles |

## Usage

This model needs FEniCSx. See [FEniCSx setup](../getting-started/fenicsx.md).

```python
from pdeforge import generate_dataset

dataset = generate_dataset(
    model="cylinder_flow_2d_unsteady",
    n_samples=10,
    resolution={"x": 110, "y": 41},
    params={"inlet_velocity": 1.0, "time_end": 8.0, "_n_time_steps": 81},
    seed=42,
)
dataset.outputs.shape   # (10, 81, 41, 110, 3)
```

## Solver

Backward Euler in time, chosen for stability at moderate Reynolds rather than
for accuracy: the scheme is first order and damps, so a coarse time step will
quietly weaken the shedding it is supposed to capture. Space is Taylor-Hood
P2/P1 as in the steady model.

## Behaviour

Shedding needs time to start. The flow begins from rest or from a ramped
inlet, passes through a transient where the wake is still symmetric, and only
then develops the alternating pattern. A trajectory that stops too early is
mostly transient, so `time_end` should cover several shedding periods if the
periodic state is what you want to learn.

The one property to watch is the Strouhal number, meaning the shedding
frequency: it is the physical quantity a rollout is most likely to get subtly
wrong, and a model can look accurate frame by frame while drifting in phase.

## Data shapes

```python
dataset.inputs.shape   # (n_samples, 1)
dataset.outputs.shape  # (n_samples, n_t, ny, nx, 3)   u, v, p per frame
```

## Related

- [`cylinder_flow_2d`](cylinder_flow_2d.md): the steady solve.
- [`cylinder_flow_2d_turbulent`](cylinder_flow_2d_turbulent.md): Re 2000 with
  a large-eddy closure and a per-sample cylinder position.
- [`ns_vorticity_2d`](ns_vorticity_2d.md): unsteady incompressible flow
  without walls.
