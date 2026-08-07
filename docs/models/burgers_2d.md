# Burgers 2D (`burgers_2d`)

The vector generalisation of Burgers: a two-component velocity advecting
itself, with no pressure and no incompressibility constraint. That omission is
the point. Sharp fronts form and diffuse in two dimensions without the elliptic
coupling that makes Navier-Stokes global, so this sits between
[`burgers_1d`](burgers_1d.md) and [`ns_vorticity_2d`](ns_vorticity_2d.md) as a
genuinely two-dimensional problem that stays local.

<figure class="pf-model-fig" markdown>
![Burgers 2D](../figures/model_burgers_2d.png)
<figcaption>Speed at t = 0 and t = T on one colour scale (<code>burgers_2d</code>): the smooth initial field has collapsed onto fronts.</figcaption>
</figure>

## Equation

$$\frac{\partial u}{\partial t} + u\,\frac{\partial u}{\partial x} + v\,\frac{\partial u}{\partial y}
= \nu\,\nabla^2 u$$

$$\frac{\partial v}{\partial t} + u\,\frac{\partial v}{\partial x} + v\,\frac{\partial v}{\partial y}
= \nu\,\nabla^2 v$$

on the periodic box.

## Operator learning task

$$(u, v)(x, y, 0) \mapsto (u, v)(x, y, T)$$

with the components stacked on a leading axis, shape `(2, ny, nx)`.

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `viscosity` | 0.01 | (1e-5, 1.0) | Viscosity $\nu$; lower gives sharper fronts |
| `time_horizon` | 1.0 | (0.05, 10.0) | Final time $T$ |

## Usage

```python
from pdeforge import generate_dataset

dataset = generate_dataset(
    model="burgers_2d",
    n_samples=1000,
    resolution={"x": 128, "y": 128},
    params={"viscosity": 0.01, "time_horizon": 1.0},
    seed=42,
)
```

## Solver

ETDRK4 on the spectral seam: viscous diffusion exact, dealiased self-advection
explicit for both components. Same machinery as the one-dimensional model,
applied twice.

## Behaviour

Two-dimensional fronts are curves rather than points, so the region a
low-viscosity solution puts its error in scales as $\sqrt{\nu}$ in width but
spans the domain in length. The practical effect is that the difficult set is
a much larger share of the field than in one dimension at the same viscosity,
and resolution requirements bite sooner.

## Data shapes

```python
dataset.inputs.shape   # (n_samples, 2, ny, nx)
dataset.outputs.shape  # (n_samples, 2, ny, nx)
```

## Related

- [`burgers_1d`](burgers_1d.md): the scalar version, with published presets.
- [`ns_vorticity_2d`](ns_vorticity_2d.md): add incompressibility and the
  dynamics become global.
- [`shallow_water_2d`](shallow_water_2d.md): the other multi-component
  hyperbolic system.
