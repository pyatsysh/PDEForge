# Advection 1D (`advection_1d`)

Linear advection is the exactly solvable anchor of the catalogue: the solution
is the initial condition translated by $cT$, nothing else happens to it, and a
band-limited spectral scheme reproduces that translation to machine precision.
Its value is diagnostic. A pipeline that gets this wrong has a bug somewhere
that no amount of tuning on a harder model will explain, and any change to
PDEForge's own solver seam is checked against it first.

<figure class="pf-model-fig" markdown>
![Advection 1D](../figures/model_advection_1d.png)
<figcaption>Space-time diagram of u(x, t) (<code>advection_1d</code>): a sharp-edged profile crosses the periodic box repeatedly with its shape intact.</figcaption>
</figure>

## Equation

$$\frac{\partial u}{\partial t} + c\,\frac{\partial u}{\partial x} = 0$$

on the periodic domain, with exact solution $u(x, T) = u_0(x - cT)$.

## Operator learning task

$$u(x, 0) \mapsto u(x, T) = u_0(x - cT)$$

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `speed` | 1.0 | (-100, 100) | Advection speed $c$; negative reverses the direction |
| `time_end` | 0.5 | (0.01, 100.0) | Final time $T$ |

## Usage

```python
from pdeforge import generate_dataset

dataset = generate_dataset(
    model="advection_1d",
    n_samples=1000,
    resolution={"x": 256},
    params={"speed": 1.0, "time_end": 0.5},
    seed=42,
)
```

## Solver

Purely linear on the spectral seam: the propagator $e^{-ickt}$ is applied
exactly, so the discrete solution **is** the band-limited translation of the
initial condition. There is no dissipation and no dispersion to accumulate,
whatever the horizon.

## Why it is worth generating

Two properties make advection a better test than its triviality suggests.
Translation preserves the whole energy spectrum, so a model that quietly
damps high wavenumbers is exposed immediately rather than flattered by a
diffusive target. And because the exact answer is available in closed form,
the error you measure is the operator's error with no solver error mixed in,
which is rarely true anywhere else in the catalogue.

## Data shapes

```python
dataset.inputs.shape   # (n_samples, nx)
dataset.outputs.shape  # (n_samples, nx)
```

## Related

- [`heat_1d`](heat_1d.md): the other exactly propagated linear model, testing
  amplitude decay where this one tests phase.
- [`burgers_1d`](burgers_1d.md): the same advection with the speed set by the
  solution itself, which is where shocks come from.
- [`wave_1d`](wave_1d.md): two characteristics instead of one.
