# Wave 1D (`wave_1d`)

The wave equation is the non-dissipative counterweight to
[`heat_1d`](heat_1d.md). Nothing decays: energy is conserved, every mode keeps
its amplitude and only turns its phase, and the solution at time $T$ carries
exactly as much fine structure as the initial condition did. Operators trained
against diffusive targets often have a quiet low-pass bias, and this is the
model that finds it.

<figure class="pf-model-fig" markdown>
![Wave 1D](../figures/model_wave_1d.png)
<figcaption>Space-time diagram of u(x, t) (<code>wave_1d</code>): a single localised bump splits into left- and right-going characteristics that pass through each other.</figcaption>
</figure>

## Equation

$$\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}$$

integrated as the first-order system

$$\frac{\partial u}{\partial t} = v, \qquad
\frac{\partial v}{\partial t} = c^2 \frac{\partial^2 u}{\partial x^2}$$

with periodic boundary conditions. The initial velocity is zero, so the
initial displacement splits evenly into two counter-propagating halves.

## Operator learning task

$$u(x, 0) \mapsto u(x, T)$$

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `wave_speed` | 1.0 | (0.1, 10.0) | Propagation speed $c$ |
| `time_end` | 2.0 | (0.1, 20.0) | Final time $T$ |

As with advection, only $cT$ matters for where the waves end up; the two
parameters are one knob wearing two labels.

## Usage

```python
from pdeforge import generate_dataset

dataset = generate_dataset(
    model="wave_1d",
    n_samples=1000,
    resolution={"x": 256},
    params={"wave_speed": 1.0, "time_end": 2.0},
    seed=42,
)
```

## Solver

Pseudo-spectral in space, with the first-order system integrated in time by
SciPy's adaptive `odeint`. Energy is therefore conserved to the integrator's
tolerance rather than exactly, so on long horizons check the energy drift
before trusting the tail of a trajectory. [`wave_2d`](wave_2d.md) applies the
exact spectral propagator instead, with no time-stepping error at all.

## Behaviour

On the periodic box the two halves wrap around and re-collide, and because the
equation is linear they pass straight through each other with no interaction.
Long horizons therefore do not simplify the problem the way they do for the
heat equation; they only rearrange it.

## Data shapes

```python
dataset.inputs.shape   # (n_samples, nx)
dataset.outputs.shape  # (n_samples, nx)
```

## Related

- [`wave_2d`](wave_2d.md): the same model on the square.
- [`heterogeneous_wave_2d`](heterogeneous_wave_2d.md): the medium becomes the
  input, which turns this into an inverse-problem-shaped task.
- [`advection_1d`](advection_1d.md): one characteristic instead of two.
