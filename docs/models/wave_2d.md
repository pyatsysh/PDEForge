# Wave 2D (`wave_2d`)

Two-dimensional wave propagation on the periodic square. Disturbances spread
as expanding rings, wrap around the box and interfere with themselves, so the
target field is oscillatory at every scale the initial condition carried.

<figure class="pf-model-fig" markdown>
![Wave 2D](../figures/model_wave_2d.png)
<figcaption>The field at t = 0 and t = T on one colour scale (<code>wave_2d</code>): amplitude is conserved, the pattern is rearranged.</figcaption>
</figure>

## Equation

$$\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u
= c^2\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)$$

with periodic boundary conditions and zero initial velocity.

## Operator learning task

$$u(x, y, 0) \mapsto u(x, y, T)$$

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `wave_speed` | 1.0 | (0.1, 10.0) | Propagation speed $c$ |
| `time_end` | 2.0 | (0.1, 20.0) | Final time $T$ |

## Usage

```python
from pdeforge import generate_dataset

dataset = generate_dataset(
    model="wave_2d",
    n_samples=1000,
    resolution={"x": 64, "y": 64},
    params={"wave_speed": 1.0, "time_end": 2.0},
    seed=42,
)
```

## Solver

Exact, with no time stepping at all. With zero initial velocity the solution
in Fourier space is

$$\hat{u}(\mathbf{k}, t) = \hat{u}(\mathbf{k}, 0)\,\cos(c\,|\mathbf{k}|\,t),$$

applied in one shot at $t = T$; a trajectory costs one transform per frame.
There is no dispersion error and no dissipation, whatever the horizon. That
exactness is what makes this model the reference for
[`heterogeneous_wave_2d`](heterogeneous_wave_2d.md), whose leapfrog scheme is
validated against this propagator in the constant-$c$ limit.

## Behaviour

Because energy is conserved and the domain is periodic, there is no long-time
limit to relax into: the field keeps redistributing itself. That makes horizon
length a genuine difficulty knob rather than a smoothing knob, since a longer
run means more wraparound interference and a target less visibly related to
its input.

## Data shapes

```python
dataset.inputs.shape   # (n_samples, ny, nx)
dataset.outputs.shape  # (n_samples, ny, nx)
```

## Related

- [`wave_1d`](wave_1d.md): the one-dimensional version.
- [`heterogeneous_wave_2d`](heterogeneous_wave_2d.md): a spatially varying
  wave speed, with the medium as the operator's input.
- [`helmholtz_2d`](helmholtz_2d.md): the same physics in the frequency domain.
