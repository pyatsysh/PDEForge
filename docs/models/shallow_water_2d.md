# Shallow water 2D (`shallow_water_2d`)

Gravity waves over a mean depth, in conservative flux form. Three coupled
fields evolve together, and the coupling is what distinguishes this from the
scalar models: an operator has to respect the relationship between the height
field and the two momenta, which per-channel accuracy does not guarantee.

<figure class="pf-model-fig" markdown>
![Shallow water 2D](../figures/model_shallow_water_2d.png)
<figcaption>Surface height at t = 0 and t = T on one colour scale (<code>shallow_water_2d</code>): the initial disturbance has spread into a criss-crossing wave field.</figcaption>
</figure>

## Equation

$$\frac{\partial h}{\partial t} + \frac{\partial (hu)}{\partial x} + \frac{\partial (hv)}{\partial y} = 0$$

$$\frac{\partial (hu)}{\partial t}
+ \frac{\partial}{\partial x}\!\left(hu^2 + \tfrac{1}{2} g h^2\right)
+ \frac{\partial (huv)}{\partial y} = -k_4\,\nabla^4 (hu)$$

$$\frac{\partial (hv)}{\partial t}
+ \frac{\partial (huv)}{\partial x}
+ \frac{\partial}{\partial y}\!\left(hv^2 + \tfrac{1}{2} g h^2\right) = -k_4\,\nabla^4 (hv)$$

on the periodic box.

## Operator learning task

$$(h, hu, hv)(x, y, 0) \mapsto (h, hu, hv)(x, y, T)$$

with the components stacked, shape `(3, ny, nx)`.

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `gravity` | 9.81 | (0.1, 100.0) | Gravitational acceleration $g$ |
| `mean_depth` | 1.0 | (0.01, 100.0) | Mean water depth $H$ |
| `time_end` | 0.2 | (0.001, 10.0) | Final time $T$ |
| `hyperviscosity` | 1e-7 | (0.0, 1e-3) | Spectral filter strength $k_4$, for stability |

Wave speed is $\sqrt{gH}$, so `gravity` and `mean_depth` together set how far
information travels in the horizon. The default combination gives
$\sqrt{9.81} \approx 3.1$, which crosses the unit box in about a third of a
time unit.

## Usage

```python
from pdeforge import generate_dataset

dataset = generate_dataset(
    model="shallow_water_2d",
    n_samples=500,
    resolution={"x": 128, "y": 128},
    params={"gravity": 9.81, "mean_depth": 1.0, "time_end": 0.2},
    seed=42,
)
```

## Solver

Pseudo-spectral, with the small hyperviscous filter forming the diagonal
linear part and integrated exactly, and the flux divergences fully explicit
and dealiased. Mass, meaning the $k = 0$ mode of $h$, is conserved to machine
precision, because the flux divergence has exactly zero mean in Fourier space.
That conservation is the model's validation invariant.

## Behaviour

The hyperviscosity is a numerical stabiliser rather than physics, and it is
worth knowing where it acts: $\nabla^4$ damps the highest wavenumbers hard and
leaves the resolved scales essentially untouched. Raising it to smooth away an
instability will also quietly smooth the data.

Long horizons produce criss-crossing wave fields as disturbances wrap the
periodic box and interfere, and there is no steady state to relax into.

## Data shapes

```python
dataset.inputs.shape   # (n_samples, 3, ny, nx)
dataset.outputs.shape  # (n_samples, 3, ny, nx)
```

## Related

- [`burgers_2d`](burgers_2d.md): the other multi-component hyperbolic system,
  without the height coupling.
- [`wave_2d`](wave_2d.md): the linear small-amplitude limit.
- [`rayleigh_benard_2d`](rayleigh_benard_2d.md): buoyancy-driven flow, in a
  closed cavity rather than on a periodic box.
