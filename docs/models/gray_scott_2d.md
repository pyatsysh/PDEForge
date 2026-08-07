# Gray-Scott 2D (`gray_scott_2d`)

The reaction-diffusion system with a parameter plane worth exploring. Pearson
(1993) mapped feed rate $F$ against kill rate $k$ and found that perturbations
of the trivial state $(U, V) = (1, 0)$ grow into spots, stripes, labyrinths or
self-replicating patterns depending on where in that plane you sit. For
operator learning this is unusual and useful: two nearby parameter values can
give qualitatively different fields, so the map from parameters to solution is
not smooth in any helpful sense.

<figure class="pf-model-fig" markdown>
![Gray-Scott](../figures/gray_scott.png)
<figcaption>Patterns at t = 8000 (<code>gray_scott_2d</code>): every (<code>feed</code>, <code>kill</code>) pair grows a different pattern.</figcaption>
</figure>

## Equation

$$\frac{\partial U}{\partial t} = D_U \nabla^2 U - U V^2 + F\,(1 - U)$$

$$\frac{\partial V}{\partial t} = D_V \nabla^2 V + U V^2 - (F + k)\,V$$

on the periodic box, in Pearson's scaling: domain $[0, 2.5]^2$,
$D_U = 2\times10^{-5}$, $D_V = 10^{-5}$.

## Operator learning task

$$(U, V)(x, y, 0) \mapsto (U, V)(x, y, T)$$

with the two fields stacked on a leading component axis, shape `(2, ny, nx)`.

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `feed` | 0.04 | (0.0, 0.12) | Feed rate $F$ |
| `kill` | 0.06 | (0.0, 0.08) | Kill rate $k$ |
| `Du` | 2e-5 | (1e-6, 1e-3) | Diffusivity of $U$ |
| `Dv` | 1e-5 | (1e-6, 1e-3) | Diffusivity of $V$ |
| `time_end` | 2000.0 | (10.0, 20000.0) | Final time; patterns need $t \sim 1000$ or more |

## Usage

```python
from pdeforge import generate_dataset

dataset = generate_dataset(
    model="gray_scott_2d",
    n_samples=200,
    resolution={"x": 128, "y": 128},
    params={"feed": 0.054, "kill": 0.063, "time_end": 8000.0},
    backend="jax",
    seed=11,
)
```

A few reference points in the Pearson plane: $(0.054, 0.063)$ gives coral-like
growth, $(0.0367, 0.0649)$ gives a space-filling labyrinth, and
$(0.04, 0.06)$, the default, gives mitosis-like self-replicating spots.

## Solver

The two components ride the spectral seam's leading component axis with a
diagonal linear symbol, and the $UV^2$ kinetics is stepped explicitly.

## Behaviour

Patterns take time. Below $t \approx 1000$ the field is still a perturbation of
the trivial state, and a dataset generated at short horizon will mostly record
the initial seeding rather than the pattern. The default `time_end` of 2000 is
the lower end of the useful range.

The initial condition matters more than usual, because the pattern grows out of
where the seeds were placed. The `n_patches` parameter of the generator sets
how many seeds start the growth, and a single seed on a large domain gives a
very different picture from forty.

## Data shapes

```python
dataset.inputs.shape   # (n_samples, 2, ny, nx)
dataset.outputs.shape  # (n_samples, 2, ny, nx)
```

## Related

- [`fitzhugh_nagumo_2d`](fitzhugh_nagumo_2d.md): the other two-component
  excitable system, with travelling fronts instead of stationary patterns.
- [`cahn_hilliard`](cahn_hilliard.md): pattern formation driven by a
  conservation law rather than by kinetics.
- [`lotka_volterra_2d`](lotka_volterra_2d.md): predator-prey kinetics with
  diffusion.
