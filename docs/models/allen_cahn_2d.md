# Allen-Cahn 2D (`allen_cahn_2d`)

Phase separation on the square. In two dimensions the interfaces are curves,
and curves move under their own curvature: droplets shrink and vanish, and
domains coarsen with a growing characteristic scale. The operator task inherits
that geometry, since predicting the field at time $T$ means predicting which
droplets survived.

<figure class="pf-model-fig" markdown>
![Allen-Cahn 2D](../figures/allen_cahn_2d.png)
<figcaption>The coarsened phase field (<code>allen_cahn_2d</code>): domains of the two wells separated by thin interfaces.</figcaption>
</figure>

## Equation

$$\frac{\partial u}{\partial t} = \varepsilon\,\nabla^2 u + u - u^3$$

with periodic boundary conditions.

## Operator learning task

$$u(x, y, 0) \mapsto u(x, y, T)$$

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `epsilon` | 0.01 | (0.001, 0.5) | Interface width |
| `time_end` | 10.0 | (0.1, 100.0) | Final time; longer gives coarser domains |

## Usage

```python
from pdeforge import generate_dataset

dataset = generate_dataset(
    model="allen_cahn_2d",
    n_samples=500,
    resolution={"x": 64, "y": 64},
    params={"epsilon": 0.01, "time_end": 10.0},
    seed=42,
)
```

## Solver

Semi-implicit IMEX stepping on the spectral grid: the cubic reaction is
stepped explicitly, then diffusion is applied implicitly in Fourier space,
where the update is a diagonal division. The splitting is first order in
time, and the coarsening dynamics it is used for are slow enough that the
splitting error is not the binding constraint. Note this differs from
[`allen_cahn_1d`](allen_cahn_1d.md) and [`allen_cahn_3d`](allen_cahn_3d.md),
which ride the ETDRK4 seam.

## Behaviour

Coarsening is a slow, self-similar process: the mean domain size grows roughly
as $\sqrt{t}$ under curvature flow, so equal increments of `time_end` buy
progressively less change. A dataset that varies the horizon linearly will
therefore sample the early dynamics far more densely than the late ones.

The volume fraction is **not** conserved here, which is the substantive
difference from [`cahn_hilliard`](cahn_hilliard.md): a domain that shrinks
away is gone, and the whole field can end up in one well.

## Data shapes

```python
dataset.inputs.shape   # (n_samples, ny, nx)
dataset.outputs.shape  # (n_samples, ny, nx)
```

## Related

- [`allen_cahn_1d`](allen_cahn_1d.md), [`allen_cahn_3d`](allen_cahn_3d.md):
  the other dimensions.
- [`cahn_hilliard`](cahn_hilliard.md): conserved dynamics, giving
  interconnected morphologies instead.
- [`stochastic_allen_cahn_2d`](stochastic_allen_cahn_2d.md): the same model
  with space-time noise and an ensemble per input.
