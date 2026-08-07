# Cahn-Hilliard (`cahn_hilliard`)

Spinodal decomposition: the model where the pattern is produced by the
dynamics rather than supplied in the initial condition. Every sample starts
from a near-uniform mixture plus small white noise, the spinodal instability
amplifies a band of wavenumbers around $\lambda^* \approx 2\pi\sqrt{2}\,
\varepsilon$, and the conserved nonlinear dynamics sharpen that band into the
labyrinthine or droplet morphology the equation is known for.

The conservation is the substantive difference from
[`allen_cahn_2d`](allen_cahn_2d.md). Here the spatial mean of $u$ is preserved
exactly, so `mean_composition` is a genuine morphology control instead of a
transient.

<figure class="pf-model-fig" markdown>
![Cahn-Hilliard](../figures/spinodal3d.png)
<figcaption>The u = 0 interface in 3D (<code>cahn_hilliard</code>): the same model runs 2D or 3D from the <code>resolution</code> dict.</figcaption>
</figure>

## Equation

$$\frac{\partial u}{\partial t}
= M\,\nabla^2\!\left(u^3 - u - \varepsilon^2 \nabla^2 u\right)$$

with periodic boundary conditions, in two or three dimensions.

## Operator learning task

$$u(\mathbf{x}, 0) \mapsto u(\mathbf{x}, T)$$

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `epsilon` | 0.01 | (0.004, 0.025) | Interface width; sets the pattern scale $\lambda^* \approx 2\pi\sqrt{2}\,\varepsilon$ |
| `mobility` | 1.0 | (0.01, 10.0) | Mobility $M$; higher separates and coarsens faster |
| `mean_composition` | 0.0 | (-0.6, 0.6) | Spatial mean of $u$, conserved exactly |
| `time_end` | 0.1 | (0.001, 10.0) | Final time; coarsening goes as $t^{1/3}$ |
| `binarize` | `False` | | Return hard $\{0, 1\}$ masks instead of the continuous field |

`mean_composition` selects the morphology: 0 gives a bicontinuous labyrinth,
and $|m| \to 0.4$ gives minority-phase droplets. Beyond $\pm 0.577$ the
mixture leaves the spinodal regime altogether and no instability grows.

!!! note "Resolution follows epsilon"
    The pattern scale is proportional to $\varepsilon$, so below about
    $\varepsilon = 0.006$ use a resolution of 256 or more if the interfaces
    are to be resolved rather than sampled.

## Usage

```python
from pdeforge import generate_dataset

# 2D labyrinth
dataset = generate_dataset(
    model="cahn_hilliard",
    n_samples=500,
    resolution={"x": 128, "y": 128},
    params={"epsilon": 0.01, "mean_composition": 0.0, "time_end": 0.1},
    seed=42,
)

# 3D: add a z axis and the same code path runs
volumes = generate_dataset(
    model="cahn_hilliard",
    n_samples=50,
    resolution={"x": 64, "y": 64, "z": 64},
    seed=4,
)

# hard two-phase masks, thresholded at u = 0
masks = generate_dataset(
    model="cahn_hilliard",
    n_samples=500,
    resolution={"x": 128, "y": 128},
    params={"binarize": True},
    seed=42,
)
```

The dimension is inferred from the `resolution` dict: `{"x", "y"}` runs 2D and
`{"x", "y", "z"}` runs 3D, with no other change.

## Solver

Spectral, with the fourth-order linear operator integrated exactly so the time
step is not tied to $\Delta x^4$. Randomised samples come from seeding the
white-noise initial condition; the spinodal appearance is produced by the PDE.

## Behaviour

Coarsening follows the Lifshitz-Slyozov law, with the characteristic domain
size growing as $t^{1/3}$. Two samples at different seeds share their
morphology statistics while differing pointwise everywhere, which makes this
a natural model for asking whether an operator has learned the statistics or
memorised the realisations.

The `binarize=True` output is what [`porous_darcy_fem`](porous_darcy_fem.md)
consumes: thresholding the phase field gives a two-phase medium that a flow
solver can then be run across.

## Data shapes

```python
dataset.inputs.shape   # (n_samples, ny, nx)  or (n_samples, nz, ny, nx)
dataset.outputs.shape  # the same
```

## Related

- [`allen_cahn_2d`](allen_cahn_2d.md): the non-conserved counterpart, where
  domains can vanish entirely.
- [`porous_darcy_fem`](porous_darcy_fem.md): flow through the microstructure
  this model grows.
- [`gray_scott_2d`](gray_scott_2d.md): the other pattern-forming system, with
  patterns selected by kinetics instead of composition.
