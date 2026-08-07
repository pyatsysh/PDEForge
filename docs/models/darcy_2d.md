# Darcy 2D (`darcy_2d`)

Steady flow through a porous medium, on the periodic box with a spectral
solver. The task is the classic one: hand the operator a permeability field
and ask for the pressure it induces. Where
[`darcy_fno_2d`](darcy_fno_2d.md) reproduces the published benchmark down to
its grid convention, `darcy_2d` is the freer periodic version, useful when you
want the physics without inheriting somebody else's measure.

<figure class="pf-model-fig" markdown>
![Darcy 2D](../figures/darcy_2d.png)
<figcaption>Pressure over a heterogeneous permeability field (<code>darcy_2d</code>): flow concentrates wherever the medium lets it.</figcaption>
</figure>

## Equation

$$-\nabla \cdot \big(\kappa(x,y)\,\nabla u\big) = f$$

with periodic boundary conditions.

## Operator learning task

$$\kappa(x, y) \mapsto u(x, y)$$

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `kappa_min` | 0.1 | (1e-3, 10.0) | Lower end of the permeability range |
| `kappa_max` | 10.0 | (1.0, 100.0) | Upper end of the permeability range |
| `source_type` | `"sine"` | `"sine"`, `"constant"` | Forcing $f$ |

The contrast ratio $\kappa_{\max}/\kappa_{\min}$ is the parameter that
actually sets the difficulty. At the defaults it is 100, which already puts
sharp gradients at the interfaces.

!!! note "`describe_model` will not list these"
    They are read from the model's defaults dictionary rather than declared as
    tunable specs, so `describe_model("darcy_2d")` prints
    `Parameters: See DEFAULT_PARAMS` instead of the table above. Passing them
    through `params=` works as documented.

## Usage

```python
from pdeforge import generate_dataset

dataset = generate_dataset(
    model="darcy_2d",
    n_samples=1000,
    resolution={"x": 64, "y": 64},
    params={"kappa_min": 0.1, "kappa_max": 10.0},
    seed=42,
)
```

## Solver

Spectral discretisation with conjugate-gradient iteration for the
variable-coefficient operator. The periodic setting is what makes the FFT
usable here; it is also what distinguishes this model from the Dirichlet
benchmark family.

## The input measure

Permeability fields come from a Gaussian random field pushed through a
sigmoid, which maps the unbounded Gaussian onto
$[\kappa_{\min}, \kappa_{\max}]$ and keeps the coefficient strictly positive
so the operator stays elliptic. The result reads as smoothly channelised
media rather than the two-phase blocks of the piecewise-constant benchmark.

## Behaviour

High contrast puts steep pressure gradients along the permeability interfaces,
and those interfaces are where a band-limited operator loses accuracy first;
smooth permeability gives smooth pressure and a correspondingly easy problem.
Wherever the sigmoid produces channelised structures the flow concentrates into
them, so a small area of the domain carries most of the signal and an averaged
error metric will under-report what went wrong.

## Data shapes

```python
dataset.inputs.shape   # (n_samples, ny, nx)
dataset.outputs.shape  # (n_samples, ny, nx)
```

## Related

- [`darcy_fno_2d`](darcy_fno_2d.md): the canonical benchmark, bit-exact
  against the distributed data.
- [`porous_darcy_fem`](porous_darcy_fem.md): Darcy flow across a
  Cahn-Hilliard microstructure, solved by finite elements.
- [`helmholtz_2d`](helmholtz_2d.md): the other steady spectral problem.
