# Canonical Darcy 3D (`darcy_fno_3d`)

The canonical Darcy benchmark extended to the unit cube, which is the knob no
frozen dataset offers. The two-dimensional [`darcy_fno_2d`](darcy_fno_2d.md) is
validated bit-for-bit against the distributed FNO data; this model carries the
**same** input measure and the same coefficient pushforwards onto three
dimensions. No canonical 3D Darcy file exists to download anywhere, so
dimension itself is what varies here.

<figure class="pf-model-fig" markdown>
![Darcy 3D](../figures/darcy3d_hero.png)
<figcaption>Pressure isosurfaces on the unit cube (<code>darcy_fno_3d</code>): <code>tau</code> sets the coefficient correlation length.</figcaption>
</figure>

## Equation

$$-\nabla \cdot \big(a(x, y, z)\,\nabla u\big) = f
\quad \text{on } [0,1]^3, \qquad u = 0 \text{ on the boundary}, \qquad f = 1.$$

## Operator learning task

$$a(x, y, z) \mapsto u(x, y, z)$$

## The input measure

The same cosine-KL Gaussian as in two dimensions,

$$\psi \sim N\big(0,\; (-\Delta + \tau^2 I)^{-\alpha}\big),$$

which is trace-class in three dimensions when $2\alpha > 3$. The canonical
$\alpha = 2$ clears that comfortably; the parameter bounds start at 1.6, so
the lower end of the range is close to where the measure stops being
well defined and fields become rough enough that the discretisation, rather
than the measure, decides what you get.

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `coeff` | `"lognormal"` | | `lognormal` or `piececonst` |
| `alpha` | 2.0 | (1.6, 6.0) | GRF spectral decay; trace-class in 3D needs $\alpha > 1.5$ |
| `tau` | 3.0 | (0.5, 30.0) | GRF inverse correlation length |
| `sigma` | `None` | | `None` gives the canonical $\tau^{\alpha-1}$; a number overrides |
| `kappa_plus` / `kappa_minus` | 12.0 / 3.0 | | Two-phase permeabilities |
| `forcing` | 1.0 | | Constant source $f$ |

## Usage

```python
from pdeforge import generate_dataset

dataset = generate_dataset(
    model="darcy_fno_3d",
    n_samples=200,
    resolution={"x": 49, "y": 49, "z": 49},
    params={"alpha": 2.0, "tau": 3.0},
    seed=1,
    to="darcy3d.h5",     # chunked to disk
)
```

## Solver

Seven-point finite differences on the boundary-inclusive grid. The matrix is
symmetric positive definite, and two solve paths are used depending on size:
direct sparse LU on small grids, and Jacobi-preconditioned conjugate gradients
on larger ones, because 3D LU fill-in is prohibitive. Both paths agree to
solver tolerance, which the test suite checks.

## Behaviour

Cost is the governing consideration. A $49^3$ grid has 117,649 unknowns, and
each factor of two in resolution multiplies that by eight, so the crossover
where iterative solving becomes the only option arrives quickly. Storage
follows the same curve: pair large runs with `to=` and chunked generation.

## Data shapes

```python
dataset.inputs.shape   # (n_samples, nz, ny, nx)
dataset.outputs.shape  # (n_samples, nz, ny, nx)
```

## Related

- [`darcy_fno_2d`](darcy_fno_2d.md): the canonical benchmark, bit-exact
  against the published data.
- [`darcy_2d`](darcy_2d.md): the periodic spectral Darcy, a different problem.
- [`heat_3d`](heat_3d.md) and [`allen_cahn_3d`](allen_cahn_3d.md): the other
  volumetric models.
