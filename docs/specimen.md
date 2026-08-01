# Specimen

*Design-system test card: every element the theme must render, on one
page. Not linked from the nav; open at `/specimen/`.*

## Type ladder

Body text in Crimson Text sits under a Fredericka the Great page title.
The ladder below shows the heading cascade.

## Heading two

### Heading three

#### Heading four

## Equations

Inline math sits in running text: the residual $R(x, p) = 0$ is traced
through folds, and the drag coefficient $C_d$ follows from the stress
tensor. Display math gets its own line:

$$
\partial_t u + (u \cdot \nabla)\, u \;=\; -\frac{1}{\rho}\nabla p
\;+\; \nu \nabla^2 u, \qquad \nabla \cdot u = 0
$$

A variational statement, because density functionals are home turf:

$$
\frac{\delta F[\rho]}{\delta \rho(\mathbf{r})} \;=\;
\mu - V_{\mathrm{ext}}(\mathbf{r})
$$

## Code

```python
import pdeforge

data = pdeforge.generate_dataset(
    "ns_vorticity_2d", n_samples=1000,
    resolution={"x": 128, "y": 128}, seed=0,
)
splits = data.split(train=0.6, val=0.15, cal=0.15, test=0.1)
```

Inline code: `pdeforge generate --model darcy_2d`.

## Admonitions

One per hue of the nord admonition palette.

!!! note "A note"
    Calibration splits are first-class: `cal` is reserved for conformal
    methods, never folded into training.

!!! abstract "An abstract"
    The dataset object carries fields, geometry channels, coefficients,
    and full provenance metadata.

!!! tip "A tip"
    The JAX backend runs the spectral models on GPU without any changes
    to the call.

!!! success "A success"
    The Darcy generator reproduces the distributed FNO data bit for bit,
    to within 2 ulp of float32.

!!! question "A question"
    Which resolution does a model support? Any: resolution is a
    parameter, not a property of the stored data.

!!! warning "A warning"
    The Docker image tag `fenicsx` is the FEM build; the default tag has
    no FEniCS.

!!! danger "A danger"
    Deleting the seed from provenance metadata makes the sample
    unreproducible.

!!! example "An example"
    `pdeforge generate --model kolmogorov_flow_2d --n 8 --seed 7`

!!! quote "A quote"
    The traced branch is read as a dynamical object: its spectrum
    classifies folds, branch points and Hopf candidates.

## Figures

A regular grid with one-line captions; letterbox bars in the figures' own
surface color; motion as a looped mp4.

<div class="pf-stills-grid" markdown>
<figure markdown>
![Kolmogorov flow vorticity](figures/kolmogorov_vorticity.png)
<figcaption>Forced 2D turbulence (<code>kolmogorov_flow_2d</code>)</figcaption>
</figure>
<figure markdown>
![Kuramoto-Sivashinsky spacetime](figures/ks_spacetime.png)
<figcaption>Kuramoto–Sivashinsky, x vs t (<code>ks_1d</code>)</figcaption>
</figure>
</div>

<div class="pf-motion-grid" markdown>
<figure class="pf-motion-item" markdown>
<video autoplay loop muted playsinline>
<source src="../figures/fhn_spiral_motion.mp4" type="video/mp4">
</video>
<figcaption>FitzHugh–Nagumo spiral waves (<code>fitzhugh_nagumo_2d</code>)</figcaption>
</figure>
</div>

## Table

| Model | Family | Resolution |
|---|---|---|
| `burgers_1d` | spectral | any |
| `darcy_2d` | finite difference | any |
| `naca_flow_2d` | finite element | mesh-defined |

## Lists

- A bulleted item
- Another, with `inline code`
    - A nested item
- A third

1. First numbered step
2. Second, with $R(x,p)=0$ inline
3. Third

## Emphasis & links

**Bold**, *italic*, and a [link to the models gallery](guide/models.md).

> A blockquote: the traced branch is read as a dynamical object: its
> spectrum classifies folds, branch points and Hopf candidates.

---

A horizontal rule closes the card.
