# Stochastic heat 2D (`stochastic_heat_2d`)

Additive space-time noise on two-dimensional diffusion, with an ensemble per
initial condition. The linear structure makes this the cleanest test case in
the catalogue for uncertainty methods: the conditional distribution is Gaussian
and its covariance is known analytically, so a predicted spread can be checked
against the truth rather than only against a held-out sample.

<figure class="pf-model-fig" markdown>
![Stochastic heat 2D](../figures/model_stochastic_heat_2d.png)
<figcaption>Ensemble mean and ensemble standard deviation at t = T (<code>stochastic_heat_2d</code>): the mean follows the deterministic solve, the spread does not.</figcaption>
</figure>

## Equation

$$\frac{\partial u}{\partial t} = \alpha\,\nabla^2 u + \sigma\,\dot{W}$$

with periodic boundary conditions.

## Operator learning tasks

1. **Realisations.** $u_0 \mapsto \{u_T^{(1)}, u_T^{(2)}, \dots\}$
2. **Moments.** $u_0 \mapsto (\mathbb{E}[u_T],\ \operatorname{Var}[u_T])$

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `diffusivity` | 0.01 | (1e-6, 1.0) | Thermal diffusivity $\alpha$ |
| `noise_intensity` | 0.1 | (0.0, 1.0) | Noise amplitude $\sigma$ |
| `n_realizations` | 20 | (1, 200) | Realisations per initial condition |
| `time_end` | 1.0 | (0.01, 10.0) | Final time |

## Usage

```python
from pdeforge import generate_dataset

dataset = generate_dataset(
    model="stochastic_heat_2d",
    n_samples=50,
    resolution={"x": 64, "y": 64},
    params={"diffusivity": 0.01, "noise_intensity": 0.1, "n_realizations": 20},
    seed=42,
)
dataset.outputs.shape   # (50, 20, 64, 64)
```

!!! note "Storage grows with the ensemble"
    Outputs are `n_samples * n_realizations` fields. At $64^2$ in float64 that
    is 32 KB per field, so the call above stores 32 MB and raising
    `n_realizations` to 200 stores 320 MB. Pass `to=` to stream to disk.

## Solver

Exponential integrator for the exact diffusion with Euler-Maruyama noise
increments, matching the conventions of the other stochastic models. At
$\sigma = 0$ the model reduces to [`heat_2d`](heat_2d.md).

## Behaviour

The mean of the ensemble follows the deterministic solution exactly, because
additive noise has zero mean and the equation is linear. All the information
that distinguishes this model from `heat_2d` therefore lives in the second
moment, and a metric computed on the ensemble mean alone will show no
difference at all.

## Data shapes

```python
dataset.inputs.shape   # (n_samples, ny, nx)
dataset.outputs.shape  # (n_samples, n_realizations, ny, nx)
```

## Related

- [`heat_2d`](heat_2d.md): the deterministic limit.
- [`stochastic_heat_1d`](stochastic_heat_1d.md): the one-dimensional version.
- [`stochastic_allen_cahn_2d`](stochastic_allen_cahn_2d.md): noise on
  bistable dynamics, where realisations can end in different states entirely.
