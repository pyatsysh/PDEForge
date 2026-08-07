# Stochastic Allen-Cahn 2D (`stochastic_allen_cahn_2d`)

Phase separation with thermal fluctuations, and the model where the ensemble
genuinely branches. Noise nucleates and roughens interfaces, and strong enough
noise flips whole domains between the $\pm 1$ wells. Two realisations from the
same initial condition can therefore end up in different macroscopic states,
which is a qualitatively different kind of uncertainty from the perturbative
spread of the linear models.

<figure class="pf-model-fig" markdown>
![Stochastic Allen-Cahn 2D](../figures/model_stochastic_allen_cahn_2d.png)
<figcaption>Two realisations from the same initial condition (<code>stochastic_allen_cahn_2d</code>): the domain pattern is not shared.</figcaption>
</figure>

## Equation

$$du = \left[\varepsilon\,\nabla^2 u + u - u^3\right] dt
+ \sigma\,dW(t, x, y)$$

on the periodic box.

## Operator learning task

$$u_0 \mapsto \{u_T^{(1)}, u_T^{(2)}, \dots\}$$

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `epsilon` | 0.01 | (0.001, 0.5) | Interface parameter |
| `noise_intensity` | 0.05 | (0.0, 2.0) | Noise amplitude $\sigma$ |
| `n_realizations` | 10 | (1, 1000) | Realisations per initial condition |
| `time_end` | 2.0 | (0.05, 100.0) | Final time |

## Usage

```python
from pdeforge import generate_dataset

dataset = generate_dataset(
    model="stochastic_allen_cahn_2d",
    n_samples=100,
    resolution={"x": 64, "y": 64},
    params={"epsilon": 0.01, "noise_intensity": 0.05, "n_realizations": 20},
    seed=42,
)
dataset.outputs.shape   # (100, 20, 64, 64)
```

## Solver

Exponential integrator for the linear part, meaning diffusion together with the
linear reaction term, explicit cubic, and Euler-Maruyama noise. At
$\sigma \to 0$ the deterministic phase-separation dynamics is recovered.

## Behaviour

The distribution over outcomes is **multimodal**, and that breaks the usual
uncertainty summaries. An ensemble mean taken across realisations that settled
into different domain patterns is a blurred field belonging to no member, and a
per-point variance reports a large number everywhere the members disagree
without saying that they disagree about the pattern rather than the value.
Conditional-distribution methods have something real to do here in a way they
do not for [`stochastic_heat_2d`](stochastic_heat_2d.md).

Noise intensity sets how often branching happens. At the default 0.05 the
interfaces roughen but domains rarely flip; raising it towards 0.5 makes flips
routine.

## Data shapes

```python
dataset.inputs.shape   # (n_samples, ny, nx)
dataset.outputs.shape  # (n_samples, n_realizations, ny, nx)
```

## Related

- [`allen_cahn_2d`](allen_cahn_2d.md): the deterministic model.
- [`stochastic_heat_2d`](stochastic_heat_2d.md): unimodal, Gaussian, and much
  easier to summarise.
- The [stochastic systems guide](../advanced/stochastic.md).
