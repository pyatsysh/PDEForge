# Burgers 1D (`burgers_1d`)

Burgers is the shortest route from a smooth initial condition to a feature no
band-limited method represents well. Advection steepens the profile, viscosity
arrests the steepening, and the balance settles at a front of width
$O(\sqrt{\nu})$. Lower the viscosity and the front thins, and the fraction of
the domain where the error concentrates thins with it, which is exactly the
regime where averaged error metrics stop telling you anything useful.

<figure class="pf-model-fig" markdown>
![Burgers 1D](../figures/burgers_1d.png)
<figcaption>Solutions at successive times (<code>burgers_1d</code>): the profile steepens until viscosity holds the front.</figcaption>
</figure>

## Equation

$$\frac{\partial u}{\partial t} + u\,\frac{\partial u}{\partial x}
= \nu\,\frac{\partial^2 u}{\partial x^2}$$

with periodic boundary conditions on $[0, 2\pi]$.

## Operator learning task

$$u(x, 0) \mapsto u(x, T)$$

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `viscosity` | $0.01/\pi$ | (1e-6, 1.0) | Diffusion coefficient $\nu$ |
| `time_horizon` | 1.0 | (0.1, 10.0) | Final time $T$ |

The advection coefficient is fixed at 1.0 by default and opened up by the
`advection` parameter where a published setup needs a different normalisation.

## Usage

```python
from pdeforge import generate_dataset

dataset = generate_dataset(
    model="burgers_1d",
    n_samples=1000,
    resolution={"x": 256},
    params={"viscosity": 0.01, "time_horizon": 1.0},
    seed=42,
)
```

## Presets

Three published Burgers setups and a three-rung regularity ladder ship as
presets, so the input measure travels with the coefficients:

| Preset | Setup |
|---|---|
| `fno_burgers_1d` | Sine prior at $\nu = 0.01/\pi$, $a_n \sim N(0, 0.49/n^3)$ |
| `fno_burgers_grf_1d` | The official FNO GRF measure $N(0, 625(-\Delta + 25)^{-2})$ |
| `burgers_smooth_1d` | Ladder, smooth end: $\nu = 0.1/\pi$, 3 modes, nearly featureless |
| `burgers_canonical_1d` | Ladder, middle: $\nu = 0.01/\pi$, 9 modes, paper-baseline fronts |
| `burgers_rough_1d` | Ladder, sharp end: $\nu = 0.0025/\pi$, 15 modes, front-dominated |
| `pdebench_burgers_1d` | PDEBench-style low viscosity, shock-rich |

```python
dataset = generate_dataset(preset="burgers_rough_1d", n_samples=1000, seed=0)
```

The ladder exists so that front sharpness can be varied without also varying
the solver or the sampling: the three rungs share everything except viscosity
and the number of excited modes. All three were validated against an
independent ETDRK4 reference implementation at about $3\times10^{-8}$
relative $L^2$.

## Solver

ETDRK4 on the spectral seam. The stiff diffusion term is integrated exactly
and only the advection nonlinearity is stepped explicitly, in conservative
form with 2/3 dealiasing. That split is what keeps the low-viscosity rungs
affordable, since an explicit treatment of diffusion would tie the step size
to $\nu / \Delta x^2$.

## Initial conditions

The default generator is a Fourier series with random coefficients,

$$u_0(x) = \sum_{k=1}^{N} a_k \sin(kx),$$

with amplitudes decaying as $k^{-\alpha}$. That decay rate is the second
difficulty knob alongside viscosity: it sets how much fine structure the front
has to form out of.

## Behaviour

Low viscosity gives sharp fronts and high viscosity keeps solutions smooth. A
longer horizon means more nonlinear evolution and more opportunity for fronts
to interact, though on the periodic box the very long-time state is dominated
by decay.

## Data shapes

```python
dataset.inputs.shape   # (n_samples, nx)
dataset.outputs.shape  # (n_samples, nx)
```

## Related

- [`burgers_2d`](burgers_2d.md): the vector version, with fronts in two
  dimensions.
- [`kdv_1d`](kdv_1d.md): the same nonlinearity balanced by dispersion instead,
  which spreads the difficulty over a wide bore rather than a thin front.
- [`stochastic_burgers_1d`](stochastic_burgers_1d.md): the same equation under
  noise, with an ensemble per initial condition.
- [`advection_1d`](advection_1d.md): the linear limit, where the speed no
  longer depends on the solution.
