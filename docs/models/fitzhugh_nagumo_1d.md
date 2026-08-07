# FitzHugh-Nagumo 1D (`fitzhugh_nagumo_1d`)

Excitable media: the model of neurons and cardiac tissue, and the model of
threshold behaviour generally. A small perturbation decays; a perturbation past
the threshold launches a travelling pulse that propagates without decay and
leaves a refractory tail behind it. That threshold is a genuine discontinuity
in the input-to-output map, and it is what makes the operator task interesting
rather than a smoothing exercise.

<figure class="pf-model-fig" markdown>
![FitzHugh-Nagumo 1D](../figures/model_fitzhugh_nagumo_1d.png)
<figcaption>Space-time diagram of the activator u (<code>fitzhugh_nagumo_1d</code>): one supra-threshold stimulus launches two counter-propagating pulses.</figcaption>
</figure>

## Equation

$$\frac{\partial u}{\partial t}
= D_u\,\frac{\partial^2 u}{\partial x^2} + u - u^3 - v$$

$$\frac{\partial v}{\partial t}
= D_v\,\frac{\partial^2 v}{\partial x^2} + \epsilon\,(u - \gamma v + \beta)$$

with periodic boundary conditions. Here $u$ is the fast activator, standing for
membrane potential, and $v$ is the slow inhibitor, standing for recovery.

## Operator learning task

$$u_0 \mapsto (u_T, v_T)$$

The inhibitor starts from its rest state rather than being drawn, so the
activator field is the only varying input.

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `epsilon` | 0.08 | (0.01, 0.5) | Timescale separation; smaller gives sharper pulses and slower recovery |
| `diffusivity_u` | 1.0 | (0.01, 10.0) | Activator diffusion $D_u$, which sets the wave speed |
| `time_end` | 50.0 | (1.0, 200.0) | Final time |

## Usage

```python
from pdeforge import generate_dataset

dataset = generate_dataset(
    model="fitzhugh_nagumo_1d",
    n_samples=500,
    resolution={"x": 256},
    params={"epsilon": 0.08, "diffusivity_u": 1.0, "time_end": 50.0},
    seed=42,
)
```

## Behaviour

The timescale separation $\epsilon$ is what makes the medium excitable. When
$\epsilon \ll 1$ the inhibitor lags far behind the activator, so a pulse has
time to form and travel before recovery catches up. Raising $\epsilon$ towards
0.5 removes that separation and the pulses stop being sharp.

An excitation seed has to be wide enough to launch a pulse at all: roughly 8
units of $\sqrt{D_u}$. Narrower stimuli decay whatever their amplitude, which
is worth knowing when designing an input measure, since a measure that mostly
produces sub-threshold perturbations will produce a dataset of decays.

## Data shapes

```python
dataset.inputs.shape   # (n_samples, nx)      u0
dataset.outputs.shape  # (n_samples, nx, 2)   u_T, v_T on a trailing axis
```

The inhibitor is initialised internally rather than drawn, so only $u_0$ comes
back as the input; both components are returned.

## Related

- [`fitzhugh_nagumo_2d`](fitzhugh_nagumo_2d.md): spirals become possible once
  a front can break.
- [`allen_cahn_1d`](allen_cahn_1d.md): the same cubic nonlinearity without the
  inhibitor, giving stationary interfaces instead of travelling pulses.
- [`gray_scott_2d`](gray_scott_2d.md): the other two-component
  reaction-diffusion system.
