# Available Models

PDEForge ships 41 models under one API. Every one of them is called the same
way, and each has its own page under [Models](../models/index.md) covering the
equation, the operator task, the parameters and their ranges, a runnable
snippet, and a figure.

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

This page is the catalogue. Start from the [Models index](../models/index.md)
if you would rather browse by physics with the figures alongside.

## What you need installed

| Group | Count | Requirement |
|---|---|---|
| Spectral | 30 | base installation |
| Finite-difference elliptic | 2 | base installation |
| Finite volume | 1 | base installation (pure NumPy) |
| Finite element | 8 | [FEniCSx setup](../getting-started/fenicsx.md) or the Docker image |

Two of the spectral models, [`ns_vorticity_2d`](../models/ns_vorticity_2d.md)
and [`kolmogorov_flow_2d`](../models/kolmogorov_flow_2d.md), run considerably
faster with `backend="jax"`, and [`gray_scott_2d`](../models/gray_scott_2d.md)
benefits at long horizons.

## The catalogue

### Diffusion and transport

| Model | Operator task |
|---|---|
| [`heat_1d`](../models/heat_1d.md) | $u(x,0) \mapsto u(x,T)$ |
| [`heat_2d`](../models/heat_2d.md) | $u(x,y,0) \mapsto u(x,y,T)$ |
| [`heat_3d`](../models/heat_3d.md) | the same on the periodic cube |
| [`advection_1d`](../models/advection_1d.md) | exact translation; the sanity anchor |
| [`darcy_2d`](../models/darcy_2d.md) | $\kappa(x,y) \mapsto u(x,y)$, periodic |

### Waves and dispersion

| Model | Operator task |
|---|---|
| [`wave_1d`](../models/wave_1d.md) | $u(x,0) \mapsto u(x,T)$, energy conserved |
| [`wave_2d`](../models/wave_2d.md) | the same on the square |
| [`heterogeneous_wave_2d`](../models/heterogeneous_wave_2d.md) | $c(x,y) \mapsto$ wavefield; the medium is the input |
| [`helmholtz_2d`](../models/helmholtz_2d.md) | $f \mapsto \operatorname{Re} u$, frequency domain |
| [`kdv_1d`](../models/kdv_1d.md) | solitons, undular bores, benchmark regimes |
| [`schrodinger_1d`](../models/schrodinger_1d.md) | complex field as two real channels |

### Nonlinear advection and turbulence

| Model | Operator task |
|---|---|
| [`burgers_1d`](../models/burgers_1d.md) | shock formation; the regularity ladder |
| [`burgers_2d`](../models/burgers_2d.md) | vector self-advection, no pressure |
| [`ks_1d`](../models/ks_1d.md) | spatiotemporal chaos |
| [`ns_vorticity_2d`](../models/ns_vorticity_2d.md) | $w(\cdot,0) \mapsto w(\cdot,T)$; the canonical benchmark |
| [`kolmogorov_flow_2d`](../models/kolmogorov_flow_2d.md) | forced, statistically steady turbulence |
| [`shallow_water_2d`](../models/shallow_water_2d.md) | $(h, hu, hv)$; mass conserved exactly |

### Pattern formation and phase separation

| Model | Operator task |
|---|---|
| [`allen_cahn_1d`](../models/allen_cahn_1d.md) | interfaces annihilating in pairs |
| [`allen_cahn_2d`](../models/allen_cahn_2d.md) | curvature-driven coarsening |
| [`allen_cahn_3d`](../models/allen_cahn_3d.md) | the same on the cube |
| [`cahn_hilliard`](../models/cahn_hilliard.md) | spinodal decomposition, 2D or 3D |
| [`gray_scott_2d`](../models/gray_scott_2d.md) | the Pearson parameter plane |
| [`fitzhugh_nagumo_1d`](../models/fitzhugh_nagumo_1d.md) | excitable pulses past a threshold |
| [`fitzhugh_nagumo_2d`](../models/fitzhugh_nagumo_2d.md) | broken fronts, spirals |
| [`lotka_volterra_2d`](../models/lotka_volterra_2d.md) | predator-prey with diffusion |

### Porous media and solids

| Model | Operator task | Needs |
|---|---|---|
| [`darcy_fno_2d`](../models/darcy_fno_2d.md) | $a \mapsto u$; bit-exact against the published data | base |
| [`darcy_fno_3d`](../models/darcy_fno_3d.md) | the same measure on the cube | base |
| [`elasticity_2d`](../models/elasticity_2d.md) | $E \mapsto (u, v, \sigma_{vM})$ | FEniCSx |
| [`porous_darcy_fem`](../models/porous_darcy_fem.md) | $k \mapsto (p, u_x, u_y)$ through a grown microstructure | FEniCSx |

### Viscous and compressible flow

| Model | Operator task | Needs |
|---|---|---|
| [`stokes_2d`](../models/stokes_2d.md) | $(f_x, f_y) \mapsto (u, v, p)$, creeping flow | base |
| [`cylinder_flow_2d`](../models/cylinder_flow_2d.md) | inlet scale $\mapsto (u, v, p)$ | FEniCSx |
| [`cylinder_flow_2d_unsteady`](../models/cylinder_flow_2d_unsteady.md) | the vortex street, as a trajectory | FEniCSx |
| [`cylinder_flow_2d_parameterized`](../models/cylinder_flow_2d_parameterized.md) | cylinder position as input | FEniCSx |
| [`cylinder_flow_2d_turbulent`](../models/cylinder_flow_2d_turbulent.md) | Re 2000, Smagorinsky LES | FEniCSx |
| [`naca_flow_2d`](../models/naca_flow_2d.md) | airfoil geometry $\mapsto$ flow, with $C_l$ and $C_d$ | FEniCSx |
| [`rayleigh_benard_2d`](../models/rayleigh_benard_2d.md) | convection in a cavity, Nusselt-validated | FEniCSx |
| [`airfoil_euler_2d`](../models/airfoil_euler_2d.md) | transonic Euler with a shock, on a C-grid | base |

### Stochastic PDEs

Each sample carries several realisations of the same solve, so outputs have an
extra realisation axis and the target is a distribution. See the
[calibration protocol](calibration.md) and the
[stochastic systems guide](../advanced/stochastic.md).

| Model | Operator task |
|---|---|
| [`stochastic_heat_1d`](../models/stochastic_heat_1d.md) | $u_0 \mapsto \{u_T^{(i)}\}$, Gaussian |
| [`stochastic_heat_2d`](../models/stochastic_heat_2d.md) | the same on the square |
| [`stochastic_burgers_1d`](../models/stochastic_burgers_1d.md) | spread concentrated at the fronts |
| [`stochastic_allen_cahn_2d`](../models/stochastic_allen_cahn_2d.md) | multimodal: realisations branch |

## Presets

Published benchmark setups ship as presets rather than as separate models,
because they differ from the base model only in coefficients, domain and input
measure. A preset pins all three together, so the measure travels with the
physics.

```python
dataset = generate_dataset(preset="fno_darcy_2d", n_samples=1000, seed=0)
```

| Preset | Model | What it pins |
|---|---|---|
| `fno_darcy_2d` | `darcy_fno_2d` | Canonical Darcy421, log-normal; bit-exact against the distributed data |
| `fno_darcy_clean_2d` | `darcy_fno_2d` | The Darcy421 measure on the node grid, with no resampling |
| `fno_darcy_piececonst_2d` | `darcy_fno_2d` | The two-phase pushforward, $\{12, 3\}$ |
| `fno_burgers_1d` | `burgers_1d` | Sine prior at $\nu = 0.01/\pi$ |
| `fno_burgers_grf_1d` | `burgers_1d` | The official GRF measure $N(0, 625(-\Delta + 25)^{-2})$ |
| `fno_ns_vorticity_2d` | `ns_vorticity_2d` | Forced NS, $\nu = 10^{-3}$, $T = 50$ |
| `burgers_smooth_1d` | `burgers_1d` | Regularity ladder, smooth end |
| `burgers_canonical_1d` | `burgers_1d` | Regularity ladder, paper-baseline fronts |
| `burgers_rough_1d` | `burgers_1d` | Regularity ladder, front-dominated |
| `pdebench_burgers_1d` | `burgers_1d` | PDEBench-style low viscosity, shock-rich |
| `kdv_dsw_1d` | `kdv_1d` | Undular bore, vigorous and un-resolvable at $n_x = 512$ |
| `kdv_dsw_epistemic_1d` | `kdv_1d` | Undular bore, near-resolvable |
| `mp_pde_kdv_1d` | `kdv_1d` | The Brandstetter et al. MP-PDE regime |
| `mp_pde_kdv_easy_1d` | `kdv_1d` | The same at $T = 50$ |

```python
from pdeforge import list_presets
from pdeforge.presets import get_preset

list_presets()
get_preset("kdv_dsw_1d")     # the full pinned configuration
```

## Model information

`describe_model` reports what a model accepts without you reading its source:

```python
from pdeforge import describe_model
print(describe_model("burgers_1d"))
```

It shows the physical parameters you can modify, their defaults and valid
ranges, the input and output field names, and the backend.
