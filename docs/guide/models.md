# Available Models

PDEForge provides implementations of several PDE problems commonly used in operator learning research.

## Spectral Models

These models use FFT-based solvers and work with the base installation.

### Burgers 1D (`burgers_1d`)

The viscous Burgers equation models advection with diffusion:

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$

**Operator learning task**: Initial condition to solution at final time

$$u(x, 0) \mapsto u(x, T)$$

```python
dataset = generate_dataset(
    model="burgers_1d",
    n_samples=1000,
    resolution={"x": 256},
    params={
        "viscosity": 0.01,      # Diffusion coefficient
        "time_horizon": 1.0,    # Final time T
    },
)
```

### Darcy 2D (`darcy_2d`)

Steady-state flow through porous media:

$$-\nabla \cdot (\kappa(x,y) \nabla u) = f$$

**Operator learning task**: Permeability field to pressure field

$$\kappa(x,y) \mapsto u(x,y)$$

```python
dataset = generate_dataset(
    model="darcy_2d",
    n_samples=1000,
    resolution={"x": 64, "y": 64},
    params={
        "kappa_min": 0.1,    # Minimum permeability
        "kappa_max": 10.0,   # Maximum permeability
    },
)
```

### Stokes 2D (`stokes_2d`)

Incompressible viscous flow at low Reynolds number:

$$-\mu \nabla^2 \mathbf{u} + \nabla p = \mathbf{f}, \quad \nabla \cdot \mathbf{u} = 0$$

**Operator learning task**: Body force to velocity and pressure fields

$$(f_x, f_y) \mapsto (u, v, p)$$

```python
dataset = generate_dataset(
    model="stokes_2d",
    n_samples=1000,
    resolution={"x": 64, "y": 64},
    params={
        "viscosity": 1.0,
        "n_force_modes": 5,   # Fourier modes in forcing
    },
)
```

### KdV 1D (`kdv_1d`)

One model, one solver, several regimes. `kdv_1d` solves

$$\frac{\partial u}{\partial t} + \mu\, u \frac{\partial u}{\partial x} + \delta^2 \frac{\partial^3 u}{\partial x^3} = 0$$

on a periodic box, with the stiff dispersion $i\,\delta^2 k^3$ integrated
exactly by ETDRK4 — the stiffness that cripples explicit schemes never
appears; only $\mu\,u u_x$ is stepped. Defaults are the textbook
normalisation ($\mu = 6$, $\delta^2 = 1$ on $[0, 20]$, soliton input measure).

The published KdV setups differ only in coefficients, box, and input measure,
so they ship as **presets** rather than separate models:

| Preset | Regime |
|---|---|
| *(model default)* | Solitons: $\mu=6$, $\delta^2=1$, $L=20$, elastic collisions |
| `kdv_dsw_1d` | Undular bore, $\delta^2 = 8\times10^{-6}$ — un-resolvable bias set |
| `kdv_dsw_epistemic_1d` | Undular bore, $\delta^2 = 4\times10^{-5}$ — near-resolvable |
| `mp_pde_kdv_1d` | Brandstetter et al. benchmark, $\mu=\delta^2=1$, $L=128$ |
| `mp_pde_kdv_easy_1d` | The same at $T = 50$ |

#### Dispersive shock waves (`kdv_dsw_1d`, `kdv_dsw_epistemic_1d`)

A localised smooth **depression** (the `depression_box` input measure) does not
steepen into a thin front; under KdV it dissolves into a **dispersive shock
wave** (an undular bore) — sustained high-wavenumber oscillations filling a
large, contiguous region. Where a Burgers shock mis-samples only a
$\sqrt{\nu}$-thin front, the bore is hard for a band-limited operator
*everywhere it lives*, which makes it a stringent operator / UQ benchmark.

```python
dataset = generate_dataset(preset="kdv_dsw_1d", n_samples=1000, seed=0)
```

The two bore presets are a matched pair. At $n_x = 512$ the vigorous bore
($\delta^2 = 8\times10^{-6}$) shows ~155 oscillations but is *not* grid
converged — it has ~429 at $n_x = 2048$, so its hardness is a **bias** floor
no amount of data removes. The longer-wavelength bore ($4\times10^{-5}$) shows
~83 oscillations and is already converged at $n_x = 512$, so its hardness is
**epistemic** (data-limited). That contrast is what makes them useful together
for method comparison.

!!! note "Spectral centroid will mislead you here"
    With dealiasing on, the 2/3 mask truncates the $8\times10^{-6}$ bore, so
    its spectral centroid reads *lower* (3.8) than the resolvable bore's
    (10.3) despite having the shorter wavelength. Count oscillations, not
    centroid — the truncation is the phenomenon, not an artefact to average over.

#### The neural-emulator benchmark regime (`mp_pde_kdv_1d`)

Brandstetter et al. (*Message Passing Neural PDE Solvers*, arXiv:2202.03376;
*Lie Point Symmetry Data Augmentation*, arXiv:2202.07643) use
$\mu = \delta^2 = 1$ on a long $L = 128$ box. That whole setup ships as a
preset:

```python
dataset = generate_dataset(preset="mp_pde_kdv_1d", n_samples=512, seed=0)
# -> inputs (512, 256), outputs (512, 140, 256) trajectories
```

The preset pins the four things that make that regime what it is:

| Setting | Value | Why it matters |
|---|---|---|
| `advection`, `dispersion` | `1.0`, `1.0` | Not the textbook $\mu = 6$ — different soliton amplitude-speed law |
| `ic_generator="sine_series"` | 10 waves, $l \in \{1, 2\}$ | Long-wave random sine series (see below) |
| `scale_jitter` | `0.1` | Each trajectory draws its own $L$ and $T$ within $\pm 10\%$ |
| `_n_frames_kept` | `140` of `250` | Trajectories start at $t \approx 0.44\,T$, from a developed soliton gas — not from the IC |

**The input measure.** `sine_series` (`TruncatedSineGenerator`) draws

$$u_0(x) = \sum_{j=1}^{N} A_j \sin\!\left(2\pi l_j x / L + \phi_j\right)$$

with $A_j \sim U(-0.5, 0.5)$, $\phi_j \sim U(0, 2\pi)$, and integer $l_j$ from
the **half-open** range $[l_\min, l_\max)$. The canonical $(1, 3)$ therefore
excites modes 1 and 2 only, never 3 — closing the interval would silently
widen the measure. The field is exactly zero-mean on a uniform periodic grid,
which KdV then conserves.

**Scale jitter.** KdV's scaling symmetry
$(u, x, t) \mapsto (\lambda^2 u, x/\lambda, t/\lambda^3)$ is what makes a
randomised box worth having rather than a relabelling. Because this measure is
a function of $x/L$, jitter leaves the *initial conditions* untouched and
perturbs only the dynamics. Note that the dataset's stored grid stays
**nominal**: per-sample $dx$ and $dt$ differ from it by up to the jitter
fraction, so the genuinely shared coordinate is $x/L$. Jitter requires
`backend="numpy"`.

**Dealiasing.** The preset sets `dealias=False` to match the reference
generator's `psdiff` right-hand side. This is not just fidelity theatre — at
$n_x = 256$ the 2/3 mask also removes *genuine* spectral content once KdV has
broadened the spectrum. Measured against a converged $n_x = 1024$ reference at
$T = 100$: un-dealiased ETDRK4 sits at $8 \times 10^{-7}$ relative $L^2$,
dealiased at $1 \times 10^{-4}$, and the reference generator's own
Radau + `psdiff` scheme at $2.9 \times 10^{-4}$. `_dt = 5e-3` is converged
there.

## FEniCSx Models

These require FEniCSx installation. See [FEniCSx Setup](../getting-started/fenicsx.md).

### Cylinder Flow 2D (`cylinder_flow_2d`)

Steady Navier-Stokes flow around a circular cylinder:

$$\rho (\mathbf{u} \cdot \nabla) \mathbf{u} - \mu \nabla^2 \mathbf{u} + \nabla p = 0$$
$$\nabla \cdot \mathbf{u} = 0$$

**Operator learning task**: Inlet velocity scale to flow field

$$\text{inlet scale} \mapsto (u, v, p)$$

```python
dataset = generate_dataset(
    model="cylinder_flow_2d",
    n_samples=100,
    resolution={"x": 128, "y": 64},
    params={
        "viscosity": 0.001,
        "inlet_velocity": 0.3,
        "cylinder_radius": 0.05,
    },
)
```

### Cylinder Flow 2D Unsteady (`cylinder_flow_2d_unsteady`)

Time-dependent flow exhibiting vortex shedding (von Karman vortex street):

**Operator learning task**: Inlet velocity to time-dependent flow trajectory

```python
dataset = generate_dataset(
    model="cylinder_flow_2d_unsteady",
    n_samples=10,
    resolution={"x": 110, "y": 41},
    params={
        "inlet_velocity": 1.0,
        "time_end": 8.0,
        "_n_time_steps": 81,
    },
)
# Output shape: (10, 81, 41, 110, 3)
```

### Elasticity 2D (`elasticity_2d`)

Plane-strain linear elasticity with random stiff inclusions; the operator
task maps the Young's-modulus field to displacement and von Mises stress.
Validated by the Clapeyron energy balance at solver precision.

```python
dataset = generate_dataset(
    model="elasticity_2d",
    n_samples=100,
    resolution={"x": 64, "y": 64},
    params={"e_inclusion": 10.0, "traction_y": -1.0},
)
```

### Rayleigh-Benard 2D (`rayleigh_benard_2d`)

Boussinesq convection in the closed unit cavity (hot bottom, cold top,
no-slip walls). Below Ra_c the solver returns the conduction state with
Nu = 1; above it, convection rolls whose selection depends on the seeded
perturbation. Plate-averaged Nusselt numbers are stored per solve.

```python
dataset = generate_dataset(
    model="rayleigh_benard_2d",
    n_samples=20,
    resolution={"x": 64, "y": 64},
    params={"rayleigh": 1e4, "prandtl": 0.71},
)
```

### Porous Darcy FEM (`porous_darcy_fem`)

The cross-model pipeline: a seeded Cahn-Hilliard run grows a two-phase
morphology, binarized into a permeability field; steady Darcy flow crosses
it under a unit pressure drop. Effective permeability is stored per
sample; flux balance and the pressure maximum principle are validated.

```python
dataset = generate_dataset(
    model="porous_darcy_fem",
    n_samples=100,
    resolution={"x": 64, "y": 64},
    params={"ch_time": 8.0, "permeability_contrast": 1e3},
)
```

## Model Information

Use `describe_model` to see configurable parameters:

```python
from pdeforge import describe_model
print(describe_model("burgers_1d"))
```

This shows:

- Physical parameters you can modify
- Default values and valid ranges
- Input/output field names
- Backend used (spectral or FEniCSx)
