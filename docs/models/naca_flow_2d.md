# NACA Flow 2D (`naca_flow_2d`)

Geometry is the data. Every sample draws its own airfoil, meaning thickness,
camber, camber position and angle of attack, plus an inlet-velocity scale; the
channel is re-meshed around it, and the solve returns the fields on a regular
grid together with the lift and drag coefficients from the surface stress
integral. The targets are the FlowBench ones, from a generator with knobs
rather than a frozen download.

<figure class="pf-model-fig" markdown>
![NACA flow 2D](../figures/naca_flow.png)
<figcaption>Speed and streamlines around a NACA 4412 at 6 degrees (<code>naca_flow_2d</code>), with the lift and drag from the surface stress integral.</figcaption>
</figure>

## Equations

Steady incompressible Navier-Stokes in a channel around the airfoil:

$$\rho\,(\mathbf{u} \cdot \nabla)\,\mathbf{u} - \mu\,\nabla^2 \mathbf{u} + \nabla p = 0,
\qquad \nabla \cdot \mathbf{u} = 0$$

with uniform far-field velocity on the inlet, top and bottom, a natural
outlet, and no slip on the airfoil.

## Operator learning task

$$\text{geometry, as a signed-distance field} \mapsto (u, v, p)$$

Per-sample parameters and the coefficients $(C_l, C_d)$ are recorded in the
dataset metadata.

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `viscosity` | 0.02 | (0.001, 1.0) | Dynamic viscosity; $\mathrm{Re} = U c / \nu$ |
| `inlet_velocity` | 1.0 | (0.1, 5.0) | Far-field velocity $U$ |
| `thickness_range` | (0.08, 0.18) | | NACA thickness draw range, as a chord fraction |
| `camber_range` | (0.0, 0.06) | | Maximum camber draw range |
| `camber_pos_range` | (0.3, 0.6) | | Camber position draw range |
| `aoa_range` | (-8.0, 8.0) | | Angle-of-attack draw range, in degrees |

At the defaults $U = 1$ and chord 1, so the Reynolds number is 50: a robust
steady laminar flow with a real viscous wake.

## Usage

This model needs FEniCSx. See [FEniCSx setup](../getting-started/fenicsx.md).

```python
from pdeforge import generate_dataset

dataset = generate_dataset(
    model="naca_flow_2d",
    n_samples=200,
    resolution={"x": 256, "y": 128},
    params={"viscosity": 0.02, "aoa_range": (-8.0, 8.0)},
    seed=42,
)
```

## Solver

Taylor-Hood elements with a Newton/SNES nonlinear solve and a direct LU
factorisation, on a mesh rebuilt per sample around the drawn airfoil.

## Scope

This is laminar incompressible Navier-Stokes at $\mathrm{Re} = 50$, and it is
airfoil-class data with knobs. It is deliberately not a recreation of the
transonic Geo-FNO airfoil dataset, nor of RANS-based AirfRANS. For a
shock-carrying compressible condition see
[`airfoil_euler_2d`](airfoil_euler_2d.md); for the viscous turbulence-closed
regime PDEForge [reads AirfRANS](../guide/data-formats.md#airfrans) rather than
rebuilding it.

## Data shapes

```python
dataset.inputs.shape   # (n_samples, ny, nx)      the signed-distance field
dataset.outputs.shape  # (n_samples, ny, nx, 3)   u, v, p
```

## Related

- [`airfoil_euler_2d`](airfoil_euler_2d.md): the compressible transonic case,
  on a body-fitted C-grid.
- [`cylinder_flow_2d_parameterized`](cylinder_flow_2d_parameterized.md):
  geometry as input, with position varying rather than shape.
- [`stokes_2d`](stokes_2d.md): the inertia-free limit.
