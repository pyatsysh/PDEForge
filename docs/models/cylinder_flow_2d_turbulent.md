# Cylinder Flow 2D Turbulent (`cylinder_flow_2d_turbulent`)

Time-dependent turbulent flow around a circular cylinder with **parameterized cylinder position** and **Smagorinsky LES** turbulence modeling for high Reynolds number flows.

<figure class="pf-model-fig" markdown>
<video autoplay loop muted playsinline>
<source src="../../figures/cylinder_turbulent_motion.mp4" type="video/mp4">
</video>
<figcaption>Re 2000 wake vorticity with the base shear removed (<code>cylinder_flow_2d_turbulent</code>): the cylinder position is a per-sample input.</figcaption>
</figure>

## Equations

LES-filtered incompressible Navier-Stokes:

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} - \nabla \cdot ((\nu + \nu_t) \nabla \mathbf{u}) + \nabla p = 0$$

$$\nabla \cdot \mathbf{u} = 0$$

### Smagorinsky Turbulence Model

The turbulent eddy viscosity is computed as:

$$\nu_t = (C_s \Delta)^2 |S|$$

where:
- $C_s \approx 0.1$ is the Smagorinsky constant
- $\Delta$ is the filter width (mesh cell size)
- $|S| = \sqrt{2 S_{ij} S_{ij}}$ is the strain rate magnitude
- $S_{ij} = \frac{1}{2}\left(\frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i}\right)$

## Reynolds Number

$$Re = \frac{U \cdot D}{\nu}$$

where:
- $U$ = inlet velocity
- $D = 2r$ = cylinder diameter
- $\nu$ = kinematic viscosity

### Flow Regimes

| Reynolds Number | Regime | Characteristics |
|-----------------|--------|-----------------|
| Re < 5 | Creeping flow | No separation |
| 5 < Re < 40 | Steady separated | Twin vortices behind cylinder |
| 40 < Re < 200 | Laminar vortex shedding | Von Kármán street |
| 200 < Re < 300,000 | Turbulent wake | Irregular vortex shedding |
| Re > 300,000 | Fully turbulent | Turbulent boundary layer |

## Boundary Conditions

- **Inlet**: Parabolic velocity profile $u(y) = \frac{4 U_{max} y (H-y)}{H^2}$
- **Outlet**: Zero-stress (do-nothing)
- **Walls**: No-slip
- **Cylinder surface**: No-slip

## Operator Learning Task

Map inlet velocity scale and cylinder position to flow field trajectory:

$$(s, c_x, c_y) \mapsto \{(u, v, p)_t\}_{t=0}^{T}$$

This enables learning:
1. Flow patterns for different cylinder positions
2. Turbulent dynamics and vortex shedding
3. Effect of Reynolds number on wake structure

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `viscosity` | 0.0001 | (1e-6, 0.01) | Kinematic viscosity ν [m²/s] |
| `inlet_velocity` | 1.0 | (0.1, 5.0) | Mean inlet velocity [m/s] |
| `cylinder_radius` | 0.05 | (0.02, 0.1) | Cylinder radius [m] |
| `cx_range` | (0.15, 0.5) | - | Range for cylinder x-position |
| `cy_range` | (0.15, 0.26) | - | Range for cylinder y-position |
| `use_les` | True | - | Enable Smagorinsky LES |
| `smagorinsky_constant` | 0.1 | (0.05, 0.2) | C_s constant |
| `time_end` | 10.0 | (1.0, 50.0) | Simulation time [s] |
| `n_time_steps` | 101 | (21, 501) | Output time steps |

## Usage

Requires FEniCSx. See [FEniCSx Setup](../getting-started/fenicsx.md).

### Basic Usage

```python
from pdeforge import generate_dataset

dataset = generate_dataset(
    model="cylinder_flow_2d_turbulent",
    n_samples=5,
    resolution={"x": 220, "y": 82},
    params={
        "inlet_velocity": 1.0,
        "viscosity": 0.0001,  # Re ~ 1000
        "use_les": True,
        "time_end": 10.0,
        "n_time_steps": 51,
    },
    seed=42,
)
```

### Direct Model Access

```python
from pdeforge import get_model

Model = get_model("cylinder_flow_2d_turbulent")
model = Model(
    resolution={"x": 220, "y": 82},
    inlet_velocity=1.0,
    viscosity=0.0001,
    cx_range=(0.2, 0.4),
    cy_range=(0.15, 0.25),
)

print(model.describe())
# Reynolds number: 1000

# Generate trajectory with specific cylinder position
trajectory = model.solve(inlet_scale=1.0, cx=0.25, cy=0.2)
# trajectory.shape: (n_time_steps, nx, ny, 3)
```

## Solver

Time-dependent finite element method using FEniCSx:

- **Spatial discretization**: Taylor-Hood elements (P2-P1)
- **Time integration**: Backward Euler (implicit)
- **Nonlinear solver**: Newton with line search
- **Turbulence**: Smagorinsky LES subgrid-scale model
- **Mesh**: gmsh with cylinder hole, finer resolution for turbulence

### Time Stepping

Adaptive time step based on CFL condition:

$$\Delta t = C_{safety} \cdot \frac{h_{min}}{U_{max}}$$

where $C_{safety} \approx 0.5$ ensures stability.

## Domain

Default channel: 2.2 × 0.41 m with cylinder of radius 0.05 m.

Cylinder position can vary within:
- x ∈ [0.15, 0.5] (default)
- y ∈ [0.15, 0.26] (default)

## Data Shapes

```python
# For time-dependent output
dataset.inputs.shape   # (n_samples, 3)  # [inlet_scale, cx, cy]
dataset.outputs.shape  # (n_samples, n_time, nx, ny, 3)  # u, v, p over time
```

## Physical Behavior

### Vortex Shedding

At moderate Reynolds numbers (Re > 40), alternating vortices shed from the cylinder, forming the **von Kármán vortex street**. The shedding frequency is characterized by the **Strouhal number**:

$$St = \frac{f \cdot D}{U} \approx 0.2$$

### Turbulent Wake

At higher Reynolds numbers (Re > 200), the wake becomes turbulent with:
- Irregular vortex shedding
- Increased mixing
- Higher drag coefficient
- Broadband frequency spectrum

## Comparison with Other Cylinder Models

| Model | Steady/Unsteady | Cylinder Position | Turbulence | Re Range |
|-------|-----------------|-------------------|------------|----------|
| `cylinder_flow_2d` | Steady | Fixed | No | < 100 |
| `cylinder_flow_2d_unsteady` | Unsteady | Fixed | No | < 500 |
| `cylinder_flow_2d_parameterized` | Steady | Variable | No | < 100 |
| `cylinder_flow_2d_turbulent` | Unsteady | Variable | LES | Up to 10,000 |

## Notes

1. **Computational cost**: Higher Re requires finer mesh and smaller time steps
2. **Initialization**: Flow is initialized with steady Stokes solution
3. **Validation**: Solutions checked for NaN/Inf, max velocity reported
4. **Grid interpolation**: FEM solutions interpolated to regular grid for ML
5. **Cylinder interior**: Filled with zeros, use `domain_mask` in metadata

## References

1. Smagorinsky, J. (1963). "General circulation experiments with the primitive equations"
2. Williamson, C.H.K. (1996). "Vortex dynamics in the cylinder wake"
3. FEniCSx documentation: https://docs.fenicsproject.org/
