# FEniCSx Setup

FEniCSx is required for models involving complex geometries, such as flow around obstacles. This guide covers installation and troubleshooting.

## Why FEniCSx?

Spectral methods work well for periodic boundary conditions on rectangular domains. For problems with:

- Obstacles (cylinders, airfoils)
- Non-rectangular domains
- Dirichlet/Neumann boundaries

we use the finite element method via FEniCSx.

## Installation

### Using the Setup Script

The simplest approach:

```bash
cd pdeforge
./setup_fenicsx_env.sh
```

This creates a conda environment `pdeforge-fenicsx` with all dependencies.

To specify a custom name:

```bash
./setup_fenicsx_env.sh my-env-name
```

### Manual Installation

```bash
# Create environment
conda create -n pdeforge-fenicsx python=3.11
conda activate pdeforge-fenicsx

# Install FEniCSx and dependencies
conda install -c conda-forge \
    fenics-dolfinx \
    mpich \
    petsc4py \
    gmsh \
    python-gmsh \
    pyvista

# Install PDEForge
pip install -e .
```

## Verifying the Installation

```python
# Test FEniCSx
import dolfinx
print(f"FEniCSx version: {dolfinx.__version__}")

# Test PDEForge with FEniCSx models
from pdeforge import list_models
models = list_models()
print(f"Available models: {models}")
assert "cylinder_flow_2d" in models
```

## Available FEniCSx Models

| Model | Description |
|-------|-------------|
| `cylinder_flow_2d` | Steady flow around a circular cylinder |
| `cylinder_flow_2d_unsteady` | Time-dependent flow with vortex shedding |

## Troubleshooting

### MPI Library Conflicts

If you see errors like `Library not loaded: @rpath/libmpi.dylib`:

1. Remove the conda environment completely
2. Reinstall from scratch using the setup script
3. Avoid mixing pip and conda installations for MPI-related packages

### gmsh Issues

Install gmsh only through conda-forge:

```bash
conda install -c conda-forge gmsh python-gmsh
```

Do not use `pip install gmsh` in a FEniCSx environment.

### Memory Issues

FEniCSx models can be memory-intensive. For large meshes:

- Reduce mesh resolution via `_mesh_resolution` parameter
- Reduce output grid resolution
- Generate samples in smaller batches

## Jupyter Integration

Register the environment as a Jupyter kernel:

```bash
conda activate pdeforge-fenicsx
python -m ipykernel install --user --name pdeforge-fenicsx
```

Then select "pdeforge-fenicsx" in Jupyter.
