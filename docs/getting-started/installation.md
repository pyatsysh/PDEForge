# Installation

PDEForge can be installed via pip. For models requiring FEniCSx (complex geometries), additional conda packages are needed.

## Basic Installation

```bash
pip install pdeforge
```

This provides access to spectral-based models:

- `burgers_1d` - 1D Burgers equation
- `darcy_2d` - 2D Darcy flow
- `stokes_2d` - 2D Stokes flow

## Installation with Optional Dependencies

```bash
# Interactive visualization in Jupyter
pip install pdeforge[notebook]

# HDF5 file format support
pip install pdeforge[hdf5]

# Development tools (testing, linting)
pip install pdeforge[dev]

# All optional dependencies
pip install pdeforge[all]
```

## Development Installation

For contributing or modifying PDEForge:

```bash
git clone https://github.com/pdeforge/pdeforge.git
cd pdeforge
pip install -e ".[dev]"
```

## FEniCSx Installation

Models with complex geometries (e.g., `cylinder_flow_2d`) require FEniCSx, which must be installed via conda:

### Option 1: Setup Script (Recommended)

```bash
cd pdeforge
./setup_fenicsx_env.sh
conda activate pdeforge-fenicsx
```

### Option 2: Manual Installation

```bash
conda create -n pdeforge-fenicsx python=3.11
conda activate pdeforge-fenicsx
conda install -c conda-forge fenics-dolfinx mpich petsc4py gmsh python-gmsh
pip install pdeforge
```

FEniCSx installation typically takes 10-15 minutes.

## Verifying Installation

```python
from pdeforge import list_models, generate_dataset

# Check available models
print(list_models())

# Quick test
dataset = generate_dataset(
    model="burgers_1d",
    n_samples=5,
    resolution={"x": 64},
)
print(f"Generated dataset with shape: {dataset.inputs.shape}")
```

## System Requirements

- Python 3.8 or higher
- NumPy 1.20+
- SciPy 1.7+
- Matplotlib 3.4+

For FEniCSx models:

- conda/mamba package manager
- FEniCSx (via conda-forge)
- 4GB+ RAM recommended
