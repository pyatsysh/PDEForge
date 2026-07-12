#!/bin/bash
# PDEForge Environment Setup with FEniCSx Support
# Usage: ./setup_fenicsx_env.sh [env_name]
# Default environment name: pdeforge-fenicsx
#
# This script uses micromamba (preferred) or mamba for fast dependency resolution.
# Conda's classic solver can hang for hours on FEniCSx dependencies.

set -e  # Exit on error

ENV_NAME="${1:-pdeforge-fenicsx}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "PDEForge + FEniCSx Environment Setup"
echo "=============================================="
echo "Environment name: $ENV_NAME"
echo ""

# Determine which package manager to use (prefer micromamba > mamba > conda)
PKG_MANAGER=""
ACTIVATE_CMD=""

if command -v micromamba &> /dev/null; then
    PKG_MANAGER="micromamba"
    echo "Using: micromamba (fastest)"
elif [ -f "$HOME/bin/micromamba" ]; then
    PKG_MANAGER="$HOME/bin/micromamba"
    echo "Using: ~/bin/micromamba (fastest)"
elif command -v mamba &> /dev/null; then
    PKG_MANAGER="mamba"
    echo "Using: mamba (fast)"
elif command -v conda &> /dev/null; then
    echo ""
    echo "WARNING: Only conda found. Conda's classic solver can be very slow"
    echo "         (30+ minutes or may hang) for FEniCSx dependencies."
    echo ""
    echo "Recommendation: Install micromamba for faster setup:"
    echo "  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C ~ bin/micromamba"
    echo ""
    read -p "Continue with conda anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. Install micromamba and re-run this script."
        exit 1
    fi
    PKG_MANAGER="conda"
else
    echo "Error: No package manager found (micromamba, mamba, or conda)"
    echo ""
    echo "Install micromamba with:"
    echo "  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C ~ bin/micromamba"
    exit 1
fi

# Set up activation command based on package manager
if [[ "$PKG_MANAGER" == *"micromamba"* ]]; then
    # Initialize micromamba shell
    eval "$($PKG_MANAGER shell hook -s bash)"
    ACTIVATE_CMD="micromamba activate"
else
    # Use conda's activation
    source "$(conda info --base)/etc/profile.d/conda.sh"
    ACTIVATE_CMD="conda activate"
fi

# Remove existing environment if it exists
echo ""
echo "Checking for existing environment..."
if $PKG_MANAGER env list 2>/dev/null | grep -q "$ENV_NAME"; then
    echo "Removing existing environment: $ENV_NAME"
    $PKG_MANAGER env remove -n "$ENV_NAME" -y 2>/dev/null || true
fi

# Create new environment with FEniCSx
echo ""
echo "Creating environment with FEniCSx..."
echo "This typically takes 30-60 seconds with micromamba/mamba."
echo ""

$PKG_MANAGER create -n "$ENV_NAME" \
    python=3.11 \
    fenics-dolfinx \
    petsc4py \
    mpi4py \
    gmsh \
    python-gmsh \
    pyvista \
    numpy \
    scipy \
    matplotlib \
    tqdm \
    ipykernel \
    ipywidgets \
    pytest \
    -c conda-forge -y

# Activate the environment
echo ""
echo "Activating environment..."
$ACTIVATE_CMD "$ENV_NAME"

# Install PDEForge in development mode
echo ""
echo "Installing PDEForge in development mode..."
pip install -e "$SCRIPT_DIR"

# Register Jupyter kernel (name must match notebooks)
echo ""
echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name pdeforge --display-name "PDEForge"

# Test FEniCSx import
echo ""
echo "Testing installation..."
python -c "
import dolfinx
print(f'FEniCSx version: {dolfinx.__version__}')

from pdeforge import list_models
models = list_models()
print(f'Available models: {len(models)}')

fenicsx_models = [m for m in models if 'cylinder' in m]
if fenicsx_models:
    print(f'FEniCSx models: {fenicsx_models}')
    print('SUCCESS: FEniCSx models loaded!')
else:
    print('WARNING: FEniCSx models not found')
"

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "To activate the environment:"
if [[ "$PKG_MANAGER" == *"micromamba"* ]]; then
    echo "    eval \"\$(~/bin/micromamba shell hook -s bash)\""
    echo "    micromamba activate $ENV_NAME"
else
    echo "    conda activate $ENV_NAME"
fi
echo ""
echo "In Jupyter, select kernel: 'PDEForge'"
echo ""
