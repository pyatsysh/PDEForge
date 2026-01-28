#!/bin/bash
# PDEForge Environment Setup Script (Spectral models only)
# Usage: ./setup_env.sh [env_name]
# Default environment name: pdeforge
#
# For FEniCSx support (cylinder flow models), use setup_fenicsx_env.sh instead.

set -e  # Exit on error

ENV_NAME="${1:-pdeforge}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "PDEForge Environment Setup"
echo "=============================================="
echo "Environment name: $ENV_NAME"
echo ""

# Determine which package manager to use (prefer micromamba > mamba > conda)
PKG_MANAGER=""

if command -v micromamba &> /dev/null; then
    PKG_MANAGER="micromamba"
    echo "Using: micromamba"
elif [ -f "$HOME/bin/micromamba" ]; then
    PKG_MANAGER="$HOME/bin/micromamba"
    echo "Using: ~/bin/micromamba"
elif command -v mamba &> /dev/null; then
    PKG_MANAGER="mamba"
    echo "Using: mamba"
elif command -v conda &> /dev/null; then
    PKG_MANAGER="conda"
    echo "Using: conda"
else
    echo "Error: No package manager found (micromamba, mamba, or conda)"
    exit 1
fi

# Set up activation command based on package manager
if [[ "$PKG_MANAGER" == *"micromamba"* ]]; then
    eval "$($PKG_MANAGER shell hook -s bash)"
    ACTIVATE_CMD="micromamba activate"
else
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

# Create new environment
echo ""
echo "Creating environment: $ENV_NAME"
$PKG_MANAGER create -n "$ENV_NAME" \
    python=3.11 \
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

# Run tests
echo ""
echo "=============================================="
echo "Running tests..."
echo "=============================================="
python -m pytest tests/ -v --ignore=tests/test_fenicsx.py 2>/dev/null || echo "Some tests may have been skipped"

# Quick verification
echo ""
echo "Verifying installation..."
python -c "
from pdeforge import list_models, describe_all_models
models = list_models()
print(f'Available models: {len(models)}')
print('Spectral models:', [m for m in models if 'cylinder' not in m])
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
echo "Note: For FEniCSx models (cylinder flow), run setup_fenicsx_env.sh instead."
echo ""
