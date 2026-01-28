# Contributing to PDEForge

We welcome contributions! This document provides guidelines for contributing to PDEForge.

## Design Philosophy

**PDEForge is for generating ML training datasets, not a general PDE solver library.**

When adding models, expose only parameters that affect the characteristics of the generated data:

| EXPOSE to Users | HIDE from Users |
|-----------------|-----------------|
| Physical parameters (viscosity, Re) | Solver tolerances |
| Domain/geometry parameters | Mesh resolution details |
| Input field characteristics | Time stepping internals |
| Output characteristics | Convergence criteria |

The goal is a clean, intuitive API where users focus on the *physics* they want to learn, not the *numerics* used to generate it.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/pdeforge.git
   cd pdeforge
   ```
3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

### Running Tests

```bash
pytest tests/
```

### Code Formatting

We use Black for code formatting and isort for import sorting:

```bash
black pdeforge/ tests/
isort pdeforge/ tests/
```

### Type Checking

```bash
mypy pdeforge/
```

## Adding a New PDE Model

PDEForge uses a registry pattern that makes adding new models straightforward.

### Step 1: Create the Model File

Create a new file in `pdeforge/models/`, e.g., `pdeforge/models/heat_2d.py`:

```python
"""
2D Heat Equation - Thermal diffusion on a periodic domain.

Maps initial temperature fields to solutions at a fixed final time.
Use this for learning diffusion operators.
"""

import numpy as np
from typing import Dict, Tuple, Union, Callable

from pdeforge.core.base import PDEModel
from pdeforge.core.registry import register_model
from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.generators.initial_conditions import get_ic_generator


@register_model("heat_2d")
class Heat2D(PDEModel):
    """
    2D Heat equation for thermal diffusion.
    
    ∂T/∂t = α ∇²T
    
    This model maps initial temperature to solution at final time:
    T(x,y,t=0) → T(x,y,t=τ)
    
    Use this for learning linear diffusion operators.
    """
    
    NDIM = 2
    INPUT_NAMES = ["T0"]
    OUTPUT_NAMES = ["T_final"]
    
    # USER-FACING PARAMETERS ONLY
    # These are what users can configure via params={}
    USER_PARAMS = [
        ParamSpec(
            name="diffusivity",
            description="Thermal diffusivity coefficient",
            default=0.01,
            param_type=ParamType.PHYSICAL,
            bounds=(1e-4, 1.0),
            units="m²/s",
            affects="Higher values → faster smoothing",
        ),
        ParamSpec(
            name="time_horizon",
            description="Final time for solution",
            default=1.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.01, 10.0),
            units="s",
            affects="Longer times → smoother solutions",
        ),
    ]
    
    # ALL DEFAULTS (user-facing + internal)
    DEFAULT_PARAMS = {
        # User-facing (documented in USER_PARAMS)
        "diffusivity": 0.01,
        "time_horizon": 1.0,
        # Internal (hidden from users, prefixed with _)
        "_n_time_steps": 101,
        "_solver_tol": 1e-8,
    }
    
    def __init__(self, resolution, domain=None, **params):
        super().__init__(resolution, domain, **params)
        # Extract user-facing params
        self.alpha = self.params["diffusivity"]
        self.T_final = self.params["time_horizon"]
        # Internal params (users shouldn't need to touch these)
        self._n_steps = self.params.get("_n_time_steps", 101)
        
    def solve(self, ic, **kwargs):
        """Solve the heat equation."""
        # Implementation here...
        return solution
    
    def generate_ic(self, generator="fourier", generator_params=None, seed=None):
        """Generate random initial temperature field."""
        if generator_params is None:
            generator_params = {}
        gen = get_ic_generator(generator, **generator_params)
        return gen.generate(shape=(self.ny, self.nx), seed=seed, grid=self.grids)
    
    def validate_solution(self, ic, solution, tol=1e-6):
        """Validate the solution."""
        return {'valid': not np.isnan(solution).any()}
```

### Key Points:

1. **USER_PARAMS**: Define only parameters users should configure
2. **DEFAULT_PARAMS**: Include both user-facing and internal (prefixed with `_`)
3. **Docstring**: Focus on what ML task this enables, not solver details

### Step 2: Register the Model

Add an import in `pdeforge/models/__init__.py`:

```python
from pdeforge.models import heat_2d
from pdeforge.models.heat_2d import Heat2D

__all__ = [..., "Heat2D"]
```

### Step 3: Add Tests

Create `tests/test_heat_2d.py`:

```python
import pytest
import numpy as np
from pdeforge import generate_dataset, get_model

def test_heat_2d_basic():
    """Test basic dataset generation."""
    dataset = generate_dataset(
        model="heat_2d",
        n_samples=10,
        resolution={"x": 32, "y": 32},
        seed=42,
    )
    
    assert dataset.n_samples == 10
    assert dataset.inputs.shape == (10, 32, 32)
    assert dataset.outputs.shape == (10, 32, 32)

def test_heat_2d_validation():
    """Test that generated solutions are valid."""
    model = get_model("heat_2d")(resolution={"x": 32, "y": 32})
    
    ic, solution, info = model.generate_sample(seed=42)
    
    assert info['valid']
    assert not np.isnan(solution).any()
```

### Step 4: Update Documentation

Add your model to the README.md and any relevant documentation.

## Guidelines for PDE Models

### Solver Requirements

1. **Spectral methods preferred**: For periodic domains, use FFT-based spectral methods for accuracy and efficiency.

2. **Validation**: Implement `validate_solution()` to check:
   - No NaN/Inf values
   - Residual is below tolerance
   - Physical constraints are satisfied

3. **Reproducibility**: Always use the provided seed for random number generation.

### Code Style

1. Follow PEP 8 and use Black formatting
2. Add comprehensive docstrings (NumPy style)
3. Include type hints
4. Write unit tests for all functionality

### Performance

1. Avoid loops over grid points when possible (use vectorized operations)
2. Precompute wavenumbers and grids in `__init__`
3. Use scipy's optimized solvers when appropriate

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/my-new-model`
2. Make your changes
3. Run tests: `pytest tests/`
4. Format code: `black . && isort .`
5. Commit with descriptive message
6. Push and create a Pull Request

## Questions?

Open an issue on GitHub or start a discussion!
