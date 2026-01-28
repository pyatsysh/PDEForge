# Adding Custom Models

PDEForge uses a registry pattern that makes adding new PDE models straightforward.

## Basic Structure

Create a new file in `pdeforge/models/`:

```python
# pdeforge/models/my_pde.py

import numpy as np
from pdeforge.core.base import PDEModel
from pdeforge.core.registry import register_model

@register_model("my_pde")
class MyPDE(PDEModel):
    """Description of your PDE model."""

    NDIM = 2  # Spatial dimensions
    DEFAULT_PARAMS = {
        "param1": 1.0,
        "param2": 0.5,
    }
    INPUT_NAMES = ["input_field"]
    OUTPUT_NAMES = ["solution"]

    def __init__(self, resolution, domain=None, **params):
        super().__init__(resolution, domain, **params)
        # Initialize solver state
        self._setup_solver()

    def _setup_solver(self):
        """Prepare discretization, operators, etc."""
        pass

    def solve(self, ic):
        """
        Solve the PDE given initial/input condition.

        Parameters
        ----------
        ic : ndarray
            Input field

        Returns
        -------
        ndarray
            Solution field
        """
        # Your solver implementation
        return solution

    def generate_ic(self, generator="fourier", generator_params=None, seed=None):
        """
        Generate random input field.

        Returns
        -------
        ndarray
            Random initial condition
        """
        from pdeforge.generators.initial_conditions import get_ic_generator

        if generator_params is None:
            generator_params = {}

        gen = get_ic_generator(generator, **generator_params)
        return gen.generate(
            shape=self._get_shape(),
            seed=seed,
            grid=self.grids,
        )

    def _get_shape(self):
        """Return spatial shape for IC generation."""
        return tuple(self.resolution[d] for d in sorted(self.resolution.keys()))
```

## Registration

The `@register_model("name")` decorator automatically registers your model. Make sure to import the module in `pdeforge/models/__init__.py`:

```python
# pdeforge/models/__init__.py
from . import my_pde
```

## Exposing Parameters

To make parameters discoverable via `describe_model()`:

```python
from pdeforge.core.params import ParamSpec, ParamType

@register_model("my_pde")
class MyPDE(PDEModel):

    USER_PARAMS = [
        ParamSpec(
            name="param1",
            description="Controls the strength of diffusion",
            default=1.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.001, 10.0),
            units="m^2/s",
            affects="Higher values smooth the solution",
        ),
    ]
```

## Using Built-in IC Generators

PDEForge provides several initial condition generators:

```python
from pdeforge.generators.initial_conditions import (
    FourierICGenerator,
    GaussianRandomFieldGenerator,
    SigmoidTransformGenerator,
)

# Fourier series with random coefficients
gen = FourierICGenerator(n_modes=10, decay=1.5)

# Gaussian random field
gen = GaussianRandomFieldGenerator(alpha=2.0, amplitude=1.0)

# Transform to bounded range
gen = SigmoidTransformGenerator(
    u_min=0.1,
    u_max=10.0,
    base_generator=GaussianRandomFieldGenerator(),
)
```

## FEniCSx Models

For models requiring finite elements, inherit from `FEniCSModel`:

```python
from pdeforge.core.fenics_base import FEniCSModel

@register_model("my_fem_model")
class MyFEMModel(FEniCSModel):

    def create_mesh(self):
        """Return a dolfinx mesh."""
        pass

    def create_function_spaces(self):
        """Set up FEM function spaces."""
        pass

    def solve(self, input_field):
        """Solve and return interpolated solution."""
        pass
```

See `cylinder_flow_2d.py` for a complete example.

## Testing

Add tests in `tests/test_models.py`:

```python
def test_my_pde():
    from pdeforge import generate_dataset

    dataset = generate_dataset(
        model="my_pde",
        n_samples=5,
        resolution={"x": 32, "y": 32},
    )

    assert dataset.inputs.shape[0] == 5
    assert not np.isnan(dataset.outputs).any()
```
