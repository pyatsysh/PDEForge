# Stochastic Systems in PDEForge

This document outlines the design for stochastic ODE/PDE support in PDEForge.

## Motivation

Stochastic systems are essential for:

1. **Uncertainty Quantification**: Real systems have noise; neural operators should quantify uncertainty
2. **Generative Modeling**: Learn to sample from conditional distributions
3. **Robustness**: Models trained on stochastic data may generalize better

## The Core Challenge

**Deterministic PDE:**
```
f(input) = output  (unique)
```

**Stochastic PDE:**
```
f(input, ω) = output(ω)  where ω ~ noise process
```

The same input produces a *distribution* of outputs. What should the learning task be?

## Two Output Formats

We support both use cases with different output formats:

### Format 1: Multiple Realizations (Sample-Based)

```python
dataset = generate_dataset(
    "stochastic_heat_2d",
    n_samples=1000,
    params={"n_realizations": 50},
)

# Shapes:
# dataset.inputs.shape = (1000, nx, ny)        # One IC per sample
# dataset.outputs.shape = (1000, 50, nx, ny)   # 50 realizations per IC
```

**Use case**: Generative models, score-based diffusion, conditional VAEs

**Learning task**: Learn p(output | input)

### Format 2: Moments (Statistics-Based)

```python
dataset = generate_dataset(
    "stochastic_heat_2d", 
    n_samples=1000,
    params={"output_moments": True, "n_realizations": 100},
)

# Shapes:
# dataset.inputs.shape = (1000, nx, ny)
# dataset.output_mean.shape = (1000, nx, ny)
# dataset.output_var.shape = (1000, nx, ny)
# Optionally: dataset.output_skew, dataset.output_kurt
```

**Use case**: Uncertainty quantification, mean + variance prediction

**Learning task**: Learn E[output | input] and Var[output | input]

## Planned Stochastic Models

### 1. Stochastic Heat Equation

$$\frac{\partial u}{\partial t} = \alpha \nabla^2 u + \sigma \dot{W}$$

- Additive space-time white noise
- Linear, well-understood
- Good first test case

**Parameters:**
- `diffusivity` (α): Thermal diffusivity
- `noise_intensity` (σ): Noise strength
- `noise_correlation_length`: Spatial correlation of noise (0 = white)

### 2. Stochastic Burgers Equation

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2} + \sigma \dot{W}$$

- Nonlinear with noise
- Tests interaction of shocks with randomness

**Parameters:**
- `viscosity` (ν)
- `noise_intensity` (σ)
- `noise_type`: "additive" or "multiplicative"

### 3. Stochastic Allen-Cahn

$$\frac{\partial u}{\partial t} = \epsilon \nabla^2 u + u - u^3 + \sigma \dot{W}$$

- Phase separation with fluctuations
- Bistable dynamics

## Noise Representation

### Generating Correlated Noise

For spatially correlated noise, we use spectral methods:

```python
def generate_gaussian_noise_field(shape, correlation_length, dt, seed):
    """Generate Gaussian random field with given correlation."""
    # In Fourier space, apply correlation kernel
    # Q(k) ∝ exp(-|k|² * correlation_length²)
    ...
```

### Noise Input to Neural Operator

For generative models, include noise as input:

```python
# Option A: Explicit noise field
inputs = stack([ic, noise_realization], axis=-1)

# Option B: Latent code (for VAE-style models)
inputs = stack([ic, z], axis=-1)  # z ~ N(0, I)
```

## Implementation Plan

### Phase 1: Data Structures

Extend `PDEDataset` to handle stochastic outputs:

```python
@dataclass
class StochasticPDEDataset(PDEDataset):
    """Dataset for stochastic PDE outputs."""
    
    # For sample-based format
    outputs: np.ndarray  # Shape: (n_samples, n_realizations, *spatial, n_channels)
    
    # For moment-based format (optional)
    output_mean: np.ndarray = None
    output_var: np.ndarray = None
    
    @property
    def n_realizations(self) -> int:
        return self.outputs.shape[1]
    
    def compute_moments(self) -> None:
        """Compute moments from realizations."""
        self.output_mean = self.outputs.mean(axis=1)
        self.output_var = self.outputs.var(axis=1)
    
    def get_realization(self, sample_idx: int, realization_idx: int):
        """Get a specific realization."""
        return self.outputs[sample_idx, realization_idx]
```

### Phase 2: Base Class

```python
class StochasticPDEModel(PDEModel):
    """Base class for stochastic PDE models."""
    
    DEFAULT_PARAMS = {
        "noise_intensity": 0.1,
        "n_realizations": 50,
        "output_moments": False,
    }
    
    @abstractmethod
    def generate_noise(self, shape, seed) -> np.ndarray:
        """Generate noise realization."""
        pass
    
    def solve(self, ic, noise=None, seed=None):
        """Solve with given or generated noise."""
        if noise is None:
            noise = self.generate_noise(ic.shape, seed)
        return self._solve_with_noise(ic, noise)
    
    @abstractmethod
    def _solve_with_noise(self, ic, noise) -> np.ndarray:
        """Solve the SPDE with specific noise realization."""
        pass
```

### Phase 3: Implement Models

1. `stochastic_heat_2d` - Linear, additive noise
2. `stochastic_burgers_1d` - Nonlinear, test shock-noise interaction
3. `stochastic_allen_cahn_2d` - Bistable dynamics

## Numerical Methods

### Time Integration

For SPDEs, we use:

1. **Euler-Maruyama**: Simple, O(Δt^0.5) for noise term
2. **Milstein**: O(Δt) for multiplicative noise
3. **Exponential integrators**: Better for stiff + noise

### Spatial Discretization

Spectral methods work well for:
- Smooth noise fields
- Periodic boundaries
- Fast evaluation of nonlinear terms

## Visualization

Extend visualization for stochastic data:

```python
def visualize_stochastic_sample(dataset, sample_idx):
    """Show realizations and statistics for one sample."""
    # Row 1: Input IC
    # Row 2: Several output realizations  
    # Row 3: Mean and std fields
```

## Validation

For stochastic models, validate:

1. **Moment convergence**: Mean/var stabilize with more realizations
2. **Path regularity**: Solutions have expected smoothness
3. **Known statistics**: For linear SPDEs, compare to analytical formulas
