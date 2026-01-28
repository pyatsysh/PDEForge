# Performance Guide

PDEForge prioritizes **simplicity and correctness** over raw speed. This document explains our design choices and provides guidance for users with different needs.

## Current Backend: NumPy + SciPy

All spectral models use NumPy for FFT operations and SciPy for time integration and linear solvers. This choice was deliberate:

| Reason | Explanation |
|--------|-------------|
| **Universal availability** | NumPy/SciPy are installed everywhere |
| **No configuration** | Works out of the box on any system |
| **Correctness first** | Well-tested, numerically stable implementations |
| **Easy debugging** | Standard Python tools work seamlessly |

## Performance Expectations

Typical generation times on a modern laptop (single core):

| Model | Resolution | Time per sample | 1000 samples |
|-------|------------|-----------------|--------------|
| `burgers_1d` | 256 | ~0.3s | ~5 min |
| `stokes_2d` | 64×64 | ~0.01s | ~10 sec |
| `darcy_2d` | 64×64 | ~0.05s | ~1 min |

**Bottlenecks by model:**
- `burgers_1d`: Time integration (SciPy's `odeint`) - sequential, hard to parallelize
- `darcy_2d`: Conjugate gradient solver - iterative, depends on condition number
- `stokes_2d`: Direct FFT solve - fast, scales well

## When is Performance a Concern?

For most operator learning research, you need:
- **Exploratory work**: 100-1000 samples → Current speed is fine
- **Full training**: 10,000+ samples → Consider overnight generation or optimization

**Our recommendation**: Generate your dataset once, save it, reuse it:

```python
# Generate once (can take time for large datasets)
dataset = generate_dataset("burgers_1d", n_samples=10000, resolution={"x": 256})
dataset.save("./burgers_training_data")

# Load instantly for experiments
from pdeforge import load_dataset
dataset = load_dataset("./burgers_training_data")
```

## Future: JAX Backend (Planned)

We plan to add an optional JAX backend for users who need maximum performance:

```python
# Future API (not yet implemented)
import pdeforge
pdeforge.set_backend("jax")  # Enable JAX acceleration

# Or per-call
dataset = generate_dataset(..., backend="jax")
```

**Expected speedups with JAX:**
- JIT compilation: 2-10× on CPU
- GPU acceleration: 10-100× for large grids
- Batch generation: Generate many samples in parallel

**Why not JAX by default?**
- Installation can be tricky (CUDA versions, platform differences)
- Adds complexity for users who don't need speed
- Debugging is harder with JIT-compiled code
- Not everyone has a GPU

## Practical Tips for Current Version

### 1. Use Coarser Resolution During Development

```python
# Development/debugging (fast)
dataset = generate_dataset("burgers_1d", n_samples=100, resolution={"x": 64})

# Final dataset (slower but higher quality)  
dataset = generate_dataset("burgers_1d", n_samples=10000, resolution={"x": 256})
```

### 2. Generate in Batches

```python
# For very large datasets, generate in chunks
from pdeforge import generate_dataset, load_dataset
import numpy as np

for i in range(10):
    chunk = generate_dataset(
        "burgers_1d", 
        n_samples=1000,
        resolution={"x": 256},
        seed=i * 1000,  # Different seed per chunk
    )
    chunk.save(f"./data/chunk_{i}")

# Combine later if needed
```

### 3. Use Exploration First

Before generating large datasets, use exploration to understand the physics:

```python
from pdeforge import explore_parameter

# Quick exploration (few samples)
dataset = explore_parameter(
    "burgers_1d",
    param_name="viscosity",
    param_values=[0.001, 0.01, 0.1],
    resolution={"x": 64},  # Coarse for speed
    n_samples_per_value=3,
)
```

### 4. Consider Overnight Generation

For publication-quality datasets:

```python
# run_generation.py
from pdeforge import generate_dataset

dataset = generate_dataset(
    "burgers_1d",
    n_samples=50000,
    resolution={"x": 512},
    seed=42,
)
dataset.save("./publication_dataset")
print("Done!")
```

```bash
# Run overnight
nohup python run_generation.py > generation.log 2>&1 &
```

## Contributing Performance Improvements

If you'd like to help implement the JAX backend or other optimizations:

1. See `CONTRIBUTING.md` for development setup
2. Key files: `pdeforge/solvers/spectral.py`, individual model files
3. Maintain NumPy fallback for users without JAX
4. Add benchmarks to `tests/` to track performance

We welcome PRs that improve performance while maintaining our API!
