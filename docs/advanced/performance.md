# Performance

PDEForge uses NumPy and SciPy for broad compatibility. For most research use cases, this is sufficient.

## Typical Performance

| Dataset Size | Spectral Models | FEniCSx Models |
|--------------|-----------------|----------------|
| 100 samples | seconds | minutes |
| 1,000 samples | minutes | tens of minutes |
| 10,000 samples | hours | not recommended |

## Recommendations

### Generate Once, Reuse

For training datasets, generate once and save:

```python
dataset = generate_dataset("burgers_1d", n_samples=10000, ...)
dataset.save("./burgers_training_data")

# Later
dataset = load_dataset("./burgers_training_data")
```

### Start Small

Begin with small datasets for development:

```python
# Development
small = generate_dataset(model, n_samples=100, resolution={"x": 64})

# Production
large = generate_dataset(model, n_samples=10000, resolution={"x": 256})
```

### Reduce Resolution for Exploration

When exploring parameters:

```python
# Quick exploration at low resolution
explore_parameter(..., resolution={"x": 64, "y": 64})

# Final dataset at target resolution
generate_dataset(..., resolution={"x": 256, "y": 256})
```

### FEniCSx Models

For cylinder flow and other FEniCSx models:

- Keep sample counts modest (50-200 samples)
- Use coarser output grids when possible
- The mesh resolution (`_mesh_resolution`) affects solve time significantly

## Parallel Generation

Some speedup is possible via parallel workers:

```python
dataset = generate_dataset(..., n_jobs=4)
```

This parallelizes sample generation. Speedup depends on solver overhead and system resources.

## Memory Considerations

Large 2D datasets can consume significant memory:

```
n_samples=1000, resolution=(256, 256), 3 output channels
= 1000 * 256 * 256 * 3 * 8 bytes
= ~1.5 GB
```

For very large datasets:

1. Generate in batches
2. Save each batch to disk
3. Concatenate when loading

## Future: GPU Acceleration

A JAX-based backend for spectral models is planned for future releases. This would enable GPU-accelerated data generation for models like Burgers and Darcy.
