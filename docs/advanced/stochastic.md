# Stochastic Systems

Support for stochastic PDEs is planned for future PDEForge releases.

## Motivation

Many physical systems involve inherent randomness:

- Turbulence in fluid flows
- Thermal fluctuations
- Material heterogeneity
- Measurement noise

Training neural operators on deterministic data alone may not capture these uncertainties.

## Planned Models

| Model | Description |
|-------|-------------|
| `stochastic_heat_2d` | Heat equation with additive noise |
| `stochastic_burgers_1d` | Burgers equation with forcing noise |
| `stochastic_allen_cahn_2d` | Phase separation with fluctuations |

## Output Formats

### Multiple Realizations

Generate multiple noise realizations per input:

```python
dataset = generate_dataset(
    model="stochastic_heat_2d",
    n_samples=100,
    params={"n_realizations": 50},
)
# dataset.outputs.shape = (100, 50, nx, ny)
```

Useful for training generative models that learn the full output distribution.

### Moment-Based

Compute mean and variance across realizations:

```python
dataset = generate_dataset(
    model="stochastic_heat_2d",
    n_samples=100,
    params={"output_moments": True, "n_realizations": 100},
)
# dataset.output_mean.shape = (100, nx, ny)
# dataset.output_var.shape = (100, nx, ny)
```

Useful for training models that directly predict uncertainty.

## Integration with Uncertainty Quantification

Stochastic systems provide natural test cases for UQ methods:

- Compare learned uncertainty against true output variance
- Validate coverage guarantees on systems with known noise
- Benchmark different UQ approaches

## Status

This feature is under development. See the GitHub repository for updates.
