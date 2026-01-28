# API Reference

## Core Functions

### generate_dataset

```python
pdeforge.generate_dataset(
    model: str,
    n_samples: int,
    resolution: dict,
    domain: dict = None,
    params: dict = None,
    ic_generator: str = "fourier",
    ic_params: dict = None,
    seed: int = None,
    validate: bool = True,
    n_jobs: int = 1,
    verbose: bool = True,
) -> PDEDataset
```

Generate a dataset from a PDE model.

**Parameters:**

- `model`: Name of the registered PDE model
- `n_samples`: Number of samples to generate
- `resolution`: Grid resolution per dimension, e.g., `{"x": 256}`
- `domain`: Domain bounds per dimension, e.g., `{"x": (0, 1)}`
- `params`: Model-specific physical parameters
- `ic_generator`: Initial condition generator type
- `ic_params`: Parameters for IC generator
- `seed`: Random seed for reproducibility
- `validate`: Whether to validate generated solutions
- `n_jobs`: Number of parallel workers
- `verbose`: Show progress bar

**Returns:** `PDEDataset` object

---

### list_models

```python
pdeforge.list_models() -> List[str]
```

Return names of all registered PDE models.

---

### describe_model

```python
pdeforge.describe_model(name: str) -> str
```

Return detailed description of a model including configurable parameters.

---

### get_model

```python
pdeforge.get_model(name: str) -> Type[PDEModel]
```

Return the model class for direct instantiation.

---

### load_dataset

```python
pdeforge.load_dataset(path: str) -> PDEDataset
```

Load a saved dataset from directory, NPZ, or HDF5 file.

---

## PDEDataset Class

### Attributes

- `inputs`: Input fields as ndarray
- `outputs`: Output fields as ndarray
- `grid`: Dictionary of spatial coordinate arrays
- `metadata`: Dictionary with model name, parameters, etc.

### Methods

#### split

```python
dataset.split(
    train: float = 0.7,
    val: float = 0.15,
    cal: float = 0.0,
    test: float = 0.15,
    seed: int = None,
) -> Dict[str, PDEDataset]
```

Split dataset into train/val/cal/test subsets.

#### save

```python
dataset.save(path: str)
```

Save dataset. Format determined by file extension:

- Directory (no extension): Creates directory with NPY files
- `.npz`: Compressed NumPy archive
- `.h5`: HDF5 file (requires h5py)

#### visualize

```python
dataset.visualize()
```

Interactive visualization widget (Jupyter only).

---

## Exploration Functions

### explore_parameter

```python
pdeforge.explore_parameter(
    model: str,
    param_name: str,
    param_values: List,
    resolution: dict,
    n_samples_per_value: int = 3,
    seed: int = None,
) -> PDEDataset
```

Generate samples varying a single parameter.

### explore_parameter_grid

```python
pdeforge.explore_parameter_grid(
    model: str,
    param_grid: Dict[str, List],
    resolution: dict,
    n_samples_per_combo: int = 1,
) -> PDEDataset
```

Generate samples for all parameter combinations.

### explore_model

```python
pdeforge.explore_model(
    model: str,
    resolution: dict,
) -> Dict[str, PDEDataset]
```

Explore all user-facing parameters of a model.

---

## IC Generators

### FourierICGenerator

Random Fourier series with decaying coefficients.

```python
gen = FourierICGenerator(
    n_modes: int = 10,
    decay: float = 1.5,
    amplitude: float = 1.0,
    use_cos: bool = True,
)
```

### GaussianRandomFieldGenerator

Gaussian random field with specified spectral decay.

```python
gen = GaussianRandomFieldGenerator(
    alpha: float = 2.0,
    amplitude: float = 1.0,
)
```

### SigmoidTransformGenerator

Wraps another generator and applies sigmoid transform to bound values.

```python
gen = SigmoidTransformGenerator(
    u_min: float,
    u_max: float,
    base_generator: ICGenerator,
)
```
