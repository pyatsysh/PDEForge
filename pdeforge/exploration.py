"""
Parameter exploration utilities for PDEForge.

This module provides tools for exploring how model hyperparameters
affect the generated data. This is valuable for:

1. Building intuition about the physics
2. Choosing appropriate parameter ranges for training
3. Understanding model generalization requirements
4. Debugging when models fail on certain parameter ranges

Usage:
    >>> from pdeforge.exploration import explore_parameter, explore_model
    >>> 
    >>> # See how viscosity affects Burgers solutions
    >>> dataset = explore_parameter(
    ...     model="burgers_1d",
    ...     param_name="viscosity",
    ...     param_values=[0.001, 0.01, 0.1],
    ...     resolution={"x": 128},
    ... )
    >>> dataset.visualize()
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from itertools import product
from tqdm import tqdm

from pdeforge.core.registry import get_model
from pdeforge.core.types import PDEDataset


def explore_parameter(
    model: str,
    param_name: str,
    param_values: List[Any],
    resolution: Dict[str, int],
    n_samples_per_value: int = 3,
    base_params: Dict = None,
    seed: int = 42,
    verbose: bool = True,
) -> PDEDataset:
    """
    Explore how a single parameter affects the model outputs.
    
    Generates samples at different parameter values to help build
    intuition about the physics.
    
    Parameters
    ----------
    model : str
        Model name (e.g., "burgers_1d")
    param_name : str
        Parameter to vary (must be in USER_PARAMS)
    param_values : List[Any]
        Values to sample
    resolution : Dict[str, int]
        Output grid resolution
    n_samples_per_value : int
        Number of samples per parameter value (same IC, different params)
    base_params : Dict, optional
        Base parameters to use (param_name will be overridden)
    seed : int
        Random seed for reproducibility
    verbose : bool
        Show progress bar
        
    Returns
    -------
    PDEDataset
        Dataset with metadata containing parameter values for each sample.
        Access via dataset.metadata['param_values']
        
    Examples
    --------
    >>> dataset = explore_parameter(
    ...     model="burgers_1d",
    ...     param_name="viscosity",
    ...     param_values=[0.001, 0.01, 0.1],
    ...     resolution={"x": 128},
    ... )
    >>> # Each group of n_samples_per_value uses the same IC
    >>> # but different viscosity values
    """
    if base_params is None:
        base_params = {}
    
    model_cls = get_model(model)
    
    # Validate parameter name
    user_param_names = [p.name for p in model_cls.USER_PARAMS]
    if param_name not in user_param_names and param_name not in model_cls.DEFAULT_PARAMS:
        raise ValueError(
            f"Parameter '{param_name}' not found. "
            f"Available user parameters: {user_param_names}"
        )
    
    inputs = []
    outputs = []
    param_values_list = []
    ic_indices = []  # Track which IC was used
    
    # Generate ICs first
    np.random.seed(seed)
    ic_seeds = [np.random.randint(0, 2**31) for _ in range(n_samples_per_value)]
    
    # Create a reference model for IC generation
    ref_params = {**base_params, param_name: param_values[0]}
    ref_model = model_cls(resolution=resolution, **ref_params)
    
    # Generate ICs
    ics = []
    for ic_seed in ic_seeds:
        ic = ref_model.generate_ic(seed=ic_seed)
        ics.append(ic)
    
    # Now solve for each parameter value with each IC
    iterator = list(product(range(len(ics)), param_values))
    if verbose:
        iterator = tqdm(iterator, desc=f"Exploring {param_name}")
    
    for ic_idx, param_val in iterator:
        # Create model with this parameter value
        params = {**base_params, param_name: param_val}
        pde_model = model_cls(resolution=resolution, **params)
        
        # Use the pre-generated IC
        ic = ics[ic_idx]
        
        # Solve
        try:
            solution = pde_model.solve(ic)
            inputs.append(ic)
            outputs.append(solution)
            param_values_list.append(param_val)
            ic_indices.append(ic_idx)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed for {param_name}={param_val}: {e}")
    
    # Stack arrays
    inputs = np.stack(inputs, axis=0)
    outputs = np.stack(outputs, axis=0)
    
    # Create metadata
    metadata = {
        'model': model,
        'exploration_type': 'single_parameter',
        'param_name': param_name,
        'param_values': param_values_list,
        'ic_indices': ic_indices,
        'unique_param_values': list(param_values),
        'n_samples_per_value': n_samples_per_value,
        'base_params': base_params,
        'resolution': dict(resolution),
        'seed': seed,
    }
    
    return PDEDataset(
        inputs=inputs,
        outputs=outputs,
        grid=ref_model.grids.copy(),
        metadata=metadata,
        input_names=model_cls.INPUT_NAMES,
        output_names=model_cls.OUTPUT_NAMES,
    )


def explore_parameter_grid(
    model: str,
    param_grid: Dict[str, List[Any]],
    resolution: Dict[str, int],
    n_samples_per_config: int = 1,
    base_params: Dict = None,
    seed: int = 42,
    verbose: bool = True,
) -> PDEDataset:
    """
    Explore a grid of parameter combinations.
    
    Generates samples for all combinations of parameter values.
    Useful for understanding interactions between parameters.
    
    Parameters
    ----------
    model : str
        Model name
    param_grid : Dict[str, List[Any]]
        Dictionary mapping parameter names to lists of values
        E.g., {"viscosity": [0.01, 0.1], "time_horizon": [0.5, 1.0]}
    resolution : Dict[str, int]
        Output grid resolution
    n_samples_per_config : int
        Number of samples per parameter configuration
    base_params : Dict, optional
        Base parameters (will be overridden by param_grid)
    seed : int
        Random seed
    verbose : bool
        Show progress
        
    Returns
    -------
    PDEDataset
        Dataset with metadata containing parameter grid info
        
    Examples
    --------
    >>> dataset = explore_parameter_grid(
    ...     model="burgers_1d",
    ...     param_grid={
    ...         "viscosity": [0.001, 0.01, 0.1],
    ...         "time_horizon": [0.5, 1.0, 2.0],
    ...     },
    ...     resolution={"x": 128},
    ... )
    """
    if base_params is None:
        base_params = {}
    
    model_cls = get_model(model)
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_value_lists = [param_grid[name] for name in param_names]
    all_combinations = list(product(*param_value_lists))
    
    inputs = []
    outputs = []
    param_configs = []
    
    np.random.seed(seed)
    
    iterator = all_combinations
    if verbose:
        iterator = tqdm(iterator, desc="Exploring parameter grid")
    
    for combination in iterator:
        # Build params dict
        params = {**base_params}
        config = {}
        for name, value in zip(param_names, combination):
            params[name] = value
            config[name] = value
        
        # Create model
        pde_model = model_cls(resolution=resolution, **params)
        
        # Generate samples for this configuration
        for _ in range(n_samples_per_config):
            sample_seed = np.random.randint(0, 2**31)
            try:
                ic, solution, _ = pde_model.generate_sample(seed=sample_seed)
                inputs.append(ic)
                outputs.append(solution)
                param_configs.append(config.copy())
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed for {config}: {e}")
    
    # Stack arrays
    inputs = np.stack(inputs, axis=0)
    outputs = np.stack(outputs, axis=0)
    
    # Create reference model for grids
    ref_params = {**base_params, **dict(zip(param_names, all_combinations[0]))}
    ref_model = model_cls(resolution=resolution, **ref_params)
    
    metadata = {
        'model': model,
        'exploration_type': 'parameter_grid',
        'param_grid': param_grid,
        'param_configs': param_configs,
        'n_samples_per_config': n_samples_per_config,
        'base_params': base_params,
        'resolution': dict(resolution),
        'seed': seed,
    }
    
    return PDEDataset(
        inputs=inputs,
        outputs=outputs,
        grid=ref_model.grids.copy(),
        metadata=metadata,
        input_names=model_cls.INPUT_NAMES,
        output_names=model_cls.OUTPUT_NAMES,
    )


def explore_model(
    model: str,
    resolution: Dict[str, int],
    n_values_per_param: int = 3,
    n_samples_per_value: int = 2,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, PDEDataset]:
    """
    Explore all user-facing parameters of a model.
    
    For each parameter in USER_PARAMS, generates samples spanning
    the parameter's valid range. Returns a dictionary of datasets,
    one per parameter.
    
    Parameters
    ----------
    model : str
        Model name
    resolution : Dict[str, int]
        Output grid resolution
    n_values_per_param : int
        Number of values to sample for each parameter
    n_samples_per_value : int
        Number of samples per parameter value
    seed : int
        Random seed
    verbose : bool
        Show progress
        
    Returns
    -------
    Dict[str, PDEDataset]
        Dictionary mapping parameter names to exploration datasets
        
    Examples
    --------
    >>> explorations = explore_model("burgers_1d", resolution={"x": 128})
    >>> 
    >>> # Visualize effect of each parameter
    >>> for param_name, dataset in explorations.items():
    ...     print(f"\\n=== {param_name} ===")
    ...     dataset.visualize()
    """
    model_cls = get_model(model)
    
    if not model_cls.USER_PARAMS:
        raise ValueError(f"Model '{model}' has no USER_PARAMS defined")
    
    results = {}
    
    for param_spec in model_cls.USER_PARAMS:
        if verbose:
            print(f"Exploring: {param_spec.name}")
        
        # Generate parameter values
        if param_spec.choices is not None:
            # Discrete parameter
            values = param_spec.choices[:n_values_per_param]
        elif param_spec.bounds is not None:
            # Continuous parameter - sample in log space if range spans orders of magnitude
            low, high = param_spec.bounds
            if high / low > 100:
                # Log scale
                values = np.logspace(
                    np.log10(low), np.log10(high), n_values_per_param
                ).tolist()
            else:
                # Linear scale
                values = np.linspace(low, high, n_values_per_param).tolist()
        else:
            # No bounds - use default with small variations
            default = param_spec.default
            if isinstance(default, (int, float)):
                values = [default * 0.5, default, default * 2.0]
            else:
                values = [default]
        
        # Explore this parameter
        dataset = explore_parameter(
            model=model,
            param_name=param_spec.name,
            param_values=values,
            resolution=resolution,
            n_samples_per_value=n_samples_per_value,
            seed=seed,
            verbose=verbose,
        )
        
        results[param_spec.name] = dataset
    
    return results


def visualize_parameter_effect(
    dataset: PDEDataset,
    figsize: Tuple[int, int] = None,
    save_path: str = None,
):
    """
    Visualize how a parameter affects model outputs.
    
    Creates a grid of plots showing solutions at different parameter values.
    Works with datasets created by explore_parameter().
    
    Parameters
    ----------
    dataset : PDEDataset
        Dataset from explore_parameter()
    figsize : Tuple[int, int], optional
        Figure size
    save_path : str, optional
        If provided, save figure to this path
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    if dataset.metadata.get('exploration_type') != 'single_parameter':
        raise ValueError("This function requires a dataset from explore_parameter()")
    
    param_name = dataset.metadata['param_name']
    param_values = dataset.metadata['param_values']
    ic_indices = dataset.metadata['ic_indices']
    unique_values = dataset.metadata['unique_param_values']
    n_per_value = dataset.metadata['n_samples_per_value']
    
    # Determine dimensionality
    ndim = len(dataset.input_shape)
    is_1d = ndim == 1 or (ndim == 2 and dataset.input_shape[-1] < 10)
    
    if is_1d:
        _visualize_1d_parameter_effect(
            dataset, param_name, unique_values, n_per_value, figsize, save_path
        )
    else:
        _visualize_2d_parameter_effect(
            dataset, param_name, unique_values, n_per_value, figsize, save_path
        )


def _visualize_1d_parameter_effect(
    dataset, param_name, unique_values, n_per_value, figsize, save_path
):
    """Visualize parameter effect for 1D models."""
    import matplotlib.pyplot as plt
    
    x = dataset.grid.get('x', np.arange(dataset.input_shape[0]))
    
    n_values = len(unique_values)
    if figsize is None:
        figsize = (12, 3 * n_per_value)
    
    fig, axes = plt.subplots(n_per_value, 2, figsize=figsize)
    if n_per_value == 1:
        axes = axes.reshape(1, -1)
    
    # Color map for parameter values
    colors = plt.cm.viridis(np.linspace(0, 1, n_values))
    
    for ic_idx in range(n_per_value):
        # Plot input (same for all param values)
        sample_idx = ic_idx * n_values
        axes[ic_idx, 0].plot(x, dataset.inputs[sample_idx], 'k-', linewidth=2)
        axes[ic_idx, 0].set_title(f'Input (IC #{ic_idx + 1})')
        axes[ic_idx, 0].set_xlabel('x')
        axes[ic_idx, 0].grid(True, alpha=0.3)
        
        # Plot outputs for each parameter value
        for val_idx, val in enumerate(unique_values):
            sample_idx = ic_idx * n_values + val_idx
            label = f'{param_name}={val:.2g}'
            axes[ic_idx, 1].plot(
                x, dataset.outputs[sample_idx], 
                color=colors[val_idx], linewidth=2, label=label
            )
        
        axes[ic_idx, 1].set_title(f'Output vs {param_name}')
        axes[ic_idx, 1].set_xlabel('x')
        axes[ic_idx, 1].legend(loc='best', fontsize=8)
        axes[ic_idx, 1].grid(True, alpha=0.3)
    
    fig.suptitle(f'Effect of {param_name} on {dataset.metadata["model"]}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def _visualize_2d_parameter_effect(
    dataset, param_name, unique_values, n_per_value, figsize, save_path
):
    """Visualize parameter effect for 2D models."""
    import matplotlib.pyplot as plt
    
    x = dataset.grid.get('x')
    y = dataset.grid.get('y')
    
    n_values = len(unique_values)
    
    # For 2D, show one IC with all parameter values
    if figsize is None:
        figsize = (4 * n_values, 8)
    
    fig, axes = plt.subplots(2, n_values, figsize=figsize)
    if n_values == 1:
        axes = axes.reshape(-1, 1)
    
    # Use first IC
    for val_idx, val in enumerate(unique_values):
        sample_idx = val_idx  # First IC
        
        # Get output (might have multiple channels)
        output = dataset.outputs[sample_idx]
        if output.ndim == 3:
            # Take first channel or magnitude
            if output.shape[-1] >= 2:
                output_viz = np.sqrt(output[:,:,0]**2 + output[:,:,1]**2)
                output_label = '|velocity|'
            else:
                output_viz = output[:,:,0]
                output_label = dataset.output_names[0]
        else:
            output_viz = output
            output_label = dataset.output_names[0]
        
        # Input
        input_data = dataset.inputs[sample_idx]
        if input_data.ndim == 3:
            input_viz = input_data[:,:,0]
        else:
            input_viz = input_data
        
        im0 = axes[0, val_idx].pcolormesh(x, y, input_viz.T, shading='auto', cmap='viridis')
        axes[0, val_idx].set_title(f'{param_name}={val:.2g}\nInput')
        axes[0, val_idx].set_aspect('equal')
        plt.colorbar(im0, ax=axes[0, val_idx])
        
        im1 = axes[1, val_idx].pcolormesh(x, y, output_viz.T, shading='auto', cmap='RdBu_r')
        axes[1, val_idx].set_title(f'Output: {output_label}')
        axes[1, val_idx].set_aspect('equal')
        plt.colorbar(im1, ax=axes[1, val_idx])
    
    fig.suptitle(f'Effect of {param_name} on {dataset.metadata["model"]}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
