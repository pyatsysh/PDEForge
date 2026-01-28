"""
Tests for PDE models.
"""

import pytest
import numpy as np
from pdeforge import generate_dataset, get_model, list_models


class TestModelRegistry:
    """Test model registry functionality."""
    
    def test_list_models(self):
        """Test that list_models returns expected models."""
        models = list_models()
        assert "burgers_1d" in models
        assert "darcy_2d" in models
        assert "stokes_2d" in models
    
    def test_get_model(self):
        """Test that get_model returns valid model class."""
        for name in list_models():
            model_cls = get_model(name)
            assert model_cls is not None
    
    def test_get_invalid_model(self):
        """Test that invalid model name raises error."""
        with pytest.raises(ValueError):
            get_model("nonexistent_model")


class TestBurgers1D:
    """Tests for 1D Burgers equation."""
    
    def test_basic_generation(self):
        """Test basic dataset generation."""
        dataset = generate_dataset(
            model="burgers_1d",
            n_samples=5,
            resolution={"x": 64},
            seed=42,
        )
        
        assert dataset.n_samples == 5
        assert dataset.inputs.shape == (5, 64)
        assert dataset.outputs.shape == (5, 64)
    
    def test_reproducibility(self):
        """Test that same seed gives same results."""
        ds1 = generate_dataset(
            model="burgers_1d",
            n_samples=3,
            resolution={"x": 32},
            seed=123,
        )
        ds2 = generate_dataset(
            model="burgers_1d",
            n_samples=3,
            resolution={"x": 32},
            seed=123,
        )
        
        np.testing.assert_array_equal(ds1.inputs, ds2.inputs)
        np.testing.assert_array_equal(ds1.outputs, ds2.outputs)
    
    def test_custom_params(self):
        """Test with custom parameters."""
        dataset = generate_dataset(
            model="burgers_1d",
            n_samples=3,
            resolution={"x": 64},
            params={"viscosity": 0.001, "time_end": 0.5},
            seed=42,
        )
        
        assert dataset.metadata["params"]["viscosity"] == 0.001
        assert dataset.metadata["params"]["time_end"] == 0.5
    
    def test_validation(self):
        """Test that solutions are valid."""
        model = get_model("burgers_1d")(resolution={"x": 64})
        
        ic, solution, info = model.generate_sample(seed=42)
        
        assert info['valid']
        assert not np.isnan(solution).any()
        assert not np.isinf(solution).any()


class TestDarcy2D:
    """Tests for 2D Darcy flow."""
    
    def test_basic_generation(self):
        """Test basic dataset generation."""
        dataset = generate_dataset(
            model="darcy_2d",
            n_samples=5,
            resolution={"x": 32, "y": 32},
            seed=42,
        )
        
        assert dataset.n_samples == 5
        assert dataset.inputs.shape == (5, 32, 32)
        assert dataset.outputs.shape == (5, 32, 32)
    
    def test_permeability_bounds(self):
        """Test that permeability stays within bounds."""
        model = get_model("darcy_2d")(
            resolution={"x": 32, "y": 32},
            kappa_min=0.5,
            kappa_max=5.0,
        )
        
        kappa = model.generate_ic(seed=42)
        
        # Check bounds (with small tolerance for numerical issues)
        assert kappa.min() >= 0.5 - 0.01
        assert kappa.max() <= 5.0 + 0.01


class TestStokes2D:
    """Tests for 2D Stokes flow."""
    
    def test_basic_generation(self):
        """Test basic dataset generation."""
        dataset = generate_dataset(
            model="stokes_2d",
            n_samples=5,
            resolution={"x": 32, "y": 32},
            seed=42,
        )
        
        assert dataset.n_samples == 5
        assert dataset.inputs.shape == (5, 32, 32, 2)  # fx, fy
        assert dataset.outputs.shape == (5, 32, 32, 3)  # u, v, p
    
    def test_divergence_free(self):
        """Test that velocity field is divergence-free."""
        model = get_model("stokes_2d")(resolution={"x": 32, "y": 32})
        
        force, solution, info = model.generate_sample(seed=42)
        
        assert info['valid']
        assert info['divergence'] < 1e-8


class TestDataset:
    """Tests for PDEDataset functionality."""
    
    def test_split(self):
        """Test dataset splitting."""
        dataset = generate_dataset(
            model="burgers_1d",
            n_samples=100,
            resolution={"x": 32},
            seed=42,
        )
        
        splits = dataset.split(train=0.6, val=0.15, cal=0.15, test=0.1)
        
        assert splits['train'].n_samples == 60
        assert splits['val'].n_samples == 15
        assert splits['cal'].n_samples == 15
        assert splits['test'].n_samples == 10
    
    def test_save_load(self, tmp_path):
        """Test save and load."""
        dataset = generate_dataset(
            model="burgers_1d",
            n_samples=10,
            resolution={"x": 32},
            seed=42,
        )
        
        save_path = tmp_path / "test_dataset"
        dataset.save(save_path)
        
        from pdeforge.io import load_dataset
        loaded = load_dataset(save_path)
        
        np.testing.assert_array_equal(dataset.inputs, loaded.inputs)
        np.testing.assert_array_equal(dataset.outputs, loaded.outputs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
