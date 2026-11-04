"""Unit tests for SpatialData class."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from captain.data.spatial_data import SpatialData


class TestSpatialDataInit:
    """Tests for SpatialData initialization."""

    def test_init_with_numpy_3d(self, rng):
        """Test initialization with 3D numpy array."""
        data = rng.random((5, 10, 10)).astype(np.float32)
        sd = SpatialData(data=data)

        assert sd.shape == (5, 10, 10)
        assert isinstance(sd.data, torch.Tensor)
        assert sd.data.shape[0] == 5  # channels preserved

    def test_init_with_numpy_2d(self, rng):
        """Test initialization with 2D numpy array (auto-adds channel dim)."""
        data = rng.random((10, 10)).astype(np.float32)
        sd = SpatialData(data=data)

        assert sd.shape == (1, 10, 10)  # Channel dim added
        assert sd.data.shape[0] == 1

    def test_init_with_torch_tensor(self, rng):
        """Test initialization with torch tensor."""
        data = torch.from_numpy(rng.random((5, 10, 10)).astype(np.float32))
        sd = SpatialData(data=data)

        assert sd.shape == (5, 10, 10)
        assert isinstance(sd.data, torch.Tensor)

    def test_init_with_mask(self, rng):
        """Test initialization with mask (NaN values excluded)."""
        data = rng.random((3, 10, 10)).astype(np.float32)
        mask = np.ones((10, 10))
        mask[:2, :2] = np.nan  # Mask 4 cells

        sd = SpatialData(data=data, mask=mask)

        # Should have fewer valid cells than full grid
        n_valid = 10 * 10 - 4
        assert sd.data.shape == (3, n_valid)

    def test_init_with_delta(self, rng):
        """Test initialization with delta_per_step."""
        data = rng.random((3, 10, 10)).astype(np.float32)
        delta = np.full_like(data, 0.01)

        sd = SpatialData(data=data, delta_per_step=delta)

        assert sd._delta is not None
        assert sd._delta.shape == sd.data.shape

    def test_init_with_bounds(self, rng):
        """Test initialization with bounds."""
        data = rng.random((3, 10, 10)).astype(np.float32)

        sd = SpatialData(data=data, lower_bound=0.0, upper_bound=1.0)

        assert sd._lower_bound == 0.0
        assert sd._upper_bound == 1.0

    def test_init_with_names(self, rng):
        """Test initialization with channel names."""
        data = rng.random((3, 10, 10)).astype(np.float32)
        names = ["species_a", "species_b", "species_c"]

        sd = SpatialData(data=data, names=names)

        assert sd.names == names

    def test_init_default_names(self, rng):
        """Test default channel names."""
        data = rng.random((3, 10, 10)).astype(np.float32)
        sd = SpatialData(data=data)

        assert sd.names == ["dat_0", "dat_1", "dat_2"]

    def test_init_with_device(self, rng):
        """Test initialization with device."""
        data = rng.random((3, 10, 10)).astype(np.float32)
        sd = SpatialData(data=data, device="cpu")

        assert sd.device == torch.device("cpu")
        assert sd.data.device.type == "cpu"


class TestSpatialDataUpdate:
    """Tests for SpatialData update methods."""

    def test_update_applies_delta(self, rng):
        """Test that update applies delta correctly."""
        data = np.full((2, 5, 5), 0.5, dtype=np.float32)
        delta = np.full((2, 5, 5), 0.1, dtype=np.float32)

        sd = SpatialData(data=data, delta_per_step=delta)
        initial_sum = sd.data.sum().item()

        sd.update(time_step=1)

        # Values should increase by delta
        assert sd.data.sum().item() > initial_sum

    def test_update_respects_upper_bound(self, rng):
        """Test that update respects upper bound."""
        data = np.full((2, 5, 5), 0.9, dtype=np.float32)
        delta = np.full((2, 5, 5), 0.2, dtype=np.float32)

        sd = SpatialData(data=data, delta_per_step=delta, upper_bound=1.0)
        sd.update(time_step=1)

        assert sd.data.max().item() <= 1.0

    def test_update_respects_lower_bound(self, rng):
        """Test that update respects lower bound."""
        data = np.full((2, 5, 5), 0.1, dtype=np.float32)
        delta = np.full((2, 5, 5), -0.2, dtype=np.float32)

        sd = SpatialData(data=data, delta_per_step=delta, lower_bound=0.0)
        sd.update(time_step=1)

        assert sd.data.min().item() >= 0.0

    def test_update_without_delta_is_noop(self, rng):
        """Test that update without delta does nothing."""
        data = rng.random((2, 5, 5)).astype(np.float32)
        sd = SpatialData(data=data)  # No delta

        original_data = sd.data.clone()
        sd.update(time_step=1)

        assert torch.allclose(sd.data, original_data)


class TestSpatialDataReset:
    """Tests for SpatialData reset functionality."""

    def test_reset_restores_initial_data(self, rng):
        """Test that reset restores initial data."""
        data = np.full((2, 5, 5), 0.5, dtype=np.float32)
        delta = np.full((2, 5, 5), 0.1, dtype=np.float32)

        sd = SpatialData(data=data, delta_per_step=delta)
        original_data = sd.data.clone()

        # Modify data
        sd.update(time_step=1)
        sd.update(time_step=1)
        assert not torch.allclose(sd.data, original_data)

        # Reset
        sd.reset()
        assert torch.allclose(sd.data, original_data)

    def test_reset_with_backup_file(self, rng):
        """Test reset with memory-mapped backup."""
        data = np.full((2, 5, 5), 0.5, dtype=np.float32)
        delta = np.full((2, 5, 5), 0.1, dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            backup_path = f.name

        try:
            sd = SpatialData(data=data, delta_per_step=delta, backup_path=backup_path)
            original_sum = sd.data.sum().item()

            sd.update(time_step=1)
            assert sd.data.sum().item() != original_sum

            sd.reset()
            assert abs(sd.data.sum().item() - original_sum) < 1e-5
        finally:
            Path(backup_path).unlink(missing_ok=True)


class TestSpatialDataColumnUpdate:
    """Tests for update_col_values method."""

    def test_update_col_values_list(self, rng):
        """Test updating column values with list indices."""
        data = np.zeros((2, 5, 5), dtype=np.float32)
        sd = SpatialData(data=data)

        sd.update_cell_values([0, 1, 2], val=1.0)

        assert sd.data[:, 0].sum().item() == 2.0  # 2 channels
        assert sd.data[:, 1].sum().item() == 2.0
        assert sd.data[:, 2].sum().item() == 2.0

    def test_update_col_values_tensor(self, rng):
        """Test updating column values with tensor indices."""
        data = np.zeros((2, 5, 5), dtype=np.float32)
        sd = SpatialData(data=data)

        idx = torch.tensor([0, 1])
        sd.update_cell_values(idx, val=0.5)

        assert sd.data[:, 0].sum().item() == 1.0
        assert sd.data[:, 1].sum().item() == 1.0

    def test_update_col_values_updates_nonzero_mask(self, rng):
        """Test that update_col_values updates nonzero mask."""
        data = np.zeros((2, 5, 5), dtype=np.float32)
        sd = SpatialData(data=data)

        assert sd._nonzero_cells_mask.sum().item() == 0

        sd.update_cell_values([0, 1], val=1.0)

        assert sd._nonzero_cells_mask.sum().item() == 2


class TestSpatialDataThreshold:
    """Tests for min_threshold functionality."""

    def test_data_min_threshold(self, rng):
        """Test data_min_threshold property."""
        data = np.array([[[0.1, 0.5, 0.3], [0.2, 0.4, 0.6]]], dtype=np.float32)
        min_threshold = np.array([0.25], dtype=np.float32)

        sd = SpatialData(data=data, min_threshold=min_threshold)

        thresholded = sd.data_min_threshold
        # Values below 0.25 should be zeroed
        assert thresholded[0, 0].item() == 0.0  # 0.1 < 0.25
        assert thresholded[0, 1].item() == 0.5  # 0.5 >= 0.25

    def test_data_min_threshold_none(self, rng):
        """Test data_min_threshold when no threshold set."""
        data = rng.random((2, 5, 5)).astype(np.float32)
        sd = SpatialData(data=data)

        # Should return original data
        assert torch.allclose(sd.data_min_threshold, sd.data)


class TestSpatialDataDevice:
    """Tests for device handling."""

    def test_to_cpu(self, rng):
        """Test moving to CPU."""
        data = rng.random((2, 5, 5)).astype(np.float32)
        sd = SpatialData(data=data)

        sd.to("cpu")

        assert sd.device == torch.device("cpu")
        assert sd.data.device.type == "cpu"

    def test_to_returns_self(self, rng):
        """Test that to() returns self for chaining."""
        data = rng.random((2, 5, 5)).astype(np.float32)
        sd = SpatialData(data=data)

        result = sd.to("cpu")

        assert result is sd


class TestSpatialDataReconstruct:
    """Tests for grid reconstruction."""

    def test_reconstruct_grid_shape(self, rng):
        """Test that reconstruct_grid returns correct shape."""
        data = rng.random((3, 10, 10)).astype(np.float32)
        sd = SpatialData(data=data)

        grid = sd.reconstruct_grid

        assert grid.shape == (3, 10, 10)

    def test_reconstruct_grid_with_mask(self, rng):
        """Test reconstruct_grid with masked cells."""
        data = rng.random((2, 5, 5)).astype(np.float32)
        mask = np.ones((5, 5))
        mask[0, 0] = np.nan  # Mask one cell

        sd = SpatialData(data=data, mask=mask)
        grid = sd.reconstruct_grid

        # Masked cell should be NaN
        assert np.isnan(grid[0, 0, 0])
        assert np.isnan(grid[1, 0, 0])
