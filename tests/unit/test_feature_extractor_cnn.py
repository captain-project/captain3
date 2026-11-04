"""Unit tests for FeatureExtractorCNN class."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from captain.agents.feature_extractor_cnn import FeatureExtractorCNN, N_INPUT_CHANNELS
from captain.agents.spatial_cnn import SpatialCNN
from captain.data.extinction_risk import ExtinctionRisk
from captain.data.spatial_data import SpatialData
from captain.environment.bioenv import BioEnv


@pytest.fixture
def cnn_env(small_grid_params, rng):
    """Create a small BioEnv for CNN testing."""
    p = small_grid_params
    mask = p["mask"]

    env = BioEnv(
        sdms=SpatialData(data=p["sdms"], mask=mask),
        disturbance=SpatialData(data=p["disturbance"], mask=mask),
        costs=SpatialData(data=p["costs"], mask=mask),
        protection_matrix=SpatialData(data=np.zeros_like(p["disturbance"]), mask=mask),
        growth_rates=p["growth_rates"],
        dispersal_rates=p["dispersal_rate"],
        species_k=p["species_k"],
        sensitivity_rates=p["sensitivity"],
        ext_risk=ExtinctionRisk(init_status=p["init_status"]),
    )
    return env


@pytest.fixture
def cnn():
    """Create a SpatialCNN with default params."""
    return SpatialCNN(n_input_channels=12, n_branch_filters=8, n_output_features=8)


@pytest.fixture
def cnn_extractor(cnn_env, cnn):
    """Create a FeatureExtractorCNN."""
    return FeatureExtractorCNN(cnn_env, cnn=cnn)


class TestFeatureExtractorCNNInit:
    """Tests for FeatureExtractorCNN initialization."""

    def test_init_basic(self, cnn_env, cnn):
        """Test basic initialization."""
        fe = FeatureExtractorCNN(cnn_env, cnn=cnn)

        assert fe.device == cnn_env.device
        assert fe.n_features == 8
        assert fe._n_cells == cnn_env.n_cells

    def test_grid_shape(self, cnn_extractor, cnn_env):
        """Test pre-allocated grid has correct shape."""
        h, w = cnn_env.sdms._data_shape[1:]
        assert cnn_extractor._grid.shape == (1, N_INPUT_CHANNELS, h, w)

    def test_valid_mask_channel(self, cnn_extractor):
        """Test that channel 11 (valid mask) is filled."""
        mask_channel = cnn_extractor._grid[0, 11]
        # Should have non-zero values at valid cell locations
        assert mask_channel.sum() == cnn_extractor._n_cells

    def test_rescaler_computed(self, cnn_extractor):
        """Test normalization stats are computed at init."""
        assert cnn_extractor._rescaler_mean is not None
        assert cnn_extractor._rescaler_std is not None
        assert cnn_extractor._rescaler_mean.shape == (1, N_INPUT_CHANNELS, 1, 1)
        assert cnn_extractor._rescaler_std.shape == (1, N_INPUT_CHANNELS, 1, 1)

    def test_rescaler_std_positive(self, cnn_extractor):
        """Test std is always >= 1."""
        assert (cnn_extractor._rescaler_std >= 1.0).all()


class TestFeatureExtractorCNNObserve:
    """Tests for observe method."""

    def test_observe_shape(self, cnn_extractor, cnn_env):
        """Test observe returns correct shape."""
        obs = cnn_extractor.observe(cnn_env)

        assert obs.shape == (8, cnn_env.n_cells)

    def test_observe_finite(self, cnn_extractor, cnn_env):
        """Test observe returns finite values."""
        obs = cnn_extractor.observe(cnn_env)

        assert torch.isfinite(obs).all()

    def test_observe_after_step(self, cnn_extractor, cnn_env):
        """Test observe works after environment step."""
        cnn_env.step()
        obs = cnn_extractor.observe(cnn_env)

        assert obs.shape == (8, cnn_env.n_cells)
        assert torch.isfinite(obs).all()

    def test_observe_after_protection(self, cnn_extractor, cnn_env):
        """Test observe reflects protection changes."""
        obs_before = cnn_extractor.observe(cnn_env).clone()

        cnn_env.update_protection_matrix([0, 1, 2])
        obs_after = cnn_extractor.observe(cnn_env)

        # Features should change after protection
        assert not torch.allclose(obs_before, obs_after)

    def test_observe_deterministic(self, cnn_extractor, cnn_env):
        """Test observe is deterministic."""
        obs1 = cnn_extractor.observe(cnn_env).clone()
        obs2 = cnn_extractor.observe(cnn_env)

        assert torch.allclose(obs1, obs2)


class TestFeatureExtractorCNNGrid:
    """Tests for grid construction."""

    def test_fill_grid_shape(self, cnn_extractor, cnn_env):
        """Test _fill_grid returns correct shape."""
        grid = cnn_extractor._fill_grid(cnn_env)
        h, w = cnn_env.sdms._data_shape[1:]

        assert grid.shape == (1, N_INPUT_CHANNELS, h, w)

    def test_fill_grid_species_richness(self, cnn_extractor, cnn_env):
        """Test channel 0 contains species richness."""
        grid = cnn_extractor._fill_grid(cnn_env)
        r, c = cnn_extractor._row_coords, cnn_extractor._col_coords

        richness = grid[0, 0, r, c]
        expected = (cnn_env.h > 1).sum(dim=0).float()

        assert torch.allclose(richness, expected)

    def test_fill_grid_protection(self, cnn_extractor, cnn_env):
        """Test channel 9 reflects protection status."""
        # Initially no protection
        grid = cnn_extractor._fill_grid(cnn_env)
        r, c = cnn_extractor._row_coords, cnn_extractor._col_coords
        assert grid[0, 9, r, c].sum() == 0

        # After protecting cells
        cnn_env.update_protection_matrix([0, 1])
        grid = cnn_extractor._fill_grid(cnn_env)
        assert grid[0, 9, r[0], c[0]] == 1.0
        assert grid[0, 9, r[1], c[1]] == 1.0

    def test_fill_grid_time(self, cnn_extractor, cnn_env):
        """Test channel 10 contains scaled time."""
        grid = cnn_extractor._fill_grid(cnn_env)
        r, c = cnn_extractor._row_coords, cnn_extractor._col_coords

        # Initially time = 0
        assert grid[0, 10, r, c].sum() == 0

        # After step
        cnn_env.step()
        grid = cnn_extractor._fill_grid(cnn_env)
        expected_time = cnn_env._env_step_num / cnn_extractor._time_rescale
        assert torch.allclose(
            grid[0, 10, r[0], c[0]],
            torch.tensor(expected_time, dtype=torch.float32),
        )


class TestFeatureExtractorCNNNormalization:
    """Tests for normalization."""

    def test_normalize_clips(self, cnn_extractor):
        """Test normalization clips to CLIP_RANGE."""
        grid = torch.ones(1, N_INPUT_CHANNELS, 10, 10) * 1000
        normalized = cnn_extractor._normalize_grid(grid)

        assert (normalized <= cnn_extractor.CLIP_RANGE).all()
        assert (normalized >= -cnn_extractor.CLIP_RANGE).all()


class TestFeatureExtractorCNNDevice:
    """Tests for device handling."""

    def test_to_cpu(self, cnn_extractor):
        """Test moving to CPU."""
        result = cnn_extractor.to("cpu")

        assert result is cnn_extractor
        assert cnn_extractor.device == torch.device("cpu")
        assert cnn_extractor._grid.device.type == "cpu"
        assert cnn_extractor._row_coords.device.type == "cpu"

    def test_n_features_property(self, cnn_extractor):
        """Test n_features matches CNN output."""
        assert cnn_extractor.n_features == 8
