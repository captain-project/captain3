"""Unit tests for FeatureExtractor class."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from captain.agents.feature_extractor import FeatureExtractor


class TestFeatureExtractorInit:
    """Tests for FeatureExtractor initialization."""

    def test_init_basic(self, small_env):
        """Test basic initialization."""
        fe = FeatureExtractor(small_env)

        assert fe.n_features > 0
        assert fe.device == small_env.device

    def test_init_default_feature_set(self, small_env):
        """Test default feature set."""
        fe = FeatureExtractor(small_env)

        expected_features = [
            "time", "disturbance", "disturbance_conv",
            "species_richness", "total_population",
            "cost", "protection_matrix"
        ]
        # ext_risk_* features are expanded
        for f in expected_features:
            assert f in fe._feature_set or "ext_risk" in f

    def test_init_custom_feature_set(self, small_env):
        """Test custom feature set."""
        custom_features = ["time", "disturbance", "cost"]
        fe = FeatureExtractor(small_env, feature_set=custom_features)

        assert fe._feature_set == custom_features
        assert fe.n_features == len(custom_features)

    def test_init_feature_names_expansion(self, small_env):
        """Test that current_ext_risk expands to per-class features."""
        fe = FeatureExtractor(small_env, feature_set=["current_ext_risk"])

        # Should have one feature per risk class (5 by default)
        assert fe.n_features == 5
        assert "ext_risk_0" in fe._feature_names
        assert "ext_risk_4" in fe._feature_names

    def test_init_convolution_matrix(self, small_env):
        """Test convolution matrix creation."""
        fe = FeatureExtractor(small_env, convolution=3)

        assert fe._conv_matrix is not None
        # PyTorch sparse CSR uses is_sparse_csr, not is_sparse
        assert fe._conv_matrix.layout == torch.sparse_csr

    def test_init_feature_buffer(self, small_env):
        """Test feature buffer shape."""
        fe = FeatureExtractor(small_env)

        assert fe.features.shape == (fe.n_features, small_env.n_cells)


class TestFeatureExtractorExtract:
    """Tests for feature extraction methods."""

    def test_extract_features_shape(self, small_env):
        """Test extract_features returns correct shape."""
        fe = FeatureExtractor(small_env)
        features = fe.extract_features(small_env)

        assert features.shape == (fe.n_features, small_env.n_cells)

    def test_extract_features_time(self, small_env):
        """Test time feature extraction."""
        fe = FeatureExtractor(small_env, feature_set=["time"])

        # Initial time should be 0
        features = fe.extract_features(small_env)
        assert torch.all(features[0] == 0)

        # After step, time should increase
        small_env.step()
        features = fe.extract_features(small_env)
        assert torch.all(features[0] > 0)

    def test_extract_features_disturbance(self, small_env):
        """Test disturbance feature extraction."""
        fe = FeatureExtractor(small_env, feature_set=["disturbance"])
        features = fe.extract_features(small_env)

        # Should be between 0 and 1
        assert torch.all(features >= 0)
        assert torch.all(features <= 1)

    def test_extract_features_species_richness(self, small_env):
        """Test species richness feature extraction."""
        fe = FeatureExtractor(small_env, feature_set=["species_richness"])
        features = fe.extract_features(small_env)

        # Should be non-negative integers <= n_species
        assert torch.all(features >= 0)
        assert torch.all(features <= small_env.n_species)

    def test_extract_features_protection(self, small_env):
        """Test protection matrix feature extraction."""
        fe = FeatureExtractor(small_env, feature_set=["protection_matrix"])

        # Initially no protection
        features = fe.extract_features(small_env)
        assert torch.all(features == 0)

        # After protection
        small_env.update_protection_matrix([0, 1, 2])
        features = fe.extract_features(small_env)
        assert features[0, 0] == 1
        assert features[0, 1] == 1
        assert features[0, 2] == 1


class TestFeatureExtractorObserve:
    """Tests for observe method (extract + normalize)."""

    def test_observe_shape(self, small_env):
        """Test observe returns correct shape."""
        fe = FeatureExtractor(small_env)
        obs = fe.observe(small_env)

        assert obs.shape == (fe.n_features, small_env.n_cells)

    def test_observe_normalized_range(self, small_env):
        """Test that observe returns values in clipped range."""
        fe = FeatureExtractor(small_env)
        obs = fe.observe(small_env)

        assert torch.all(obs >= -fe.CLIP_RANGE)
        assert torch.all(obs <= fe.CLIP_RANGE)


class TestFeatureExtractorTransform:
    """Tests for transform method."""

    def test_transform_z_score(self, small_env):
        """Test z-score normalization."""
        fe = FeatureExtractor(small_env)

        # Create test observation
        obs = torch.randn(fe.n_features, small_env.n_cells)
        transformed = fe.transform(obs)

        # Should be clipped to range
        assert torch.all(transformed >= -fe.CLIP_RANGE)
        assert torch.all(transformed <= fe.CLIP_RANGE)

    def test_transform_clipping(self, small_env):
        """Test that extreme values are clipped."""
        fe = FeatureExtractor(small_env)

        # Create observation with extreme values
        obs = torch.ones(fe.n_features, small_env.n_cells) * 1000
        transformed = fe.transform(obs)

        # Should be clipped
        assert torch.all(transformed <= fe.CLIP_RANGE)


class TestFeatureExtractorRescaler:
    """Tests for rescaler functionality."""

    def test_set_rescaler(self, small_env):
        """Test set_rescaler computes statistics."""
        fe = FeatureExtractor(small_env)

        assert fe._rescaler_mean is not None
        assert fe._rescaler_std is not None
        assert fe._rescaler_mean.shape[0] == fe.n_features
        assert fe._rescaler_std.shape[0] == fe.n_features

    def test_rescaler_std_positive(self, small_env):
        """Test that std is always positive (minimum 1)."""
        fe = FeatureExtractor(small_env)

        assert torch.all(fe._rescaler_std >= 1.0)

    def test_keys_to_reset(self, small_env):
        """Test that keys_to_reset features are not rescaled."""
        fe = FeatureExtractor(small_env)

        # Check that features in keys_to_reset have std=1, mean=0
        for key in fe.keys_to_reset:
            if key in fe.feature_map:
                idx = fe.feature_map[key]
                assert fe._rescaler_std[idx].item() == 1.0
                assert fe._rescaler_mean[idx].item() == 0.0


class TestFeatureExtractorDevice:
    """Tests for device handling."""

    def test_to_cpu(self, small_env):
        """Test moving to CPU."""
        fe = FeatureExtractor(small_env)
        fe.to("cpu")

        assert fe.device == torch.device("cpu")
        assert fe.features.device.type == "cpu"

    def test_to_returns_self(self, small_env):
        """Test that to() returns self for chaining."""
        fe = FeatureExtractor(small_env)
        result = fe.to("cpu")

        assert result is fe


class TestFeatureExtractorProperties:
    """Tests for FeatureExtractor properties."""

    def test_n_features_property(self, small_env):
        """Test n_features property."""
        fe = FeatureExtractor(small_env)

        assert isinstance(fe.n_features, int)
        assert fe.n_features > 0

    def test_feature_map(self, small_env):
        """Test feature_map lookup."""
        fe = FeatureExtractor(small_env, feature_set=["time", "cost"])

        assert "time" in fe.feature_map
        assert "cost" in fe.feature_map
        assert fe.feature_map["time"] == 0
        assert fe.feature_map["cost"] == 1


class TestFeatureExtractorEdgeFeatures:
    """Tests for the optional edge_exposure and buffer_value features."""

    @pytest.fixture
    def edge_env(self, small_grid_params):
        from captain.data.extinction_risk import ExtinctionRisk
        from captain.data.spatial_data import SpatialData
        from captain.environment.bioenv import BioEnv

        p = small_grid_params
        mask = p["mask"]
        protection = np.zeros_like(p["disturbance"])
        protection[0, 4:7, 4:7] = 1.0
        return BioEnv(
            sdms=SpatialData(data=p["sdms"], mask=mask),
            disturbance=SpatialData(data=p["disturbance"], mask=mask),
            costs=SpatialData(data=p["costs"], mask=mask),
            protection_matrix=SpatialData(data=protection, mask=mask),
            growth_rates=p["growth_rates"],
            dispersal_rates=p["dispersal_rate"],
            species_k=p["species_k"],
            sensitivity_rates=p["sensitivity"],
            ext_risk=ExtinctionRisk(init_status=p["init_status"]),
            edge_factor=0.9,
        )

    def test_edge_features_match_env(self, edge_env):
        """Features equal env.edge_exposure() / env.edge_buffer() and are not rescaled."""
        values = torch.rand(edge_env.n_cells)
        fe = FeatureExtractor(
            edge_env,
            feature_set=[*FeatureExtractor.DEFAULT_FEATURE_SET, "edge_exposure", "buffer_value"],
            cell_values=values,
        )
        obs = fe.observe(edge_env)

        assert torch.allclose(obs[fe.feature_map["edge_exposure"]], edge_env.edge_exposure())
        assert torch.allclose(obs[fe.feature_map["buffer_value"]], edge_env.edge_buffer(values))
        assert fe.n_features == FeatureExtractor(edge_env).n_features + 2

    def test_edge_features_update_with_protection(self, edge_env):
        """The features are recomputed from the current protection at each observe()."""
        fe = FeatureExtractor(
            edge_env,
            feature_set=["edge_exposure", "buffer_value"],
            cell_values=torch.ones(edge_env.n_cells),
        )
        before = fe.observe(edge_env).clone()
        edge_env.update_protection_matrix(list(range(10)))
        after = fe.observe(edge_env)

        assert not torch.allclose(before, after)

    def test_buffer_value_requires_cell_values(self, edge_env):
        """buffer_value without cell_values is rejected."""
        with pytest.raises(ValueError, match="cell_values"):
            FeatureExtractor(edge_env, feature_set=["buffer_value"])


class TestWorstSpeciesFeature:
    """Tests for the worst_species_value feature."""

    def test_marks_cells_of_worst_species(self, small_env):
        """The feature equals the scaled K share of the worst-served species, per cell."""
        from captain.agents.rewards import protected_fractions, worst_species

        small_env.update_protection_matrix(list(range(30)))
        fe = FeatureExtractor(small_env, feature_set=["worst_species_value"], worst_fraction=0.4)
        obs = fe.observe(small_env)[0]

        worst = worst_species(protected_fractions(small_env), 0.4)
        k_full = small_env._species_k[worst, None] * small_env.sdms.data_min_threshold[worst]
        expected = (k_full / small_env.ext_risk.init_pop[worst, None]).sum(0)
        assert len(worst) == 2  # ceil(0.4 * 5 species)
        assert torch.allclose(obs, expected / expected.max())

    def test_follows_protection(self, small_env):
        """Protecting the worst-served species' cells changes which species are worst."""
        fe = FeatureExtractor(small_env, feature_set=["worst_species_value"], worst_fraction=0.2)
        small_env.update_protection_matrix(list(range(10)))
        before = fe.observe(small_env).clone()
        small_env.update_protection_matrix(list(range(10, small_env.n_cells, 2)))
        after = fe.observe(small_env)

        assert not torch.allclose(before, after)

    def test_threat_weights(self, small_env):
        """With worst_threat_weights each worst species' share is scaled by its weight."""
        from captain.agents.rewards import protected_fractions, worst_species

        small_env.update_protection_matrix(list(range(30)))
        weights = [1.0, 2.0, 4.0, 8.0, 16.0]
        fe = FeatureExtractor(small_env, feature_set=["worst_species_value"], worst_fraction=0.4,
                              worst_threat_weights=weights)
        obs = fe.observe(small_env)[0]

        worst = worst_species(protected_fractions(small_env), 0.4)
        w = torch.tensor(weights)[small_env.ext_risk.init_status[worst]]
        k_full = small_env._species_k[worst, None] * small_env.sdms.data_min_threshold[worst]
        expected = (w[:, None] * k_full / small_env.ext_risk.init_pop[worst, None]).sum(0)
        assert torch.allclose(obs, expected / expected.max())
