"""Integration tests for CNN feature extractor with episode runner and ES."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from captain.agents.feature_extractor_cnn import FeatureExtractorCNN
from captain.agents.policy_network import CellCNNPolicy, CellNN, PolicyNetwork
from captain.agents.reward_aggregator import Rewards
from captain.agents.rewards import CalcRewardExtRisk, CalcRewardPersistentCost
from captain.agents.spatial_cnn import SpatialCNN
from captain.algorithms.budget_manager import GlobalBudgetManager
from captain.algorithms.episode import EpisodeRunner
from captain.data.extinction_risk import ExtinctionRisk
from captain.data.spatial_data import SpatialData
from captain.environment.bioenv import BioEnv


@pytest.fixture
def cnn_episode_runner(small_grid_params, rng):
    """Create a complete CNN-based episode runner."""
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

    # CNN-based setup
    cnn = SpatialCNN(n_input_channels=12, n_branch_filters=8, n_output_features=8)
    cell_nn = CellNN(input_dim=8, hidden_dim=16)
    composite = CellCNNPolicy(cnn, cell_nn)

    policy = PolicyNetwork(composite, device="cpu", seed=42)
    feature_extractor = FeatureExtractorCNN(env, cnn=cnn)

    rewards = Rewards([
        CalcRewardExtRisk(threat_weights=[1, 0, -8, -16, -32]),
        CalcRewardPersistentCost(rescaler=0.1),
    ])

    runner = EpisodeRunner(
        env=env,
        feature_extractor=feature_extractor,
        policy_network=policy,
        rewards=rewards,
        n_steps=5,
        budget_manager=GlobalBudgetManager(
            total_target=20, cells_per_time_step=5, feature_updates_per_time_step=1
        ),
    )

    return runner


class TestCNNEpisode:
    """Test full episode with CNN features."""

    def test_run_episode(self, cnn_episode_runner):
        """Test episode runs to completion with CNN features."""
        params = cnn_episode_runner.policy.get_flat_weights()
        info, reward = cnn_episode_runner.run_episode(params)

        assert isinstance(reward, float)
        assert np.isfinite(reward)
        assert info["protected_cells"] > 0
        assert info["n_steps"] == 5

    def test_run_episode_finite_rewards(self, cnn_episode_runner):
        """Test all rewards are finite."""
        params = cnn_episode_runner.policy.get_flat_weights()
        info, reward = cnn_episode_runner.run_episode(params)

        for r in info["reward_history"]:
            assert np.all(np.isfinite(r))

    def test_run_multiple_episodes(self, cnn_episode_runner):
        """Test multiple episodes run consistently."""
        params = cnn_episode_runner.policy.get_flat_weights()

        _, reward1 = cnn_episode_runner.run_episode(params)
        _, reward2 = cnn_episode_runner.run_episode(params)

        assert np.isfinite(reward1)
        assert np.isfinite(reward2)
        assert abs(reward1 - reward2) < 0.1 * max(abs(reward1), abs(reward2), 1.0)


class TestCNNWeightPropagation:
    """Test shared weight updates between policy and feature extractor."""

    def test_param_count(self, cnn_episode_runner):
        """Test composite model has expected param count."""
        n_params = cnn_episode_runner.policy.get_n_params()

        # CNN: 1880, CellNN(8->16->1): 8*16+16 + 16*1+1 = 161
        assert n_params == 2041

    def test_weight_roundtrip(self, cnn_episode_runner):
        """Test get/set flat weights roundtrip."""
        policy = cnn_episode_runner.policy

        original = policy.get_flat_weights().copy()
        policy.set_flat_weights(original)
        restored = policy.get_flat_weights()

        assert np.allclose(original, restored)

    def test_shared_cnn_reference(self, cnn_episode_runner):
        """Test feature extractor and policy share same CNN object."""
        policy_cnn = cnn_episode_runner.policy.model.cnn
        extractor_cnn = cnn_episode_runner.feature_extractor.cnn

        assert policy_cnn is extractor_cnn

    def test_weight_update_propagates(self, cnn_episode_runner):
        """Test that set_flat_weights updates CNN in feature extractor."""
        policy = cnn_episode_runner.policy
        extractor = cnn_episode_runner.feature_extractor

        # Get CNN weights before
        cnn_weight_before = extractor.cnn.local_conv.weight.clone()

        # Set new weights via policy
        n_params = policy.get_n_params()
        new_weights = np.ones(n_params, dtype=np.float32) * 0.1
        policy.set_flat_weights(new_weights)

        # CNN weights in extractor should have changed
        cnn_weight_after = extractor.cnn.local_conv.weight
        assert not torch.allclose(cnn_weight_before, cnn_weight_after)

    def test_different_weights_different_features(self, cnn_episode_runner):
        """Test different CNN weights produce different features."""
        env = cnn_episode_runner.env
        policy = cnn_episode_runner.policy
        extractor = cnn_episode_runner.feature_extractor

        # Observe with initial weights
        obs1 = extractor.observe(env).clone()

        # Change weights
        n_params = policy.get_n_params()
        policy.set_flat_weights(np.random.randn(n_params).astype(np.float32))

        # Observe with new weights
        obs2 = extractor.observe(env)

        assert not torch.allclose(obs1, obs2)

    def test_different_weights_different_rewards(self, cnn_episode_runner):
        """Test different weights produce different episode outcomes."""
        policy = cnn_episode_runner.policy
        n_params = policy.get_n_params()

        params1 = np.random.RandomState(1).randn(n_params).astype(np.float32)
        params2 = np.random.RandomState(2).randn(n_params).astype(np.float32)

        _, reward1 = cnn_episode_runner.run_episode(params1)
        _, reward2 = cnn_episode_runner.run_episode(params2)

        assert np.isfinite(reward1)
        assert np.isfinite(reward2)


class TestCellCNNPolicyUnit:
    """Unit tests for CellCNNPolicy composite module."""

    def test_forward_delegates_to_cell_nn(self):
        """Test forward only uses CellNN, not CNN."""
        cnn = SpatialCNN(12, 8, 8)
        cell_nn = CellNN(input_dim=8, hidden_dim=16)
        composite = CellCNNPolicy(cnn, cell_nn)

        x = torch.randn(8, 100)
        out = composite(x)

        # Should match CellNN output
        expected = cell_nn(x)
        assert torch.allclose(out, expected)

    def test_all_params_registered(self):
        """Test all parameters from both submodules are registered."""
        cnn = SpatialCNN(12, 8, 8)
        cell_nn = CellNN(input_dim=8, hidden_dim=16)
        composite = CellCNNPolicy(cnn, cell_nn)

        cnn_params = sum(p.numel() for p in cnn.parameters())
        cell_params = sum(p.numel() for p in cell_nn.parameters())
        total_params = sum(p.numel() for p in composite.parameters())

        assert total_params == cnn_params + cell_params
        assert total_params == 2041

    def test_seeded_init_conv2d(self):
        """Test seeded_init handles Conv2d layers."""
        cnn = SpatialCNN(12, 8, 8)
        cell_nn = CellNN(input_dim=8, hidden_dim=16)
        composite = CellCNNPolicy(cnn, cell_nn)

        # Should not raise
        policy = PolicyNetwork(composite, device="cpu", seed=42)
        weights = policy.get_flat_weights()

        assert len(weights) == 2041
        assert np.isfinite(weights).all()
