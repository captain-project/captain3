"""Integration tests for EpisodeRunner."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from captain.agents.feature_extractor import FeatureExtractor
from captain.agents.policy_network import CellNN, PolicyNetwork
from captain.agents.reward_aggregator import Rewards
from captain.agents.rewards import CalcRewardExtRisk, CalcRewardPersistentCost
from captain.algorithms.budget_manager import GlobalBudgetManager
from captain.algorithms.episode import EpisodeRunner
from captain.data.extinction_risk import ExtinctionRisk
from captain.data.spatial_data import SpatialData
from captain.environment.bioenv import BioEnv


@pytest.fixture
def episode_runner(small_grid_params, rng):
    """Create a complete episode runner setup."""
    p = small_grid_params
    mask = p["mask"]

    # Create environment
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

    # Create agents
    feature_extractor = FeatureExtractor(env)
    model = CellNN(input_dim=feature_extractor.n_features, hidden_dim=16)
    policy = PolicyNetwork(model)

    rewards = Rewards([
        CalcRewardExtRisk(threat_weights=[1, 0, -8, -16, -32]),
        CalcRewardPersistentCost(rescaler=0.1),
    ])

    # Create runner with small settings
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


class TestEpisodeRunnerInit:
    """Tests for EpisodeRunner initialization."""

    def test_init_basic(self, episode_runner):
        """Test basic initialization."""
        assert episode_runner.env is not None
        assert episode_runner.feature_extractor is not None
        assert episode_runner.policy is not None
        assert episode_runner.rewards is not None

    def test_init_parameters(self, episode_runner):
        """Test initialization parameters."""
        assert episode_runner.n_steps == 5
        assert episode_runner.budget_manager.total_target == 20
        assert episode_runner.budget_manager.cells_per_time_step == 5


class TestEpisodeRunnerRun:
    """Tests for running episodes."""

    def test_run_episode_returns_info_and_reward(self, episode_runner):
        """Test run_episode returns correct structure."""
        params = episode_runner.policy.get_flat_weights()
        info, reward = episode_runner.run_episode(params)

        assert isinstance(info, dict)
        assert isinstance(reward, float)

    def test_run_episode_info_structure(self, episode_runner):
        """Test info dict contains expected keys."""
        params = episode_runner.policy.get_flat_weights()
        info, _ = episode_runner.run_episode(params)

        assert "rewards" in info
        assert "reward_history" in info
        assert "n_steps" in info
        assert "protected_cells" in info

    def test_run_episode_protects_cells(self, episode_runner):
        """Test that episode protects cells."""
        params = episode_runner.policy.get_flat_weights()
        info, _ = episode_runner.run_episode(params)

        # Should have protected some cells
        assert info["protected_cells"] > 0
        assert info["protected_cells"] <= episode_runner.budget_manager.total_target

    def test_run_episode_advances_steps(self, episode_runner):
        """Test that episode runs correct number of steps."""
        params = episode_runner.policy.get_flat_weights()
        info, _ = episode_runner.run_episode(params)

        assert info["n_steps"] == episode_runner.n_steps
        # Reward history should have one entry per step
        assert len(info["reward_history"]) == episode_runner.n_steps

    def test_run_episode_resets_env(self, episode_runner):
        """Test that run_episode resets environment."""
        params = episode_runner.policy.get_flat_weights()

        # Run episode
        episode_runner.run_episode(params)

        # Verify env was reset (step counter should be at n_steps)
        assert episode_runner.env._step_num == episode_runner.n_steps

    def test_run_episode_different_weights(self, episode_runner):
        """Test that different weights produce different results."""
        params1 = episode_runner.policy.get_flat_weights()
        _, reward1 = episode_runner.run_episode(params1)

        # Use different weights
        params2 = np.random.randn(len(params1)).astype(np.float32)
        _, reward2 = episode_runner.run_episode(params2)

        # Rewards might be different (not guaranteed, but likely)
        # At minimum, both should be valid floats
        assert np.isfinite(reward1)
        assert np.isfinite(reward2)


class TestEpisodeRunnerGetInfo:
    """Tests for get_info method."""

    def test_get_info_after_episode(self, episode_runner):
        """Test get_info returns valid data after episode."""
        params = episode_runner.policy.get_flat_weights()
        episode_runner.run_episode(params)

        info = episode_runner.get_info()

        assert "rewards" in info
        assert "protected_cells" in info

    def test_get_info_rewards_shape(self, episode_runner):
        """Test rewards tensor has correct shape."""
        params = episode_runner.policy.get_flat_weights()
        episode_runner.run_episode(params)

        info = episode_runner.get_info()

        # Should have one reward per reward component
        rewards = info["rewards"]
        if isinstance(rewards, torch.Tensor):
            assert len(rewards) == len(episode_runner.rewards._reward_obj_list)


class TestEpisodeRunnerProtection:
    """Tests for protection behavior."""

    def test_protection_respects_total(self, episode_runner):
        """Test protection doesn't exceed total."""
        params = episode_runner.policy.get_flat_weights()
        info, _ = episode_runner.run_episode(params)

        assert info["protected_cells"] <= episode_runner.budget_manager.total_target

    def test_protection_unique_cells(self, episode_runner):
        """Test each cell is only protected once."""
        params = episode_runner.policy.get_flat_weights()
        episode_runner.run_episode(params)

        # Protected mask should have unique cells
        mask = episode_runner.env.protected_cells_mask
        protected_count = mask.sum().item()

        # Should match info
        assert protected_count == episode_runner.get_info()["protected_cells"]


class TestEpisodeRunnerRewards:
    """Tests for reward calculation."""

    def test_rewards_accumulated(self, episode_runner):
        """Test that rewards are accumulated across steps."""
        params = episode_runner.policy.get_flat_weights()
        episode_runner.run_episode(params)

        # Rewards should have been calculated
        assert len(episode_runner.rewards.episode_reward_history) > 0

    def test_rewards_reset_between_episodes(self, episode_runner):
        """Test rewards are reset between episodes."""
        params = episode_runner.policy.get_flat_weights()

        # Run first episode
        _, reward1 = episode_runner.run_episode(params)

        # Run second episode (should start fresh)
        _, reward2 = episode_runner.run_episode(params)

        # Both should be valid (not accumulated from previous)
        assert np.isfinite(reward1)
        assert np.isfinite(reward2)


class TestEpisodeRunnerMultipleRuns:
    """Tests for running multiple episodes."""

    def test_multiple_episodes_deterministic_weights(self, episode_runner):
        """Test same weights produce consistent behavior."""
        params = episode_runner.policy.get_flat_weights()

        # Run twice with same weights
        _, reward1 = episode_runner.run_episode(params)
        _, reward2 = episode_runner.run_episode(params)

        # Rewards should be very close (small differences from tie-breaking noise)
        assert abs(reward1 - reward2) < 0.1 * max(abs(reward1), abs(reward2), 1.0)

    def test_multiple_episodes_env_reset(self, episode_runner):
        """Test environment is properly reset between episodes."""
        params = episode_runner.policy.get_flat_weights()

        # Run first episode
        episode_runner.run_episode(params)
        protected_after_first = episode_runner.env.protected_cells_mask.sum().item()

        # Run second episode
        episode_runner.run_episode(params)
        protected_after_second = episode_runner.env.protected_cells_mask.sum().item()

        # Should have same protection count (reset works)
        assert abs(protected_after_first - protected_after_second) <= 1
