"""Integration tests for Evolution Strategies training.

Note: These tests don't use the full multiprocessing pool to avoid
complexity in test environment. They test the core logic.
"""

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
from captain.algorithms.evolution_train import (
    EvolStrategiesTrainer,
    compute_evolutionary_update,
)
from captain.algorithms.scheduler import LearningScheduler
from captain.algorithms.train_utils import summarize_episodes
from captain.data.extinction_risk import ExtinctionRisk
from captain.data.spatial_data import SpatialData
from captain.environment.bioenv import BioEnv


@pytest.fixture
def simple_episode_runner(small_grid_params, rng):
    """Create a simple episode runner for testing."""
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

    feature_extractor = FeatureExtractor(env)
    model = CellNN(input_dim=feature_extractor.n_features, hidden_dim=8)
    policy = PolicyNetwork(model)

    rewards = Rewards([
        CalcRewardExtRisk(threat_weights=[1, 0, -8, -16, -32]),
    ])

    runner = EpisodeRunner(
        env=env,
        feature_extractor=feature_extractor,
        policy_network=policy,
        rewards=rewards,
        n_steps=3,
        budget_manager=GlobalBudgetManager(total_target=10, cells_per_time_step=5),
    )

    return runner


class TestComputeEvolutionaryUpdate:
    """Tests for compute_evolutionary_update function."""

    def test_update_returns_array(self):
        """Test update returns numpy array."""
        n_params = 100
        n_workers = 4

        results = [({"rewards": np.array([1.0])}, 1.0) for _ in range(n_workers)]
        epoch_coeff = np.zeros(n_params)
        noise = np.random.randn(n_workers, n_params)

        new_coeff = compute_evolutionary_update(
            results=results,
            epoch_coeff=epoch_coeff,
            noise=noise,
            alpha=0.1,
            sigma=0.1,
            running_reward=0.5,
        )

        assert isinstance(new_coeff, np.ndarray)
        assert new_coeff.shape == (n_params,)

    def test_update_no_change_if_sigma_zero(self):
        """Test no update when sigma is zero."""
        n_params = 100
        n_workers = 4

        results = [({"rewards": np.array([1.0])}, 1.0) for _ in range(n_workers)]
        epoch_coeff = np.ones(n_params)
        noise = np.random.randn(n_workers, n_params)

        new_coeff = compute_evolutionary_update(
            results=results,
            epoch_coeff=epoch_coeff,
            noise=noise,
            alpha=0.1,
            sigma=0.0,  # Zero sigma
            running_reward=0.5,
        )

        assert np.allclose(new_coeff, epoch_coeff)

    def test_update_direction(self):
        """Test update moves towards better solutions."""
        n_params = 10
        n_workers = 4

        # First two have positive reward, last two have negative
        results = [
            ({"rewards": np.array([1.0])}, 2.0),
            ({"rewards": np.array([1.0])}, 2.0),
            ({"rewards": np.array([1.0])}, -1.0),
            ({"rewards": np.array([1.0])}, -1.0),
        ]
        epoch_coeff = np.zeros(n_params)
        # Set up noise so positive rewards have positive noise
        noise = np.array([
            np.ones(n_params),
            np.ones(n_params),
            -np.ones(n_params),
            -np.ones(n_params),
        ])

        new_coeff = compute_evolutionary_update(
            results=results,
            epoch_coeff=epoch_coeff,
            noise=noise,
            alpha=1.0,
            sigma=1.0,
            running_reward=0.0,
        )

        # Should move in positive direction (towards better noise)
        assert np.mean(new_coeff) > 0


class TestSummarizeEpisodes:
    """Tests for summarize_episodes function."""

    def test_summarize_basic(self):
        """Test basic summarization."""
        results = [
            ({"rewards": np.array([1.0, 2.0]), "reward_history": [[1.0, 2.0]], "protected_cells": 10, "protection_matrix": np.zeros((1, 3, 3)), "extinction_risk": {"LC": 3, "CR": 1}}, 3.0),
            ({"rewards": np.array([2.0, 3.0]), "reward_history": [[2.0, 3.0]], "protected_cells": 15, "protection_matrix": np.zeros((1, 3, 3)), "extinction_risk": {"LC": 3, "CR": 1}}, 5.0),
        ]

        summary = summarize_episodes(results)

        assert "avg_rewards_by_type" in summary
        assert "avg_reward_history" in summary
        assert "avg_protected_cells" in summary

    def test_summarize_averages_rewards(self):
        """Test that rewards are averaged correctly."""
        results = [
            ({"rewards": np.array([1.0, 2.0]), "reward_history": [[1.0, 2.0]], "protected_cells": 10, "protection_matrix": np.zeros((1, 3, 3)), "extinction_risk": {"LC": 3, "CR": 1}}, 3.0),
            ({"rewards": np.array([3.0, 4.0]), "reward_history": [[3.0, 4.0]], "protected_cells": 20, "protection_matrix": np.zeros((1, 3, 3)), "extinction_risk": {"LC": 3, "CR": 1}}, 7.0),
        ]

        summary = summarize_episodes(results)

        expected_avg = np.array([2.0, 3.0])
        assert np.allclose(summary["avg_rewards_by_type"], expected_avg)

    def test_summarize_averages_protected_cells(self):
        """Test that protected cells are averaged correctly."""
        results = [
            ({"rewards": np.array([1.0]), "reward_history": [[1.0]], "protected_cells": 10, "protection_matrix": np.zeros((1, 3, 3)), "extinction_risk": {"LC": 3, "CR": 1}}, 1.0),
            ({"rewards": np.array([1.0]), "reward_history": [[1.0]], "protected_cells": 20, "protection_matrix": np.zeros((1, 3, 3)), "extinction_risk": {"LC": 3, "CR": 1}}, 1.0),
            ({"rewards": np.array([1.0]), "reward_history": [[1.0]], "protected_cells": 30, "protection_matrix": np.zeros((1, 3, 3)), "extinction_risk": {"LC": 3, "CR": 1}}, 1.0),
        ]

        summary = summarize_episodes(results)

        assert summary["avg_protected_cells"] == 20.0

    def test_summarize_with_torch_rewards(self):
        """Test summarization works with torch tensor rewards."""
        results = [
            ({"rewards": torch.tensor([1.0, 2.0]), "reward_history": [[1.0, 2.0]], "protected_cells": 10, "protection_matrix": np.zeros((1, 3, 3)), "extinction_risk": {"LC": 3, "CR": 1}}, 3.0),
            ({"rewards": torch.tensor([2.0, 3.0]), "reward_history": [[2.0, 3.0]], "protected_cells": 15, "protection_matrix": np.zeros((1, 3, 3)), "extinction_risk": {"LC": 3, "CR": 1}}, 5.0),
        ]

        summary = summarize_episodes(results)

        expected_avg = np.array([1.5, 2.5])
        assert np.allclose(summary["avg_rewards_by_type"], expected_avg)


class TestLearningScheduler:
    """Tests for LearningScheduler."""

    def test_scheduler_init_defaults(self):
        """Test scheduler default initialization."""
        scheduler = LearningScheduler()

        assert scheduler.alpha > 0
        assert scheduler.sigma > 0

    def test_scheduler_step(self):
        """Test scheduler step method."""
        scheduler = LearningScheduler(
            initial_alpha=1.0,
            initial_sigma=1.0,
            alpha_decay=0.5,
            sigma_decay=0.5,
        )

        initial_alpha = scheduler.alpha
        initial_sigma = scheduler.sigma

        scheduler.step()

        assert scheduler.alpha == initial_alpha * 0.5
        assert scheduler.sigma == initial_sigma * 0.5

    def test_scheduler_respects_min_values(self):
        """Test scheduler respects minimum values."""
        scheduler = LearningScheduler(
            initial_alpha=0.1,
            initial_sigma=0.1,
            alpha_decay=0.1,
            sigma_decay=0.1,
            min_alpha=0.05,
            min_sigma=0.05,
        )

        # Decay many times
        for _ in range(100):
            scheduler.step()

        assert scheduler.alpha >= scheduler.min_alpha
        assert scheduler.sigma >= scheduler.min_sigma

    def test_scheduler_state_dict(self):
        """Test scheduler state_dict and load_state_dict."""
        scheduler = LearningScheduler(initial_alpha=0.5, initial_sigma=0.3)
        scheduler.step()

        state = scheduler.state_dict()

        new_scheduler = LearningScheduler()
        new_scheduler.load_state_dict(state)

        assert new_scheduler.alpha == scheduler.alpha
        assert new_scheduler.sigma == scheduler.sigma


class TestEndToEndSimulation:
    """End-to-end integration tests (without multiprocessing)."""

    def test_single_episode_execution(self, simple_episode_runner):
        """Test running a single episode."""
        runner = simple_episode_runner
        params = runner.policy.get_flat_weights()

        info, reward = runner.run_episode(params)

        assert np.isfinite(reward)
        assert info["protected_cells"] > 0

    def test_manual_evolution_step(self, simple_episode_runner):
        """Test manual evolution step without pool."""
        runner = simple_episode_runner
        n_workers = 3

        # Current weights
        epoch_coeff = runner.policy.get_flat_weights()
        n_params = len(epoch_coeff)

        # Generate noise
        noise = np.random.randn(n_workers, n_params).astype(np.float32)

        # Run episodes manually
        results = []
        for i in range(n_workers):
            perturbed_params = epoch_coeff + 0.1 * noise[i]
            info, reward = runner.run_episode(perturbed_params)
            results.append((info, reward))

        # Compute update with running_reward offset from mean to create gradient signal
        mean_reward = np.mean([r[1] for r in results])
        new_coeff = compute_evolutionary_update(
            results=results,
            epoch_coeff=epoch_coeff,
            noise=noise,
            alpha=0.1,
            sigma=0.1,
            running_reward=mean_reward - 1.0,  # Offset to ensure non-zero advantages
        )

        # Weights should have changed (or at least be valid)
        assert new_coeff.shape == epoch_coeff.shape
        assert np.all(np.isfinite(new_coeff))

    def test_training_loop_simulation(self, simple_episode_runner):
        """Test simulated training loop."""
        runner = simple_episode_runner
        scheduler = LearningScheduler(initial_alpha=0.1, initial_sigma=0.1)

        epoch_coeff = runner.policy.get_flat_weights()
        n_params = len(epoch_coeff)
        running_reward = None

        rewards_over_time = []

        # Simulate 3 epochs
        for epoch in range(3):
            n_workers = 2
            noise = np.random.randn(n_workers, n_params).astype(np.float32)

            # Evaluate
            results = []
            for i in range(n_workers):
                perturbed_params = epoch_coeff + scheduler.sigma * noise[i]
                info, reward = runner.run_episode(perturbed_params)
                results.append((info, reward))

            avg_reward = np.mean([r[1] for r in results])
            rewards_over_time.append(avg_reward)

            if running_reward is None:
                running_reward = avg_reward
            else:
                running_reward = 0.5 * avg_reward + 0.5 * running_reward

            # Update
            epoch_coeff = compute_evolutionary_update(
                results=results,
                epoch_coeff=epoch_coeff,
                noise=noise,
                alpha=scheduler.alpha,
                sigma=scheduler.sigma,
                running_reward=running_reward,
            )

            scheduler.step()

        # Should have run without errors
        assert len(rewards_over_time) == 3
        assert all(np.isfinite(r) for r in rewards_over_time)


class TestSequentialTrainer:
    """Tests for EvolStrategiesTrainer in sequential (single-runner) mode."""

    def test_sequential_mode_init(self, simple_episode_runner):
        """Test that a single runner triggers sequential mode."""
        runner = simple_episode_runner
        weights = runner.policy.get_flat_weights()

        trainer = EvolStrategiesTrainer(
            [runner],
            initial_coeffs=weights,
            n_perturbations=5,
        )

        assert trainer._use_pool is False
        assert trainer.n == 5
        assert trainer._runner is runner

        trainer.close()  # should be a no-op

    def test_sequential_train_epoch(self, simple_episode_runner):
        """Test running a training epoch in sequential mode."""
        runner = simple_episode_runner
        weights = runner.policy.get_flat_weights()

        trainer = EvolStrategiesTrainer(
            [runner],
            initial_coeffs=weights,
            n_perturbations=3,
            scheduler=LearningScheduler(initial_alpha=0.1, initial_sigma=0.1),
        )

        avg_reward, summary = trainer.train_epoch()

        assert np.isfinite(avg_reward)
        assert "avg_rewards_by_type" in summary
        assert "avg_reward_history" in summary
        assert "avg_protected_cells" in summary
        assert trainer.running_reward is not None

        trainer.close()
