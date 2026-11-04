"""Tests for static single-step episodes (budget rounding and env_dynamics=False)."""

from __future__ import annotations

import pytest
import torch

from captain.agents.reward_aggregator import NoRewards
from captain.algorithms.budget_manager import GlobalBudgetManager, RegionalBudgetManager
from captain.algorithms.episode import EpisodeRunner


class TestBudgetRounding:
    def test_global_cells_per_step_rounds_up(self):
        budget = GlobalBudgetManager(
            total_target=23, cells_per_time_step=23, feature_updates_per_time_step=5
        )
        assert budget.cells_per_step == 5  # 4 would leave 3 cells unprotected

    def test_regional_cells_per_step_rounds_up(self):
        budget = RegionalBudgetManager(
            masks={"a": None},
            total_targets={"a": 23},
            cells_per_time_step={"a": 23},
            feature_updates_per_time_step=5,
        )
        assert budget.per_step_targets["a"] == 5


def _single_step_runner(env, extractor, policy, total, n_updates, env_dynamics):
    budget = GlobalBudgetManager(
        total_target=total, cells_per_time_step=total, feature_updates_per_time_step=n_updates
    )
    return EpisodeRunner(
        env=env,
        feature_extractor=extractor,
        policy_network=policy,
        rewards=NoRewards(),
        budget_manager=budget,
        n_steps=1,
        env_dynamics=env_dynamics,
    )


class TestStaticSingleStepEpisode:
    @pytest.mark.parametrize("total,n_updates", [(23, 5), (20, 4), (7, 7)])
    def test_reaches_target_in_one_step(
        self, small_env, small_feature_extractor, small_policy, total, n_updates
    ):
        runner = _single_step_runner(
            small_env, small_feature_extractor, small_policy, total, n_updates, False
        )
        info, _ = runner.run_episode()
        assert info["protected_cells"] == total

    def test_no_dynamics_keeps_populations(
        self, small_env, small_feature_extractor, small_policy
    ):
        runner = _single_step_runner(
            small_env, small_feature_extractor, small_policy, 10, 2, env_dynamics=False
        )
        small_env.reset()
        init_h = small_env.h.clone()
        info, _ = runner.run_episode()
        assert info["protected_cells"] == 10
        assert small_env._env_step_num == 0  # no population step was run
        assert torch.equal(small_env.h, init_h)

    def test_dynamics_run_by_default(self, small_env, small_feature_extractor, small_policy):
        runner = _single_step_runner(
            small_env, small_feature_extractor, small_policy, 10, 2, env_dynamics=True
        )
        runner.run_episode()
        assert small_env._env_step_num == 1
