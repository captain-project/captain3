"""Tests for reward weight schedules (FactorSchedule, RewardWeightSchedule) in ES training."""

from __future__ import annotations

import copy
import csv

import numpy as np
import pytest

from captain.agents.feature_extractor import FeatureExtractor
from captain.agents.policy_network import CellNN, PolicyNetwork
from captain.agents.reward_aggregator import Rewards
from captain.agents.rewards import CalcRewardProtectedPopulation, CalcRewardWorstSpecies
from captain.algorithms.budget_manager import GlobalBudgetManager
from captain.algorithms.episode import EpisodeRunner
from captain.algorithms.evolution_train import EvolStrategiesTrainer
from captain.algorithms.scheduler import FactorSchedule, RewardWeightSchedule
from captain.algorithms.train_utils import TrainingLogger

REWARD_WEIGHTS = np.array([1.0, 0.5])


class TestFactorSchedule:
    def test_geometric(self):
        f = FactorSchedule(start=100.0, epochs=4)
        assert [f(e) for e in range(6)] == pytest.approx(
            [100.0, 100**0.75, 10.0, 100**0.25, 1.0, 1.0]
        )

    def test_linear_to_custom_end(self):
        f = FactorSchedule(start=0.0, end=2.0, epochs=4, shape="linear")
        assert [f(e) for e in (0, 2, 4, 10)] == pytest.approx([0.0, 1.0, 2.0, 2.0])

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"start": 10.0, "epochs": 0},
            {"start": 0.0, "epochs": 5},  # geometric needs start > 0
            {"start": 10.0, "epochs": 5, "shape": "cosine"},
            {"start": -1.0, "epochs": 5, "shape": "linear"},
        ],
    )
    def test_invalid(self, kwargs):
        with pytest.raises(ValueError):
            FactorSchedule(**kwargs)


class TestRewardWeightSchedule:
    def test_factors_in_reward_order(self):
        schedule = RewardWeightSchedule({"b": FactorSchedule(start=10.0, epochs=2)})
        np.testing.assert_allclose(schedule.factors(0, ["a", "b", "c"]), [1.0, 10.0, 1.0])
        np.testing.assert_allclose(schedule.factors(2, ["a", "b", "c"]), [1.0, 1.0, 1.0])

    def test_unknown_term(self):
        schedule = RewardWeightSchedule({"missing": FactorSchedule(start=10.0, epochs=2)})
        with pytest.raises(ValueError, match="missing"):
            schedule.factors(0, ["a", "b"])

    def test_requires_factor_schedules(self):
        with pytest.raises(ValueError):
            RewardWeightSchedule({})
        with pytest.raises(TypeError):
            RewardWeightSchedule({"a": 10.0})


def _runner(env, extractor):
    model = CellNN(input_dim=extractor.n_features, hidden_dim=8)
    rewards = Rewards(
        [
            CalcRewardProtectedPopulation(name="protpop"),
            CalcRewardWorstSpecies(name="worst", worst_fraction=0.4),
        ],
        reward_weights=REWARD_WEIGHTS,
        cumulative_reward=False,
    )
    return EpisodeRunner(
        env=env,
        feature_extractor=extractor,
        policy_network=PolicyNetwork(model, seed=1),
        rewards=rewards,
        budget_manager=GlobalBudgetManager(
            total_target=12, cells_per_time_step=12, feature_updates_per_time_step=3
        ),
        n_steps=1,
        env_dynamics=False,
    )


def _schedule():
    return RewardWeightSchedule({"worst": FactorSchedule(start=10.0, epochs=2)})


def _assert_epoch_used(trainer, avg_reward, summary, expected_multipliers):
    """The epoch's reward is the term rewards weighted by weights x multipliers."""
    np.testing.assert_allclose(trainer.last_reward_multipliers, expected_multipliers)
    expected = summary["avg_rewards_by_type"] @ (REWARD_WEIGHTS * expected_multipliers)
    assert avg_reward == pytest.approx(expected, rel=1e-5)


class TestTrainerSchedule:
    def test_multipliers_follow_schedule(self, small_env, small_feature_extractor):
        runner = _runner(small_env, small_feature_extractor)
        trainer = EvolStrategiesTrainer(
            [runner],
            runner.policy.get_flat_weights(),
            n_perturbations=3,
            seed=1,
            reward_weight_schedule=_schedule(),
        )
        for expected in ([1.0, 10.0], [1.0, 10**0.5], [1.0, 1.0], [1.0, 1.0]):
            avg_reward, summary = trainer.train_epoch()
            _assert_epoch_used(trainer, avg_reward, summary, np.array(expected))
        assert trainer.epoch == 4

    def test_schedule_on_top_of_calibration(self, small_env, small_feature_extractor):
        runner = _runner(small_env, small_feature_extractor)
        trainer = EvolStrategiesTrainer(
            [runner],
            runner.policy.get_flat_weights(),
            n_perturbations=2,
            seed=1,
            reward_weight_schedule=_schedule(),
        )
        trainer.calibrate_reward_scales(np.array([2.0, 3.0]))
        avg_reward, summary = trainer.train_epoch()
        _assert_epoch_used(trainer, avg_reward, summary, np.array([2.0, 30.0]))

    def test_unknown_term_rejected(self, small_env, small_feature_extractor):
        runner = _runner(small_env, small_feature_extractor)
        schedule = RewardWeightSchedule({"cost": FactorSchedule(start=10.0, epochs=2)})
        with pytest.raises(ValueError, match="cost"):
            EvolStrategiesTrainer(
                [runner],
                runner.policy.get_flat_weights(),
                n_perturbations=2,
                reward_weight_schedule=schedule,
            )

    def test_no_schedule_keeps_calibration(self, small_env, small_feature_extractor):
        runner = _runner(small_env, small_feature_extractor)
        trainer = EvolStrategiesTrainer(
            [runner], runner.policy.get_flat_weights(), n_perturbations=2, seed=1
        )
        trainer.calibrate_reward_scales(np.array([2.0, 3.0]))
        avg_reward, summary = trainer.train_epoch()
        _assert_epoch_used(trainer, avg_reward, summary, np.array([2.0, 3.0]))

    def test_checkpoint_keeps_epoch_and_calibration(
        self, small_env, small_feature_extractor, tmp_path
    ):
        runner = _runner(small_env, small_feature_extractor)
        kwargs = {"n_perturbations": 2, "seed": 1, "reward_weight_schedule": _schedule()}
        trainer = EvolStrategiesTrainer([runner], runner.policy.get_flat_weights(), **kwargs)
        trainer.calibrate_reward_scales(np.array([2.0, 3.0]))
        trainer.train_epoch()
        trainer.save_checkpoint(tmp_path / "ckpt.npz")

        resumed = EvolStrategiesTrainer([runner], runner.policy.get_flat_weights(), **kwargs)
        resumed.load_checkpoint(tmp_path / "ckpt.npz")
        assert resumed.epoch == 1
        np.testing.assert_allclose(resumed.reward_multipliers(), [2.0, 3.0 * 10**0.5])

    def test_log_has_multiplier_columns(self, small_env, small_feature_extractor, tmp_path):
        runner = _runner(small_env, small_feature_extractor)
        trainer = EvolStrategiesTrainer(
            [runner],
            runner.policy.get_flat_weights(),
            n_perturbations=2,
            seed=1,
            reward_weight_schedule=_schedule(),
        )
        log = TrainingLogger(trainer, runner, tmp_path, "log.tsv", "w.npy", plot_freq=1000)
        for epoch in range(2):
            avg_reward, summary = trainer.train_epoch()
            log.log_epoch(epoch, avg_reward, summary, 0.0)
        with open(tmp_path / "log.tsv") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        assert [float(r["mult_worst"]) for r in rows] == pytest.approx([10.0, 10**0.5])
        assert [float(r["mult_protpop"]) for r in rows] == pytest.approx([1.0, 1.0])

    def test_log_without_schedule_unchanged(self, small_env, small_feature_extractor, tmp_path):
        runner = _runner(small_env, small_feature_extractor)
        trainer = EvolStrategiesTrainer(
            [runner], runner.policy.get_flat_weights(), n_perturbations=2, seed=1
        )
        TrainingLogger(trainer, runner, tmp_path, "log.tsv", "w.npy", plot_freq=1000)
        with open(tmp_path / "log.tsv") as f:
            header = f.readline().strip().split("\t")
        assert not any(h.startswith("mult_") for h in header)


def test_pool_workers_use_scheduled_multipliers(small_env):
    """With worker processes every episode uses the epoch's multipliers."""
    # one environment per worker, as in the training scripts (workers share tensor memory)
    envs = [small_env, copy.deepcopy(small_env)]
    runners = [_runner(env, FeatureExtractor(env)) for env in envs]
    trainer = EvolStrategiesTrainer(
        runners,
        runners[0].policy.get_flat_weights(),
        n_perturbations=6,
        seed=1,
        reward_weight_schedule=_schedule(),
    )
    try:
        assert trainer._use_pool
        trainer.calibrate_reward_scales(np.array([2.0, 3.0]))
        for expected in ([2.0, 30.0], [2.0, 3.0 * 10**0.5], [2.0, 3.0]):
            avg_reward, summary = trainer.train_epoch()
            _assert_epoch_used(trainer, avg_reward, summary, np.array(expected))
    finally:
        trainer.close()
