"""Unit tests for the evolution strategies update and advantage normalization."""

from __future__ import annotations

import numpy as np
import pytest

from captain.agents.reward_aggregator import NoRewards
from captain.algorithms.evolution_train import (
    EvolStrategiesTrainer,
    compute_evolutionary_update,
    normalize_advantages,
)


def _update(rewards, noise, method=None, running_reward=0.0, alpha=0.1, sigma=0.2):
    results = [({}, r) for r in rewards]
    return compute_evolutionary_update(results, np.zeros(noise.shape[1]), noise, alpha, sigma,
                                       running_reward, advantage_normalization=method)


@pytest.fixture
def setup():
    rng = np.random.default_rng(0)
    return rng.normal(size=8), rng.normal(size=(8, 5))


class TestNormalizeAdvantages:
    def test_none_is_reward_minus_baseline(self):
        assert np.allclose(normalize_advantages(np.array([1.0, 3.0]), None, 2.0), [-1.0, 1.0])

    def test_zscore(self):
        adv = normalize_advantages(np.array([1.0, 2.0, 3.0]), "zscore", None)
        assert adv.mean() == pytest.approx(0.0)
        assert adv.std() == pytest.approx(1.0)

    def test_rank(self):
        adv = normalize_advantages(np.array([10.0, -5.0, 3.0]), "rank", None)
        assert np.allclose(adv, [0.5, -0.5, 0.0])

    def test_constant_rewards_give_no_update(self):
        assert np.all(normalize_advantages(np.full(4, 7.0), "zscore", None) == 0)
        assert np.all(normalize_advantages(np.full(1, 7.0), "rank", None) == 0)

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="normalization"):
            normalize_advantages(np.ones(3), "minmax", None)


class TestComputeUpdate:
    def test_default_matches_previous_formula(self, setup):
        rewards, noise = setup
        expected = 0.1 / (8 * 0.2) * ((rewards - 0.3)[:, None] * noise).sum(0)

        assert np.allclose(_update(rewards, noise, None, running_reward=0.3), expected)

    def test_zscore_invariant_to_reward_scale(self, setup):
        """Multiplying and shifting all rewards does not change the update."""
        rewards, noise = setup
        base = _update(rewards, noise, "zscore")

        assert np.allclose(_update(30 * rewards + 5, noise, "zscore"), base)
        # without normalization, the step grows with the reward scale
        assert np.linalg.norm(_update(30 * rewards, noise)) == pytest.approx(
            30 * np.linalg.norm(_update(rewards, noise)))

    def test_rank_invariant_to_monotonic_transform(self, setup):
        rewards, noise = setup
        assert np.allclose(_update(np.exp(rewards), noise, "rank"), _update(rewards, noise, "rank"))

    @pytest.mark.parametrize("method", ["zscore", "rank"])
    def test_relative_term_weights_still_matter(self, method):
        """Re-weighting two reward terms changes the update; scaling both does not."""
        rng = np.random.default_rng(1)
        term_a, term_b = rng.normal(size=10), rng.normal(size=10)
        noise = rng.normal(size=(10, 4))

        balanced = _update(term_a + term_b, noise, method)
        b_heavy = _update(term_a + 5 * term_b, noise, method)
        both_scaled = _update(7 * term_a + 7 * term_b, noise, method)

        assert not np.allclose(balanced, b_heavy)
        assert np.allclose(balanced, both_scaled)


def test_trainer_rejects_invalid_normalization():
    with pytest.raises(ValueError, match="advantage_normalization"):
        EvolStrategiesTrainer([object()], initial_coeffs=np.zeros(3), n_perturbations=2,
                              advantage_normalization="minmax")


def test_trainer_defaults_to_zscore():
    class Runner:  # minimal stand-in: the trainer only reads .rewards.names
        rewards = NoRewards()

    trainer = EvolStrategiesTrainer([Runner()], initial_coeffs=np.zeros(3), n_perturbations=2)
    assert trainer.advantage_normalization == "zscore"
