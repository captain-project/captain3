"""Unit tests for Rewards classes."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from captain.agents.reward_aggregator import Rewards
from captain.agents.rewards import (
    CalcReward,
    CalcRewardExtRisk,
    CalcRewardLayerOverlap,
    CalcRewardPersistentCost,
    CalcRewardProtectedPopulation,
    CalcRewardWorstSpecies,
)


class TestCalcRewardBase:
    """Tests for CalcReward base class."""

    def test_init_defaults(self):
        """Test default initialization."""
        reward = CalcReward()

        assert reward.name == "base"
        assert reward._rescaler == 1.0

    def test_init_custom_name(self):
        """Test custom name."""
        reward = CalcReward(name="custom")

        assert reward.name == "custom"

    def test_init_negative_rescaler(self):
        """Test negative rescaler when positive=False."""
        reward = CalcReward(rescaler=2.0, positive=False)

        assert reward._rescaler == -2.0

    def test_calc_reward_returns_zero(self, small_env):
        """Test base calc_reward returns zero."""
        reward = CalcReward()

        assert reward.calc_reward(small_env) == 0.0

    def test_reset_is_noop(self):
        """Test reset does nothing (base class)."""
        reward = CalcReward()
        # Should not raise
        reward.reset()


class TestCalcRewardPersistentCost:
    """Tests for CalcRewardPersistentCost class."""

    def test_init_default_negative(self):
        """Test cost reward is negative by default."""
        reward = CalcRewardPersistentCost()

        assert reward._rescaler < 0

    def test_calc_reward_no_protection(self, small_env):
        """Test cost reward with no protection."""
        reward = CalcRewardPersistentCost(rescaler=1.0)

        # No protection = zero cost
        cost = reward.calc_reward(small_env)
        assert cost == 0.0

    def test_calc_reward_with_protection(self, small_env):
        """Test cost reward increases with protection."""
        reward = CalcRewardPersistentCost(rescaler=1.0, positive=True)

        # Add protection
        small_env.update_protection_matrix([0, 1, 2])

        cost = reward.calc_reward(small_env)
        assert cost > 0

    def test_calc_reward_respects_rescaler(self, small_env):
        """Test rescaler affects cost reward."""
        reward1 = CalcRewardPersistentCost(rescaler=1.0, positive=True)
        reward2 = CalcRewardPersistentCost(rescaler=2.0, positive=True)

        small_env.update_protection_matrix([0, 1, 2])

        cost1 = reward1.calc_reward(small_env)
        cost2 = reward2.calc_reward(small_env)

        assert abs(cost2 - 2 * cost1) < 1e-5


class TestCalcRewardExtRisk:
    """Tests for CalcRewardExtRisk class."""

    def test_init_requires_threat_weights(self):
        """Test that threat_weights is required."""
        with pytest.raises(ValueError, match="threat_weights must be specified"):
            CalcRewardExtRisk()

    def test_init_with_list_weights(self):
        """Test initialization with list weights."""
        weights = [1, 0, -8, -16, -32]
        reward = CalcRewardExtRisk(threat_weights=weights)

        assert torch.allclose(
            reward._threat_weights,
            torch.tensor(weights, dtype=torch.float32)
        )

    def test_init_with_numpy_weights(self):
        """Test initialization with numpy weights."""
        weights = np.array([1, 0, -8, -16, -32])
        reward = CalcRewardExtRisk(threat_weights=weights)

        assert reward._threat_weights.shape == (5,)

    def test_init_with_torch_weights(self):
        """Test initialization with torch weights."""
        weights = torch.tensor([1, 0, -8, -16, -32])
        reward = CalcRewardExtRisk(threat_weights=weights)

        assert reward._threat_weights.shape == (5,)

    def test_calc_reward_initial_zero(self, small_env):
        """Test first calc_reward compares to init status (should be ~0)."""
        weights = [1, 0, -8, -16, -32]
        reward = CalcRewardExtRisk(threat_weights=weights)

        # First call initializes previous_status_counts
        r = reward.calc_reward(small_env)

        # Should be zero or near zero (no change from init)
        assert abs(r) < 1.0

    def test_calc_reward_tracks_changes(self, small_env):
        """Test reward tracks changes across steps."""
        weights = [1, 0, -8, -16, -32]
        reward = CalcRewardExtRisk(threat_weights=weights)

        # Initialize
        reward.calc_reward(small_env)

        # Run some steps and check reward calculation
        for _ in range(5):
            small_env.step()

        r = reward.calc_reward(small_env)
        # Reward should be a number (may be positive or negative)
        assert isinstance(r, float)

    def test_reset_clears_state(self, small_env):
        """Test reset clears previous_status_counts."""
        weights = [1, 0, -8, -16, -32]
        reward = CalcRewardExtRisk(threat_weights=weights)

        # Initialize
        reward.calc_reward(small_env)
        assert reward._previous_status_counts is not None

        # Reset
        reward.reset()
        assert reward._previous_status_counts is None

    def test_to_device(self):
        """Test moving to device."""
        weights = [1, 0, -8, -16, -32]
        reward = CalcRewardExtRisk(threat_weights=weights)

        reward.to("cpu")

        assert reward.device == torch.device("cpu")
        assert reward._threat_weights.device.type == "cpu"


def _stub_env(init_pop, protected_pop, init_status=None):
    """Minimal env exposing what CalcRewardProtectedPopulation reads."""
    from types import SimpleNamespace

    init_pop = torch.tensor(init_pop, dtype=torch.float32)
    if init_status is None:
        init_status = torch.zeros(len(init_pop), dtype=torch.long)
    return SimpleNamespace(
        device=torch.device("cpu"),
        protected_population=torch.tensor(protected_pop, dtype=torch.float32),
        ext_risk=SimpleNamespace(
            init_pop=init_pop, init_status=torch.as_tensor(init_status, dtype=torch.long)
        ),
    )


class TestCalcRewardProtectedPopulation:
    """Tests for CalcRewardProtectedPopulation."""

    RANGES = [10.0, 100.0, 1000.0]  # one individual per cell

    def test_favours_small_ranges(self):
        """Protecting 10/10/10 cells beats 1/5/24 cells of ranges 10/100/1000."""
        reward = CalcRewardProtectedPopulation()

        even = reward.calc_reward(_stub_env(self.RANGES, [10.0, 10.0, 10.0]))
        skewed = reward.calc_reward(_stub_env(self.RANGES, [1.0, 5.0, 24.0]))

        assert even == pytest.approx((1.0 + 0.1 + 0.01) / 3)
        assert skewed == pytest.approx((0.1 + 0.05 + 0.024) / 3)
        assert even > skewed

    def test_zero_exponent_ignores_range(self):
        """range_exponent=0 sums absolute populations: equal areas tie."""
        reward = CalcRewardProtectedPopulation(range_exponent=0.0)

        even = reward.calc_reward(_stub_env(self.RANGES, [10.0, 10.0, 10.0]))
        skewed = reward.calc_reward(_stub_env(self.RANGES, [1.0, 5.0, 24.0]))

        assert even == pytest.approx(skewed)

    def test_bounded_by_full_protection(self):
        """Full protection gives 1; extra growth beyond the initial population is capped."""
        reward = CalcRewardProtectedPopulation(range_exponent=0.5)

        assert reward.calc_reward(_stub_env(self.RANGES, self.RANGES)) == pytest.approx(1.0)
        assert reward.calc_reward(_stub_env(self.RANGES, [20.0, 200.0, 2000.0])) == (
            pytest.approx(1.0)
        )
        assert reward.calc_reward(_stub_env(self.RANGES, [0.0, 0.0, 0.0])) == 0.0

    def test_threat_weights_use_initial_status(self):
        """Higher-risk species count more for the same protected fraction."""
        reward = CalcRewardProtectedPopulation(threat_weights=[1, 2, 4, 8, 16])
        status = [0, 4]  # LC, CR

        protect_lc = reward.calc_reward(_stub_env([100.0, 100.0], [50.0, 0.0], status))
        protect_cr = reward.calc_reward(_stub_env([100.0, 100.0], [0.0, 50.0], status))

        assert protect_lc == pytest.approx(0.5 * 1 / 17)
        assert protect_cr == pytest.approx(0.5 * 16 / 17)

    def test_target_saturates(self):
        """Beyond the target fraction a species earns no extra reward."""
        reward = CalcRewardProtectedPopulation(target=0.3)

        at_target = reward.calc_reward(_stub_env([100.0, 100.0], [30.0, 0.0]))
        above = reward.calc_reward(_stub_env([100.0, 100.0], [90.0, 0.0]))
        spread = reward.calc_reward(_stub_env([100.0, 100.0], [30.0, 30.0]))

        assert at_target == pytest.approx(0.5)
        assert above == pytest.approx(at_target)
        assert spread == pytest.approx(1.0)

    def test_absent_species_ignored(self):
        """Species with no initial population do not dilute the reward."""
        reward = CalcRewardProtectedPopulation()

        assert reward.calc_reward(_stub_env([100.0, 0.0], [100.0, 0.0])) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"threat_weights": [1, -1, 1, 1, 1]},
            {"threat_weights": [0, 0, 0, 0, 0]},
            {"range_exponent": -1.0},
            {"target": 0.0},
            {"target": 1.5},
        ],
    )
    def test_invalid_args(self, kwargs):
        """Invalid weights, exponent or target are rejected."""
        with pytest.raises(ValueError):
            CalcRewardProtectedPopulation(**kwargs)

    def test_too_few_threat_weights(self):
        """threat_weights must cover every initial status class."""
        reward = CalcRewardProtectedPopulation(threat_weights=[1, 2])

        with pytest.raises(ValueError, match="threat_weights"):
            reward.calc_reward(_stub_env([100.0], [10.0], [4]))

    def test_protection_increases_reward(self, small_env):
        """On a real environment, protecting cells increases the reward."""
        reward = CalcRewardProtectedPopulation()
        before = reward.calc_reward(small_env)

        small_env.update_protection_matrix(list(range(20)))
        after = reward.calc_reward(small_env)

        assert before == 0.0
        assert after > before

    def test_edge_effect_lowers_reward(self):
        """An isolated protected cell in disturbed land is worth less with an edge effect."""
        from captain.data.extinction_risk import ExtinctionRisk
        from captain.data.spatial_data import SpatialData
        from captain.environment.bioenv import BioEnv

        def run(edge_factor: float) -> float:
            shape = (1, 5, 5)
            mask = np.ones(shape[1:])
            env = BioEnv(
                sdms=SpatialData(data=np.ones(shape), mask=mask),
                disturbance=SpatialData(data=np.ones(shape), mask=mask),
                costs=SpatialData(data=np.ones(shape), mask=mask),
                protection_matrix=SpatialData(data=np.zeros(shape), mask=mask),
                growth_rates=np.array([1.1]),
                species_k=np.array([100.0]),
                sensitivity_rates=np.array([[0.5]]),
                ext_risk=ExtinctionRisk(init_status=np.array([0])),
                edge_factor=edge_factor,
            )
            env.update_protection_matrix([12])  # center cell
            for _ in range(20):
                env.step()
            return CalcRewardProtectedPopulation().calc_reward(env)

        assert run(edge_factor=0.5) < run(edge_factor=0.0)


class TestCalcRewardWorstSpecies:
    """Tests for CalcRewardWorstSpecies (soft maximin over species)."""

    INIT = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]

    def test_mean_of_worst_share(self):
        """worst_fraction=0.2 of 10 species = mean of the 2 lowest fractions."""
        protected = [50.0, 10.0, 90.0, 30.0, 70.0, 20.0, 60.0, 80.0, 40.0, 100.0]
        reward = CalcRewardWorstSpecies(worst_fraction=0.2)

        assert reward.calc_reward(_stub_env(self.INIT, protected)) == pytest.approx(0.15)

    def test_pure_maximin(self):
        """A tiny worst_fraction keeps a single species: the minimum."""
        protected = [50.0, 10.0, 90.0, 30.0, 70.0, 20.0, 60.0, 80.0, 40.0, 100.0]
        reward = CalcRewardWorstSpecies(worst_fraction=0.01)

        assert reward.calc_reward(_stub_env(self.INIT, protected)) == pytest.approx(0.10)

    def test_ignores_threat_status_and_range(self):
        """Fractions, not absolute populations: a small range counts as much as a large one."""
        reward = CalcRewardWorstSpecies(worst_fraction=0.5)
        env = _stub_env([10.0, 1000.0], [5.0, 100.0], init_status=[4, 0])

        assert reward.calc_reward(env) == pytest.approx(0.1)

    def test_target_and_absent_species(self):
        """A target caps fractions; species with no initial population are skipped."""
        reward = CalcRewardWorstSpecies(worst_fraction=0.5, target=0.5)
        env = _stub_env([100.0, 100.0, 0.0], [25.0, 90.0, 0.0])

        assert reward.calc_reward(env) == pytest.approx(0.5)  # 0.25 / 0.5 for the worst

    def test_rewards_lifting_the_worst(self):
        """Protecting the worst species raises the reward, the best does not."""
        reward = CalcRewardWorstSpecies(worst_fraction=0.1)
        base = [50.0, 10.0, 90.0, 30.0, 70.0, 20.0, 60.0, 80.0, 40.0, 100.0]
        lift_worst = list(base)
        lift_worst[1] = 45.0
        lift_best = list(base)
        lift_best[2] = 100.0

        before = reward.calc_reward(_stub_env(self.INIT, base))
        assert reward.calc_reward(_stub_env(self.INIT, lift_worst)) > before
        assert reward.calc_reward(_stub_env(self.INIT, lift_best)) == pytest.approx(before)

    def test_threat_weights_within_tail(self):
        """Weights change the mean within the tail, not which species form it."""
        protected = [50.0, 10.0, 90.0, 30.0, 70.0, 20.0, 60.0, 80.0, 40.0, 100.0]
        status = [0, 0, 0, 0, 0, 4, 0, 0, 0, 0]  # the 2nd-worst species (0.2) is CR
        weighted = CalcRewardWorstSpecies(worst_fraction=0.2, threat_weights=[1, 2, 4, 8, 16])
        env = _stub_env(self.INIT, protected, init_status=status)

        # tail = fractions 0.1 (LC, weight 1) and 0.2 (CR, weight 16)
        assert weighted.calc_reward(env) == pytest.approx((0.1 * 1 + 0.2 * 16) / 17)

    def test_threat_weights_favour_lifting_threatened(self):
        """Raising the CR species in the tail gains more than raising the LC one."""
        reward = CalcRewardWorstSpecies(worst_fraction=0.2, threat_weights=[1, 2, 4, 8, 16])
        status = [0, 0, 0, 0, 0, 4, 0, 0, 0, 0]
        base = [50.0, 10.0, 90.0, 30.0, 70.0, 20.0, 60.0, 80.0, 40.0, 100.0]
        lift_lc, lift_cr = list(base), list(base)
        lift_lc[1] = 15.0
        lift_cr[5] = 25.0

        before = reward.calc_reward(_stub_env(self.INIT, base, init_status=status))
        gain_lc = reward.calc_reward(_stub_env(self.INIT, lift_lc, init_status=status)) - before
        gain_cr = reward.calc_reward(_stub_env(self.INIT, lift_cr, init_status=status)) - before
        assert gain_cr > gain_lc > 0

    @pytest.mark.parametrize("kwargs", [{"worst_fraction": 0.0}, {"worst_fraction": 1.5},
                                        {"target": 0.0}, {"threat_weights": [1, -1, 1, 1, 1]}])
    def test_invalid_args(self, kwargs):
        with pytest.raises(ValueError):
            CalcRewardWorstSpecies(**kwargs)


class TestCalcRewardLayerOverlap:
    """Tests for CalcRewardLayerOverlap."""

    def test_protected_share_of_layer(self, small_env):
        layer = torch.zeros(small_env.n_cells)
        layer[:4] = torch.tensor([1.0, 1.0, 0.5, 0.5])
        reward = CalcRewardLayerOverlap(layer, name="trail")

        assert reward.calc_reward(small_env) == pytest.approx(0.0)
        small_env.update_protection_matrix([0, 2])
        assert reward.calc_reward(small_env) == pytest.approx(1.5 / 3.0)
        assert reward.name == "trail"

    def test_invalid_layer(self):
        with pytest.raises(ValueError):
            CalcRewardLayerOverlap(np.zeros(5))
        with pytest.raises(ValueError):
            CalcRewardLayerOverlap(np.array([1.0, -1.0]))


class TestRewards:
    """Tests for Rewards aggregator class."""

    def test_init_single_reward(self):
        """Test initialization with single reward."""
        ext_risk = CalcRewardExtRisk(threat_weights=[1, 0, -8, -16, -32])
        rewards = Rewards([ext_risk])

        assert len(rewards._reward_obj_list) == 1

    def test_init_multiple_rewards(self):
        """Test initialization with multiple rewards."""
        ext_risk = CalcRewardExtRisk(threat_weights=[1, 0, -8, -16, -32])
        cost = CalcRewardPersistentCost(rescaler=0.1)
        rewards = Rewards([ext_risk, cost])

        assert len(rewards._reward_obj_list) == 2

    def test_init_default_weights(self):
        """Test default weights are uniform."""
        ext_risk = CalcRewardExtRisk(threat_weights=[1, 0, -8, -16, -32])
        cost = CalcRewardPersistentCost(rescaler=0.1)
        rewards = Rewards([ext_risk, cost])

        assert torch.allclose(rewards._reward_weights, torch.ones(2))

    def test_init_custom_weights(self):
        """Test custom reward weights."""
        ext_risk = CalcRewardExtRisk(threat_weights=[1, 0, -8, -16, -32])
        cost = CalcRewardPersistentCost(rescaler=0.1)
        rewards = Rewards([ext_risk, cost], reward_weights=[0.8, 0.2])

        expected = torch.tensor([0.8, 0.2])
        assert torch.allclose(rewards._reward_weights, expected)

    def test_calc_reward_accumulates(self, small_env):
        """Test calc_reward accumulates to episode_rewards."""
        ext_risk = CalcRewardExtRisk(threat_weights=[1, 0, -8, -16, -32])
        rewards = Rewards([ext_risk])

        rewards.reset()
        initial_sum = rewards.episode_rewards.sum().item()

        rewards.calc_reward(small_env)

        # Should have accumulated something
        assert len(rewards.episode_reward_history) == 1

    def test_calc_total_reward(self, small_env):
        """Test calc_total_reward returns weighted sum."""
        ext_risk = CalcRewardExtRisk(threat_weights=[1, 0, -8, -16, -32])
        rewards = Rewards([ext_risk])

        rewards.reset()
        total = rewards.calc_total_reward(small_env)

        assert isinstance(total, float)

    def test_get_weighted_reward(self, small_env):
        """Test get_weighted_reward returns current total."""
        ext_risk = CalcRewardExtRisk(threat_weights=[1, 0, -8, -16, -32])
        cost = CalcRewardPersistentCost(rescaler=0.1)
        rewards = Rewards([ext_risk, cost], reward_weights=[1.0, 0.5])

        rewards.reset()
        rewards.calc_reward(small_env)

        weighted = rewards.get_weighted_reward()
        expected = (rewards.episode_rewards * rewards._reward_weights).sum().item()

        assert abs(weighted - expected) < 1e-5

    def test_reset_clears_state(self, small_env):
        """Test reset clears all state."""
        ext_risk = CalcRewardExtRisk(threat_weights=[1, 0, -8, -16, -32])
        rewards = Rewards([ext_risk])

        rewards.calc_reward(small_env)
        assert len(rewards.episode_reward_history) > 0

        rewards.reset()

        assert len(rewards.episode_reward_history) == 0
        assert torch.all(rewards.episode_rewards == 0)

    def test_names_property(self):
        """Test names property returns reward names."""
        ext_risk = CalcRewardExtRisk(
            name="ext_risk",
            threat_weights=[1, 0, -8, -16, -32]
        )
        cost = CalcRewardPersistentCost(name="cost", rescaler=0.1)
        rewards = Rewards([ext_risk, cost])

        assert rewards.names == ["ext_risk", "cost"]
