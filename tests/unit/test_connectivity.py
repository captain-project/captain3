"""Unit tests for PAConnectivity, CalcRewardConnectivity and the pa_linking feature."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from captain.agents.rewards import CalcRewardConnectivity
from captain.utils.connectivity import PAConnectivity

SHAPE = (5, 14)


def _coords(shape=SHAPE):
    rows, cols = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    return rows.ravel(), cols.ravel()


def _flat(grid):
    return np.asarray(grid).ravel()


def _two_units(shape=SHAPE):
    """Unit 1 = columns 0-1, unit 2 = columns 12-13: 10 unprotected columns in between."""
    units = np.zeros(shape, int)
    units[:, :2] = 1
    units[:, -2:] = 2
    return units


def _make(units, alpha=5.0, area_exponent=1.0, resistance=None, **kwargs):
    units = _flat(units)
    if resistance is None:
        resistance = np.ones(units.size)
    return PAConnectivity(
        _coords(), units, resistance, initial_protection=units > 0, alpha=alpha,
        area_exponent=area_exponent, **kwargs,
    )


class TestPAConnectivity:
    def test_gap_counts_unprotected_cells(self):
        """With unit resistance the gap is the number of cells to cross."""
        units = _two_units()
        conn = _make(units)

        assert conn.gaps(_flat(units > 0))[0, 1] == pytest.approx(10.0)

        protected = units > 0
        protected[2, 2:7] = True  # 5 cells of a straight link
        assert conn.gaps(_flat(protected))[0, 1] == pytest.approx(5.0)

    def test_gap_follows_least_cost_route(self):
        """High resistance in the middle row is avoided by the least-cost path."""
        units = _two_units()
        res = np.ones(SHAPE)
        res[:, 6] = 10.0
        res[0, 6] = 1.0  # one cheap crossing point
        conn = _make(units, resistance=_flat(res))
        res[0, 6] = 10.0
        blocked = _make(units, resistance=_flat(res))

        assert conn.gaps(_flat(units > 0))[0, 1] == pytest.approx(10.0)
        assert blocked.gaps(_flat(units > 0))[0, 1] == pytest.approx(19.0)

    def test_eca_gain_bounds(self):
        """Gain is 0 at the initial protection and 1 once all units are connected."""
        units = _two_units()
        conn = _make(units)
        connected = units > 0
        connected[2, :] = True

        assert conn.eca_gain(_flat(units > 0)) == pytest.approx(0.0)
        assert conn.eca_gain(_flat(connected)) == pytest.approx(1.0)
        assert conn.gaps(_flat(connected))[0, 1] == 0.0

    def test_eca_formula(self):
        """ECA = sqrt(sum_ij w_i w_j exp(-gap_ij / alpha)) with w = area."""
        units = _two_units()
        conn = _make(units, alpha=4.0)
        protected = units > 0
        protected[2, 2:5] = True  # gap 7
        w = np.array([10.0, 10.0])
        p = np.exp(-np.array([[0, 7.0], [7.0, 0]]) / 4.0)

        assert conn.eca(_flat(protected)) == pytest.approx(np.sqrt(w @ p @ w))

    def test_gain_increases_as_gap_shrinks(self):
        """The reward is continuous: each cell that shortens the link helps."""
        units = _two_units()
        conn = _make(units)
        gains = []
        for k in range(0, 11, 2):
            protected = units > 0
            protected[2, 2:2 + k] = True
            gains.append(conn.eca_gain(_flat(protected)))

        assert all(b > a for a, b in zip(gains, gains[1:], strict=False))

    def test_area_exponent(self):
        """area_exponent=0 weighs a small unit as much as a large one."""
        units = np.zeros(SHAPE, int)
        units[:, :2] = 1  # 10 cells
        units[2, -1] = 2  # 1 cell
        large = _make(units, area_exponent=1.0)
        equal = _make(units, area_exponent=0.0)
        protected = units > 0
        protected[2, 2:13] = True  # connects them

        assert large._w.tolist() == [10.0, 1.0]
        assert equal._w.tolist() == [1.0, 1.0]
        assert equal.eca_gain(_flat(protected)) == pytest.approx(1.0)

    def test_linking_value_peaks_on_link(self):
        """Cells on the least-cost link score 1, cells off it less, protected cells 0."""
        units = _two_units()
        res = np.full(SHAPE, 5.0)
        res[2, :] = 1.0  # cheap middle row = the least-cost link
        conn = _make(units, beta=0.5, resistance=_flat(res))
        value = conn.linking_value(_flat(units > 0)).reshape(SHAPE)

        assert torch.all(value[units > 0] == 0)
        assert torch.allclose(value[2, 2:12], torch.ones(10))
        assert torch.all(value[0, 2:12] < 0.01)

    def test_linking_value_zero_when_connected(self):
        units = _two_units()
        conn = _make(units)
        connected = units > 0
        connected[2, :] = True

        assert torch.all(conn.linking_value(_flat(connected)) == 0)

    def test_feature_refresh_and_reset(self):
        """With refresh_every > 1 the cache is reused, but recomputed after a reset."""
        units = _two_units()
        conn = _make(units, feature_refresh_every=3)
        start = _flat(units > 0)
        first = conn.linking_value(start)
        cell = 2 * SHAPE[1] + 2
        grown = start.copy()
        grown[cell] = True
        cached = conn.linking_value(grown)  # 2nd call: reuses cache, zeroes the new cell
        others = torch.ones(start.size, dtype=torch.bool)
        others[cell] = False

        assert cached[cell] == 0
        assert torch.allclose(cached[others], first[others])
        # protection shrinking back = new episode: recomputed from the start state
        assert torch.allclose(conn.linking_value(start), first)

    @pytest.mark.parametrize("kwargs", [{"alpha": 0.0}, {"beta": -1.0}, {"feature_refresh_every": 0}])
    def test_invalid_args(self, kwargs):
        args = {"alpha": 5.0, **kwargs}
        with pytest.raises(ValueError):
            _make(_two_units(), **args)

    def test_needs_two_units(self):
        units = np.zeros(SHAPE, int)
        units[:, :2] = 1
        with pytest.raises(ValueError, match="two"):
            _make(units)


class TestCalcRewardConnectivity:
    def test_reward_matches_eca_gain(self, small_env):
        """The reward reads the env's protection and returns the scaled ECA gain."""
        n = small_env.n_cells
        units = np.zeros(n, int)
        units[:3] = 1
        units[-3:] = 2
        small_env.update_protection_matrix(np.flatnonzero(units > 0))
        conn = PAConnectivity(small_env.sdms._coords, units, np.ones(n),
                              initial_protection=units > 0, alpha=5.0)
        reward = CalcRewardConnectivity(conn, rescaler=2.0)

        assert reward.calc_reward(small_env) == pytest.approx(0.0)
        small_env.update_protection_matrix(list(range(n)))
        assert reward.calc_reward(small_env) == pytest.approx(2.0)
        assert reward.name == "connectivity"


class TestPALinkingFeature:
    def test_feature_matches_connectivity(self, small_env):
        from captain.agents.feature_extractor import FeatureExtractor

        n = small_env.n_cells
        units = np.zeros(n, int)
        units[:3] = 1
        units[-3:] = 2
        small_env.update_protection_matrix(np.flatnonzero(units > 0))
        conn = PAConnectivity(small_env.sdms._coords, units, np.ones(n),
                              initial_protection=units > 0, alpha=5.0)
        fe = FeatureExtractor(small_env, feature_set=["pa_linking"], connectivity=conn)

        assert torch.allclose(fe.observe(small_env)[0],
                              conn.linking_value(small_env.protected_cells_mask))

    def test_requires_connectivity(self, small_env):
        from captain.agents.feature_extractor import FeatureExtractor

        with pytest.raises(ValueError, match="connectivity"):
            FeatureExtractor(small_env, feature_set=["pa_linking"])
