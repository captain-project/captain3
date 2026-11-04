"""Tests for InferenceRecorder (saving features and cell rankings at inference)."""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch

from captain.agents.policy_network import CellNN, RegionalPolicyNetwork
from captain.agents.reward_aggregator import NoRewards
from captain.algorithms.budget_manager import GlobalBudgetManager, RegionalBudgetManager
from captain.algorithms.episode import EpisodeRunner
from captain.algorithms.inference_recorder import InferenceRecorder, rank_cells

CELLS_PER_STEP = 5
TOTAL_TARGET = 20
N_STEPS = 6


def _global_runner(env, extractor, policy, recorder):
    budget = GlobalBudgetManager(total_target=TOTAL_TARGET, cells_per_time_step=CELLS_PER_STEP)
    return EpisodeRunner(
        env=env,
        feature_extractor=extractor,
        policy_network=policy,
        rewards=NoRewards(),
        budget_manager=budget,
        n_steps=N_STEPS,
        recorder=recorder,
    )


class TestRankCells:
    def test_global_rank(self):
        scores = np.array([0.1, 0.9, 0.5, 0.7])
        eligible = np.array([True, True, False, True])
        np.testing.assert_array_equal(rank_cells(scores, eligible), [3, 1, -1, 2])

    def test_grouped_rank(self):
        scores = np.array([0.1, 0.9, 0.5, 0.7])
        eligible = np.ones(4, dtype=bool)
        groups = [np.array([True, True, False, False]), np.array([False, False, True, True])]
        np.testing.assert_array_equal(rank_cells(scores, eligible, groups), [2, 1, 2, 1])


class TestInferenceRecorder:
    def test_invalid_steps(self, tmp_path):
        with pytest.raises(ValueError):
            InferenceRecorder(tmp_path, steps="first")

    def test_global_all_steps(self, small_env, small_feature_extractor, small_policy, tmp_path):
        recorder = InferenceRecorder(tmp_path, steps="all")
        runner = _global_runner(small_env, small_feature_extractor, small_policy, recorder)
        runner.run_episode()

        n_cells = small_env.n_cells
        n_features = small_feature_extractor.n_features
        n_decisions = TOTAL_TARGET // CELLS_PER_STEP
        step_files = sorted(tmp_path.glob("step_*.npz"))
        assert [f.name for f in step_files] == [f"step_{t:03d}_u00.npz" for t in range(n_decisions)]

        cells = np.load(tmp_path / "cells.npz")
        assert cells["coords_row"].shape == (n_cells,)
        assert tuple(cells["grid_shape"]) == small_env.sdms._data_shape[1:]
        assert list(cells["feature_names"]) == small_feature_extractor.feature_names

        n_protected_before = 0
        for f in step_files:
            d = np.load(f)
            assert d["features_input"].shape == (n_features, n_cells)
            assert d["features_raw"].shape == (n_features, n_cells)
            assert d["feature_names"].shape == (n_features,)
            assert d["scores"].shape == (n_cells,)
            assert "region_id" not in d

            # Masks reflect the state before the decision was applied
            assert d["already_protected"].sum() == n_protected_before
            assert not (d["selected"] & ~d["eligible"]).any()
            assert d["selected"].sum() == CELLS_PER_STEP
            n_protected_before += CELLS_PER_STEP

            rank = d["rank"]
            np.testing.assert_array_equal(rank == -1, ~d["eligible"])
            top_k = (rank > 0) & (rank <= CELLS_PER_STEP)
            np.testing.assert_array_equal(d["selected"], top_k)

            transformed = small_feature_extractor.transform(torch.from_numpy(d["features_raw"]))
            np.testing.assert_allclose(d["features_input"], transformed.numpy(), atol=1e-5)

    def test_selected_steps(
        self, small_env, small_feature_extractor, small_policy, tmp_path, caplog
    ):
        recorder = InferenceRecorder(tmp_path, steps=[0, N_STEPS - 1])
        runner = _global_runner(small_env, small_feature_extractor, small_policy, recorder)
        with caplog.at_level(logging.WARNING):
            runner.run_episode()

        assert [f.name for f in tmp_path.glob("step_*.npz")] == ["step_000_u00.npz"]
        assert f"[{N_STEPS - 1}]" in caplog.text

    def test_regional(self, small_env, small_feature_extractor, tmp_path):
        n_cells = small_env.n_cells
        idx = torch.arange(n_cells)
        masks = {"west": idx < n_cells // 2, "east": idx >= n_cells // 2}
        budget = RegionalBudgetManager(
            masks=masks,
            total_targets={"west": 6, "east": 6},
            cells_per_time_step={"west": 3, "east": 3},
        )
        model = CellNN(input_dim=small_feature_extractor.n_features, hidden_dim=16)
        recorder = InferenceRecorder(tmp_path, steps=[0])
        runner = EpisodeRunner(
            env=small_env,
            feature_extractor=small_feature_extractor,
            policy_network=RegionalPolicyNetwork(model),
            rewards=NoRewards(),
            budget_manager=budget,
            n_steps=3,
            recorder=recorder,
        )
        runner.run_episode()

        cells = np.load(tmp_path / "cells.npz")
        assert list(cells["region_names"]) == ["west", "east"]

        d = np.load(tmp_path / "step_000_u00.npz")
        np.testing.assert_array_equal(d["region_id"] == 0, masks["west"].numpy())
        for i in range(2):
            in_region = d["region_id"] == i
            region_rank = d["rank"][in_region]
            assert region_rank.max() == in_region.sum()
            np.testing.assert_array_equal(
                d["selected"][in_region], (region_rank > 0) & (region_rank <= 3)
            )

    def test_invalid_sample_fraction(self, tmp_path):
        with pytest.raises(ValueError):
            InferenceRecorder(tmp_path, feature_sample_fraction=0.0)

    @pytest.mark.parametrize("fraction", [0.2, 0.01])
    def test_feature_sampling_global(
        self, small_env, small_feature_extractor, small_policy, tmp_path, fraction
    ):
        recorder = InferenceRecorder(tmp_path, feature_sample_fraction=fraction, seed=1)
        runner = _global_runner(small_env, small_feature_extractor, small_policy, recorder)
        runner.run_episode()

        n_features = small_feature_extractor.n_features
        for f in sorted(tmp_path.glob("step_*.npz")):
            d = np.load(f)
            idx, is_top, prob = d["sample_cell_idx"], d["sample_is_top"], d["sample_inclusion_prob"]
            n_eligible = d["eligible"].sum()
            n = max(int(np.ceil(fraction * n_eligible)), CELLS_PER_STEP)

            # Full per-cell arrays are unaffected
            assert d["scores"].shape == d["rank"].shape == (small_env.n_cells,)
            # Equal-sized top and random samples of distinct, eligible cells
            assert is_top.sum() == (~is_top).sum() == n
            assert len(np.unique(idx)) == idx.size
            assert d["eligible"][idx].all()
            np.testing.assert_array_equal(np.sort(idx[is_top]), np.flatnonzero(
                (d["rank"] > 0) & (d["rank"] <= n)
            ))
            assert d["selected"][idx[is_top]].sum() == CELLS_PER_STEP
            assert (d["rank"][idx[~is_top]] > n).all()
            np.testing.assert_allclose(prob[is_top], 1.0)
            np.testing.assert_allclose(prob[~is_top], n / (n_eligible - n))

            assert d["features_input"].shape == (n_features, idx.size)
            assert d["features_raw"].shape == (n_features, idx.size)
            transformed = small_feature_extractor.transform(torch.from_numpy(d["features_raw"]))
            np.testing.assert_allclose(d["features_input"], transformed.numpy(), atol=1e-5)

    def test_feature_sampling_regional(self, small_env, small_feature_extractor, tmp_path):
        n_cells = small_env.n_cells
        idx = torch.arange(n_cells)
        masks = {"west": idx < n_cells // 2, "east": idx >= n_cells // 2}
        budget = RegionalBudgetManager(
            masks=masks,
            total_targets={"west": 6, "east": 6},
            cells_per_time_step={"west": 3, "east": 3},
        )
        model = CellNN(input_dim=small_feature_extractor.n_features, hidden_dim=16)
        recorder = InferenceRecorder(tmp_path, steps=[0], feature_sample_fraction=0.1, seed=1)
        runner = EpisodeRunner(
            env=small_env,
            feature_extractor=small_feature_extractor,
            policy_network=RegionalPolicyNetwork(model),
            rewards=NoRewards(),
            budget_manager=budget,
            n_steps=3,
            recorder=recorder,
        )
        runner.run_episode()

        d = np.load(tmp_path / "step_000_u00.npz")
        sample_region = d["region_id"][d["sample_cell_idx"]]
        for i, mask in enumerate(masks.values()):
            n = max(int(np.ceil(0.1 * mask.sum().item())), 3)
            in_region = sample_region == i
            assert d["sample_is_top"][in_region].sum() == n
            assert (~d["sample_is_top"][in_region]).sum() == n

    @pytest.mark.parametrize("fraction", [None, 0.1])
    def test_sample_coords(
        self, small_env, small_feature_extractor, small_policy, tmp_path, fraction
    ):
        recorder = InferenceRecorder(
            tmp_path, steps=[0], feature_sample_fraction=fraction, save_sample_coords=True
        )
        runner = _global_runner(small_env, small_feature_extractor, small_policy, recorder)
        runner.run_episode()

        cells = np.load(tmp_path / "cells.npz")
        d = np.load(tmp_path / "step_000_u00.npz")
        idx = d["sample_cell_idx"] if fraction else np.arange(small_env.n_cells)
        assert d["sample_coords_row"].shape == idx.shape
        np.testing.assert_array_equal(d["sample_coords_row"], cells["coords_row"][idx])
        np.testing.assert_array_equal(d["sample_coords_col"], cells["coords_col"][idx])

    def test_sample_coords_off_by_default(
        self, small_env, small_feature_extractor, small_policy, tmp_path
    ):
        recorder = InferenceRecorder(tmp_path, steps=[0])
        runner = _global_runner(small_env, small_feature_extractor, small_policy, recorder)
        runner.run_episode()
        assert "sample_coords_row" not in np.load(tmp_path / "step_000_u00.npz")

    def test_feature_sampling_seed(self, small_env, small_feature_extractor, small_policy, tmp_path):
        samples = []
        for run in range(2):
            out = tmp_path / str(run)
            recorder = InferenceRecorder(out, steps=[0], feature_sample_fraction=0.1, seed=7)
            runner = _global_runner(small_env, small_feature_extractor, small_policy, recorder)
            runner.run_episode()
            samples.append(np.load(out / "step_000_u00.npz")["sample_cell_idx"])
        np.testing.assert_array_equal(samples[0], samples[1])

    def test_no_recorder(self, small_env, small_feature_extractor, small_policy, tmp_path):
        runner = _global_runner(small_env, small_feature_extractor, small_policy, None)
        info, _ = runner.run_episode()
        assert info["protected_cells"] == TOTAL_TARGET
        assert not any(tmp_path.iterdir())
