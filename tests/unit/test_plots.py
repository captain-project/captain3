"""Tests for feature-vs-score plots."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from captain.agents.reward_aggregator import NoRewards  # noqa: E402
from captain.algorithms.budget_manager import GlobalBudgetManager  # noqa: E402
from captain.algorithms.episode import EpisodeRunner  # noqa: E402
from captain.algorithms.inference_recorder import InferenceRecorder  # noqa: E402
from captain.utils import plots  # noqa: E402


class TestPlotFeatureScores:
    def test_writes_file(self, rng, tmp_path):
        features = rng.random((5, 200))
        features[1] = rng.integers(0, 3, size=200)  # discrete feature (jittered)
        features[2] = 0.5  # constant feature
        scores = rng.normal(size=200)
        selected = scores > 1.0
        outfile = tmp_path / "plot.png"

        plots.plot_feature_scores(
            features, scores, selected, feature_names=list("abcde"), outfile=outfile,
            max_points=100,
        )
        assert outfile.exists()

    def test_shape_mismatch(self, rng):
        with pytest.raises(ValueError):
            plots.plot_feature_scores(rng.random((3, 10)), rng.random(9), np.zeros(9, bool))
        with pytest.raises(ValueError):
            plots.plot_feature_scores(rng.random((3, 10)), rng.random(10), np.zeros(9, bool))

    @pytest.mark.parametrize("fraction", [None, 0.1])
    def test_from_recorder_file(
        self, small_env, small_feature_extractor, small_policy, tmp_path, fraction
    ):
        recorder = InferenceRecorder(tmp_path, steps=[0, 1], feature_sample_fraction=fraction)
        runner = EpisodeRunner(
            env=small_env,
            feature_extractor=small_feature_extractor,
            policy_network=small_policy,
            rewards=NoRewards(),
            budget_manager=GlobalBudgetManager(total_target=20, cells_per_time_step=5),
            n_steps=3,
            recorder=recorder,
        )
        runner.run_episode()
        assert [f.name for f in recorder.files] == ["step_000_u00.npz", "step_001_u00.npz"]

        for step_file in recorder.files:
            outfile = step_file.with_suffix(".png")
            plots.plot_feature_scores_from_file(step_file, outfile=outfile)
            assert outfile.exists()
