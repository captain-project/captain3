"""Tests for the ES learning-rate / noise scheduler."""

from __future__ import annotations

import pytest

from captain.algorithms.scheduler import TARGET_IOU, LearningScheduler


def test_adaptive_sigma_respects_floor():
    sched = LearningScheduler(initial_sigma=0.06, min_sigma=0.05)
    for _ in range(20):
        sched.step(jaccard_indx=0.0)  # perturbed plans disagree -> sigma shrinks
    assert sched.sigma == pytest.approx(0.05)


def test_adaptive_sigma_grows_when_plans_agree():
    sched = LearningScheduler(initial_sigma=0.1)
    sched.step(jaccard_indx=1.0)
    assert sched.sigma == pytest.approx(0.11)


def test_adaptive_sigma_unchanged_near_target():
    sched = LearningScheduler(initial_sigma=0.1)
    sched.step(jaccard_indx=TARGET_IOU)
    assert sched.sigma == pytest.approx(0.1)


def test_decay_without_jaccard_respects_floor():
    sched = LearningScheduler(initial_sigma=0.02, sigma_decay=0.5, min_sigma=0.01)
    for _ in range(5):
        sched.step()
    assert sched.sigma == pytest.approx(0.01)
