"""Schedulers for evolution strategies training.

This module provides schedulers for controlling learning rate (alpha) and noise
standard deviation (sigma) during training, and for changing the weights of reward
terms over the epochs (RewardWeightSchedule).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# HYPERPARAMETERS
# target ~80% overlap among perturbations (meaning 20% variation)
TARGET_IOU = 0.80
ADAPTATION_RATE = 0.10


@dataclass
class LearningScheduler:
    """Scheduler for evolution strategies hyperparameters.

    Implements exponential decay for learning rate (alpha) and noise
    standard deviation (sigma) with configurable floors.

    Attributes:
        alpha: Current learning rate.
        sigma: Current noise standard deviation.

    Example:
        >>> scheduler = LearningScheduler(initial_alpha=0.2, initial_sigma=0.1)
        >>> for epoch in range(100):
        ...     # Use scheduler.alpha and scheduler.sigma
        ...     scheduler.step()
    """

    # Initial values
    initial_alpha: float = 0.2
    initial_sigma: float = 0.2

    # Decay rates
    alpha_decay: float = 0.99
    sigma_decay: float = 0.99

    # Minimum values (floors)
    min_alpha: float = 0.0001
    min_sigma: float = 0.01

    target_iou_min: float = TARGET_IOU - (TARGET_IOU * 0.05)
    target_iou_max: float = TARGET_IOU + (TARGET_IOU * 0.05)

    # Current values (set in post_init)
    alpha: float = field(init=False)
    sigma: float = field(init=False)
    # Add a boolean option to lock sigma in place
    fixed_sigma: bool = False

    def __post_init__(self):
        """Initialize current values."""
        self.alpha = self.initial_alpha
        self.sigma = self.initial_sigma

    def step(self, jaccard_indx: None | float = None) -> None:
        """Apply one step of decay. Call at end of each epoch."""
        if not self.fixed_sigma:
            if jaccard_indx is not None:
                if jaccard_indx > self.target_iou_max:
                    # Solutions are too similar -> Increase exploration
                    self.sigma *= 1.0 + ADAPTATION_RATE
                elif jaccard_indx < self.target_iou_min:
                    # Solutions are too chaotic -> Decrease noise
                    self.sigma *= 1.0 - ADAPTATION_RATE
                # Keep the floor: with zscore advantages the step size scales as
                # alpha / sigma, so a collapsing sigma gives steps far larger than sigma
                self.sigma = max(self.sigma, self.min_sigma)
            else:
                self.sigma = max(self.sigma * self.sigma_decay, self.min_sigma)

        self.alpha = max(self.alpha * self.alpha_decay, self.min_alpha)

    def reset(self) -> None:
        """Reset to initial values."""
        self.alpha = self.initial_alpha
        self.sigma = self.initial_sigma

    def state_dict(self) -> dict[str, Any]:
        """Get scheduler state for checkpointing.

        Returns:
            Dictionary with current state.
        """
        return {
            "alpha": self.alpha,
            "sigma": self.sigma,
            "initial_alpha": self.initial_alpha,
            "initial_sigma": self.initial_sigma,
            "alpha_decay": self.alpha_decay,
            "sigma_decay": self.sigma_decay,
            "min_alpha": self.min_alpha,
            "min_sigma": self.min_sigma,
            "fixed_sigma": self.fixed_sigma,  # Added to state
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load scheduler state from checkpoint.

        Args:
            state: State dictionary from state_dict().
        """
        self.alpha = state["alpha"]
        self.sigma = state["sigma"]
        self.initial_alpha = state["initial_alpha"]
        self.initial_sigma = state["initial_sigma"]
        self.alpha_decay = state["alpha_decay"]
        self.sigma_decay = state["sigma_decay"]
        self.min_alpha = state["min_alpha"]
        self.min_sigma = state["min_sigma"]
        # .get() guarantees backwards compatibility with older checkpoints
        self.fixed_sigma = state.get("fixed_sigma", False)


FACTOR_SHAPES = ("geometric", "linear")


@dataclass(frozen=True)
class FactorSchedule:
    """Multiplier of one reward term over the training epochs.

    Goes from ``start`` at epoch 0 to ``end`` at epoch ``epochs`` and stays at ``end``
    afterwards: geometrically (the same ratio per epoch, e.g. 100 -> 10 -> 1 at
    epochs 0, epochs/2, epochs) or linearly.

    Attributes:
        start: Multiplier at epoch 0.
        epochs: Number of epochs to reach ``end``.
        end: Multiplier from epoch ``epochs`` on (default 1: the term's own weight).
        shape: "geometric" (default) or "linear".

    Example:
        >>> f = FactorSchedule(start=100.0, epochs=100)
        >>> f(0), round(f(50), 6), f(100), f(300)
        (100.0, 10.0, 1.0, 1.0)
    """

    start: float
    epochs: int
    end: float = 1.0
    shape: str = "geometric"

    def __post_init__(self) -> None:
        if self.shape not in FACTOR_SHAPES:
            raise ValueError(f"shape must be one of {FACTOR_SHAPES}, got {self.shape!r}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.shape == "geometric" and (self.start <= 0 or self.end <= 0):
            raise ValueError("a geometric schedule needs start > 0 and end > 0")
        if self.start < 0 or self.end < 0:
            raise ValueError("multipliers must be non-negative")

    def __call__(self, epoch: int) -> float:
        """Multiplier at a (0-based) epoch."""
        if epoch >= self.epochs:
            return float(self.end)
        t = max(epoch, 0) / self.epochs
        if self.shape == "geometric":
            return float(self.start * (self.end / self.start) ** t)
        return float(self.start + (self.end - self.start) * t)

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "epochs": self.epochs, "shape": self.shape}


class RewardWeightSchedule:
    """Per-epoch multipliers of reward terms, on top of their weights and calibration.

    The weight of a term in epoch ``e`` is ``reward_weight * calibration * factor(e)``.
    Terms without a schedule keep a factor of 1. Use it for a term that gives no training
    signal early on: e.g. a worst-served species term that is exactly 0 while more than
    ``worst_fraction`` of the species are unprotected, which an untrained policy (and a
    policy that has learned only the other terms) does. Starting such a term with a
    large weight and lowering it to 1 lets training find plans where it varies, before
    the final weights apply. With factors ending at 1, the objective at the end of
    training is the one set by the reward weights.

    Pass it to ``EvolStrategiesTrainer(..., reward_weight_schedule=...)``, which applies
    it at every epoch and sends the multipliers with every episode, including to worker
    processes.

    Args:
        factors: FactorSchedule per reward term, keyed by the term's ``name``.

    Example:
        >>> schedule = RewardWeightSchedule(
        ...     {"species_worst": FactorSchedule(start=100.0, epochs=100)}
        ... )
        >>> schedule.factors(50, ["protected_population", "species_worst"]).round(6)
        array([ 1., 10.])
    """

    def __init__(self, factors: dict[str, FactorSchedule]):
        if not factors:
            raise ValueError("RewardWeightSchedule needs at least one term")
        for name, f in factors.items():
            if not isinstance(f, FactorSchedule):
                raise TypeError(f"schedule of {name!r} must be a FactorSchedule, got {type(f)}")
        self._factors = dict(factors)

    def check_names(self, reward_names: Sequence[str]) -> None:
        """Raise ValueError if a scheduled term is not among the reward terms."""
        unknown = sorted(set(self._factors) - set(reward_names))
        if unknown:
            raise ValueError(
                f"Scheduled reward terms {unknown} not found in the reward terms "
                f"{list(reward_names)}"
            )

    def factors(self, epoch: int, reward_names: Sequence[str]) -> np.ndarray:
        """Multiplier of each reward term (in the order of reward_names) at an epoch."""
        self.check_names(reward_names)
        return np.array(
            [self._factors[n](epoch) if n in self._factors else 1.0 for n in reward_names]
        )

    @property
    def terms(self) -> list[str]:
        return list(self._factors)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Schedule settings by term name (e.g. to store with a run)."""
        return {name: f.to_dict() for name, f in self._factors.items()}
