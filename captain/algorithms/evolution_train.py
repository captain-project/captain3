"""Evolution strategies trainer for policy optimization.

This module implements Natural Evolution Strategies (NES) for training
conservation policies without gradients.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from captain.algorithms import scheduler as sched
from captain.algorithms import train_utils as train_utils

if TYPE_CHECKING:
    from captain.algorithms.episode import EpisodeRunner
    from captain.algorithms.scheduler import RewardWeightSchedule

logger = logging.getLogger(__name__)


ADVANTAGE_NORMALIZATIONS = (None, "zscore", "rank")


def normalize_advantages(rewards: np.ndarray, method: str | None,
                         running_reward: float | None) -> np.ndarray:
    """Advantages of the perturbations from their total rewards.

    Args:
        rewards: Total reward of each perturbation, shape (n,).
        method: None = reward minus the running (EMA) baseline, in reward units;
            "zscore" = (reward - batch mean) / batch std; "rank" = centred ranks in
            [-0.5, 0.5] (fitness shaping, robust to outliers). With "zscore" and
            "rank" the step size no longer depends on the scale of the rewards, only
            on the ranking (and, for "zscore", relative differences) of perturbations.
        running_reward: Baseline used when method is None (0 if None).

    Returns:
        Advantages, shape (n,).
    """
    rewards = np.asarray(rewards, dtype=float)
    if method is None:
        return rewards - (0.0 if running_reward is None else running_reward)
    if method == "zscore":
        std = rewards.std()
        return (rewards - rewards.mean()) / std if std > 1e-12 else np.zeros_like(rewards)
    if method == "rank":
        n = len(rewards)
        if n < 2:
            return np.zeros_like(rewards)
        ranks = np.empty(n)
        ranks[np.argsort(rewards, kind="stable")] = np.arange(n)
        return ranks / (n - 1) - 0.5
    raise ValueError(f"advantage normalization must be one of {ADVANTAGE_NORMALIZATIONS}, "
                     f"got {method!r}")


def compute_evolutionary_update(
    results: list[tuple[dict, float]],
    epoch_coeff: np.ndarray,
    noise: np.ndarray,
    alpha: float,
    sigma: float,
    running_reward: float | None,
    advantage_normalization: str | None = None,
) -> np.ndarray:
    """Compute evolution strategies weight update.

    Args:
        results: List of (info, reward) tuples from parallel evaluation.
        epoch_coeff: Current weight vector.
        noise: Noise perturbations used for this epoch.
        alpha: Learning rate.
        sigma: Noise standard deviation.
        running_reward: Baseline reward for advantage computation (used when
            advantage_normalization is None).
        advantage_normalization: None, "zscore" or "rank" (see normalize_advantages).

    Returns:
        Updated weight vector.
    """
    if running_reward is None:
        running_reward = 0

    if sigma == 0:
        return epoch_coeff

    # Extract rewards (handle both scalar and array rewards)
    rewards = np.array([np.sum(r[1]) for r in results])
    n = len(rewards)
    advantages = normalize_advantages(rewards, advantage_normalization, running_reward)

    # Compute advantage-weighted noise
    perturbed_advantage = [aa * nn for aa, nn in zip(advantages, noise, strict=True)]

    # Update weights
    new_coeff = epoch_coeff + (alpha / (n * sigma)) * np.sum(perturbed_advantage, 0)

    return new_coeff


# Global worker state
_runner: EpisodeRunner | None = None
_setup_barrier: Any = None  # set in pool workers by init_setup_barrier
SETUP_TIMEOUT = 600.0  # seconds a worker waits for the others to receive their runner


def setup_worker(runner: EpisodeRunner) -> None:
    """Initialize worker with episode runner.

    Called once per worker process to set up persistent state. In a pool, the worker
    then waits at a barrier until every worker has its runner: otherwise a worker that
    finishes early could take a second runner and leave another worker without one.

    Args:
        runner: EpisodeRunner instance for this worker.
    """
    global _runner
    torch.set_num_threads(1)  # Prevent thread oversubscription
    _runner = runner
    if _setup_barrier is not None:
        _setup_barrier.wait(timeout=SETUP_TIMEOUT)


def init_setup_barrier(barrier: Any) -> None:
    """Pool initializer: store the barrier that setup_worker waits on."""
    global _setup_barrier
    _setup_barrier = barrier


def set_reward_multipliers(runner: EpisodeRunner, multipliers: np.ndarray | None) -> None:
    """Set the reward multipliers (calibration x schedule) of a runner, if it has any."""
    if multipliers is not None and len(multipliers) == len(runner.rewards.names):
        runner.rewards.set_multipliers(multipliers)


def execute_task(
    task: np.ndarray | tuple[np.ndarray, np.ndarray | None],
) -> tuple[dict[str, Any], float]:
    """Execute one episode with given parameters.

    Args:
        task: Policy weights to evaluate, or a (weights, reward multipliers) tuple; the
            multipliers are set on the worker's rewards before the episode runs.

    Returns:
        Tuple of (info dict, total reward).
    """
    global _runner
    if _runner is None:
        raise RuntimeError("Worker not initialized. Call setup_worker first.")
    if isinstance(task, tuple):
        params, multipliers = task
        set_reward_multipliers(_runner, multipliers)
    else:
        params = task
    return _runner.run_episode(params)


class EvolStrategiesTrainer:
    """Natural Evolution Strategies trainer for policy optimization.

    Uses parallel episode evaluation and fitness-weighted averaging to
    optimize policy weights without computing gradients.

    Attributes:
        epoch_coeff: Current policy weight vector.
        running_reward: Exponential moving average of rewards.
        scheduler: Learning rate and noise scheduler.

    Example:
        >>> trainer = EvolStrategiesTrainer(runners, initial_weights)
        >>> for epoch in range(100):
        ...     reward, summary = trainer.train_epoch()
        ...     print(f"Epoch {epoch}: {reward:.4f}")
    """

    def __init__(
        self,
        list_of_env_params: list[EpisodeRunner],
        initial_coeffs: np.ndarray,
        scheduler: sched.LearningScheduler | None = None,
        epsilon_reward: float = 0.5,
        n_perturbations: int | None = None,
        seed: int | None = None,
        advantage_normalization: str | None = "zscore",
        reward_weight_schedule: RewardWeightSchedule | None = None,
    ):
        """Initialize trainer.

        Args:
            list_of_env_params: List of EpisodeRunner instances, one per worker.
                Pass a single-element list to use sequential mode (no pool).
            initial_coeffs: Initial policy weight vector.
            scheduler: Learning rate/noise scheduler (default: create new).
            epsilon_reward: EMA smoothing factor for baseline.
            n_perturbations: Number of ES noise samples per epoch. Defaults to
                ``len(list_of_env_params)`` for backward compatibility. When using
                sequential mode (single runner), this should be set explicitly.
            seed: Random seed for reproducibility.
            advantage_normalization: How perturbation rewards become advantages:
                "zscore" (default) = standardized within each epoch; "rank" = centred
                ranks; None = reward minus the running baseline, so the step size
                scales with the reward values (e.g. much larger after reward
                calibration, which can make early epochs unstable). With "zscore" /
                "rank" the step size depends on alpha and sigma only, while the
                relative weights (and calibration) of the reward terms still decide
                which perturbations are better.
            reward_weight_schedule: Optional per-epoch multipliers of reward terms
                (RewardWeightSchedule), applied on top of the reward weights and the
                calibration. The multipliers of each epoch are sent with every episode,
                so they also reach all worker processes.
        """
        if advantage_normalization not in ADVANTAGE_NORMALIZATIONS:
            raise ValueError(f"advantage_normalization must be one of "
                             f"{ADVANTAGE_NORMALIZATIONS}, got {advantage_normalization!r}")
        self.advantage_normalization = advantage_normalization
        self.epoch_coeff = initial_coeffs.astype(np.float32).copy()
        self.scheduler = scheduler or sched.LearningScheduler()
        self.running_reward: float | None = None
        self.epsilon_reward = epsilon_reward
        self.rg = np.random.default_rng(seed)

        if len(list_of_env_params) == 1:
            # Sequential mode — run episodes in a loop on the current process.
            # This avoids pickling runners, which is required for GPU tensors.
            self._use_pool = False
            self._runner = list_of_env_params[0]
            self.n = n_perturbations if n_perturbations is not None else 1
            logger.info(
                f"Initialized sequential trainer with {self.n} perturbations, "
                f"{len(self.epoch_coeff)} parameters"
            )
        else:
            # Pool mode — distribute runners across worker processes.
            self._use_pool = True
            self.n = (
                n_perturbations
                if n_perturbations is not None
                else len(list_of_env_params)
            )
            n_workers = min(len(list_of_env_params), mp.cpu_count())

            # Use forkserver to prevent OpenMP deadlock.
            # With the default "fork" start method, child processes inherit
            # the parent's OpenMP/MKL thread-pool state. Since PyTorch
            # typically initialises OpenMP with many threads before the pool
            # is created, the forked children deadlock on the first tensor
            # operation.  "forkserver" avoids this by spawning workers from
            # a clean server process that has not yet initialised OMP.
            if os.name == "posix":  # For Unix-based systems (macOS, Linux)
                ctx = mp.get_context("forkserver")
            else:  # For Windows
                ctx = mp.get_context("spawn")

            # One runner per worker: after its setup, each worker waits at the barrier
            # until all workers have one, so no worker can take a second runner and
            # leave another without
            self.pool = ctx.Pool(
                processes=n_workers,
                initializer=init_setup_barrier,
                initargs=(ctx.Barrier(n_workers),),
            )
            self.pool.map(setup_worker, list_of_env_params[:n_workers], chunksize=1)
            logger.info(
                f"Initialized pool trainer with {n_workers} workers, "
                f"{self.n} perturbations, {len(self.epoch_coeff)} parameters"
            )

        self.reward_names = list_of_env_params[0].rewards.names

        # Reward multipliers: calibration (calibrate_reward_scales) x schedule. Start from
        # the multipliers already set on the runners' rewards (ones by default).
        calib = getattr(list_of_env_params[0].rewards, "_reward_calibration", None)
        if calib is not None and len(calib) == len(self.reward_names):
            self._calibration = np.asarray(calib, dtype=float).copy()
        else:
            self._calibration = np.ones(len(self.reward_names))
        self.reward_weight_schedule = reward_weight_schedule
        if reward_weight_schedule is not None:
            reward_weight_schedule.check_names(self.reward_names)
            if advantage_normalization is None:
                logger.warning(
                    "reward_weight_schedule with advantage_normalization=None: the "
                    "running reward baseline mixes reward scales across epochs; "
                    "'zscore' or 'rank' is recommended"
                )
        self.epoch = 0  # epochs trained (sets the reward weight schedule)
        self.last_reward_multipliers = self.reward_multipliers(0)

    @staticmethod
    def _apply_calibration_to_worker(multipliers, verbose=False):
        """Helper to update the global _runner on each worker process."""
        global _runner
        if _runner is not None:
            _runner.rewards.set_multipliers(multipliers, verbose=verbose)

    def get_reward_calibrated_weights(
        self, n_probes: int = 20, target_std: float = 1.0, verbose: bool = False
    ) -> np.ndarray:
        """
        Heuristic Reward Scaling (or Calibration)
        Runs probe episodes to equalize the 'volume' of different reward types.
        """
        if verbose:
            logger.info("Calibrating reward scales with %d probes...", n_probes)

        # 1. Use a wider variety of noise scales to 'wake up' all reward types
        # Some probes are small tweaks, some are larger explorations
        probe_params = []
        for i in range(n_probes):
            scale = 0.05 if i < (n_probes // 2) else 0.25
            probe_params.append(
                self.epoch_coeff + np.random.randn(len(self.epoch_coeff)) * scale
            )
        if self._use_pool:
            results = self.pool.map(execute_task, probe_params)
        else:
            results = [self._runner.run_episode(p) for p in probe_params]
        all_component_totals = np.array([res[0]["rewards"] for res in results])

        # 2. Use Standard Deviation but with a floor to prevent explosion
        component_stds = np.std(all_component_totals, axis=0)

        # Identify components that never triggered (Std is 0)
        # We set their multiplier to 1.0 (no change) so they don't dominate
        # if they happen to trigger once later.
        valid_mask = component_stds > 1e-6

        multipliers = np.ones_like(component_stds)
        multipliers[valid_mask] = target_std / component_stds[valid_mask]

        # 3. Clip the multipliers
        # Prevent any single reward from being boosted by more than, say, 1000x
        # This prevents 'extinction' (if rare) from becoming 1,000,000x more
        # important than 'cost' just because of one random event.
        multipliers = np.clip(multipliers, 0.001, 1000.0)

        if verbose:
            logger.info("Calibration complete. Multipliers: %s", multipliers)
            logger.info("Probe component totals: %s", all_component_totals)
            logger.info("Means: %s", np.mean(all_component_totals, axis=0))
            logger.info("St dev: %s", np.std(all_component_totals, axis=0))

        return multipliers

    def reward_multipliers(self, epoch: int | None = None) -> np.ndarray:
        """Multiplier of each reward term at an epoch (default: the next one).

        Calibration multipliers times the reward weight schedule's factors.
        """
        epoch = self.epoch if epoch is None else epoch
        if self.reward_weight_schedule is None:
            return self._calibration.copy()
        return self._calibration * self.reward_weight_schedule.factors(epoch, self.reward_names)

    def calibrate_reward_scales(
        self, multipliers: np.ndarray, verbose: bool = False
    ) -> None:
        # Kept by the trainer and sent with every episode (train_epoch); the copies
        # pushed to the workers below only cover episodes run outside train_epoch
        self._calibration = np.asarray(multipliers, dtype=float).copy()
        # 4. Update the local worker rewards
        # We use starmap to push these multipliers to the persistent workers
        if self._use_pool:
            self.pool.starmap(
                self._apply_calibration_to_worker,
                [
                    (
                        multipliers,
                        verbose,
                    )
                    for _ in range(self.n)
                ],
            )
        else:
            # Sequential: update the runner owned by this process, then push
            # the same calibration to any persistent worker copies.
            self._runner.rewards.set_multipliers(multipliers, verbose=verbose)

            if verbose:
                logger.info("Directly calibrated %d runners in main process.", self.n)

            for _ in range(self.n):
                # Unpack the arguments manually from the tuple
                self._apply_calibration_to_worker(multipliers, verbose)

    def train_epoch(self) -> tuple[float, dict[str, Any]]:
        """Run one training epoch.

        Returns:
            Tuple of (average reward, episode summary).
        """
        n_params = len(self.epoch_coeff)

        # 1. Generate noise perturbations
        noise = self.rg.standard_normal((self.n, n_params)).astype(np.float32)

        # 2. Create perturbed parameter vectors
        params = [
            self.epoch_coeff + self.scheduler.sigma * noise[i] for i in range(self.n)
        ]

        # 3. Evaluate perturbations, with this epoch's reward multipliers (calibration x
        # schedule) set for every episode
        multipliers = self.reward_multipliers()
        self.last_reward_multipliers = multipliers
        if self._use_pool:
            results = self.pool.map(execute_task, [(p, multipliers) for p in params])
        else:
            set_reward_multipliers(self._runner, multipliers)
            results = [self._runner.run_episode(p) for p in params]

        # 4. Initialize baseline if first epoch
        if self.running_reward is None:
            self.running_reward = float(np.mean([np.sum(r[1]) for r in results]))

        # 5. Compute update
        new_coeff = compute_evolutionary_update(
            results=results,
            epoch_coeff=self.epoch_coeff,
            noise=noise,
            alpha=self.scheduler.alpha,
            sigma=self.scheduler.sigma,
            running_reward=self.running_reward,
            advantage_normalization=self.advantage_normalization,
        )
        self.epoch_coeff = new_coeff

        # 6. Update baseline (EMA)
        avg_reward = float(np.mean([np.sum(r[1]) for r in results]))
        self.running_reward = (
            self.epsilon_reward * avg_reward
            + (1 - self.epsilon_reward) * self.running_reward
        )

        # 7. Summarize
        summary = train_utils.summarize_episodes(results)

        # 8. Update scheduler
        self.scheduler.step(summary["jaccard_indx"])
        self.epoch += 1

        return avg_reward, summary

    def get_weights(self) -> np.ndarray:
        """Get current policy weights.

        Returns:
            Current weight vector.
        """
        return self.epoch_coeff.copy()

    def save_checkpoint(self, path: str | Path) -> None:
        """Save trainer state to a .npz file.

        Saves weights, running reward, scheduler state, and RNG state
        so training can be resumed exactly.

        Args:
            path: File path (will be saved as .npz).
        """
        rng_state = self.rg.bit_generator.state
        state = {
            "epoch_coeff": self.epoch_coeff,
            "running_reward": np.array(
                self.running_reward if self.running_reward is not None else np.nan
            ),
            "epsilon_reward": np.array(self.epsilon_reward),
            "epoch": np.array(self.epoch),
            "reward_calibration": self._calibration,
            # RNG state: large ints stored as strings in byte arrays
            "rng_state_state": np.void(str(rng_state["state"]["state"]).encode()),
            "rng_state_inc": np.void(str(rng_state["state"]["inc"]).encode()),
            "rng_has_uint32": np.array(rng_state["has_uint32"]),
            "rng_uinteger": np.array(rng_state["uinteger"]),
        }
        # Flatten scheduler state into the dict
        for k, v in self.scheduler.state_dict().items():
            state[f"sched_{k}"] = np.array(v)

        np.savez(path, **state)
        logger.info("Saved checkpoint to %s", path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load trainer state from a .npz checkpoint.

        Args:
            path: Path to .npz checkpoint file.
        """
        data = np.load(path, allow_pickle=False)
        self.epoch_coeff = data["epoch_coeff"].astype(np.float32)

        rr = float(data["running_reward"])
        self.running_reward = None if np.isnan(rr) else rr
        self.epsilon_reward = float(data["epsilon_reward"])
        # Older checkpoints have no epoch / calibration (schedule restarts at epoch 0)
        if "epoch" in data.files:
            self.epoch = int(data["epoch"])
        if "reward_calibration" in data.files:
            calib = data["reward_calibration"]
            if len(calib) == len(self.reward_names):
                self._calibration = calib.astype(float)

        # Restore RNG state
        bg_state = self.rg.bit_generator.state
        bg_state["state"]["state"] = int(bytes(data["rng_state_state"]).decode())
        bg_state["state"]["inc"] = int(bytes(data["rng_state_inc"]).decode())
        bg_state["has_uint32"] = int(data["rng_has_uint32"])
        bg_state["uinteger"] = int(data["rng_uinteger"])
        self.rg.bit_generator.state = bg_state

        # Restore scheduler
        sched_state = {
            k.removeprefix("sched_"): float(data[k])
            for k in data.files
            if k.startswith("sched_")
        }
        self.scheduler.load_state_dict(sched_state)

        logger.info(
            "Loaded checkpoint from %s (running_reward=%.4f, alpha=%.6f, sigma=%.6f)",
            path,
            self.running_reward if self.running_reward is not None else float("nan"),
            self.scheduler.alpha,
            self.scheduler.sigma,
        )

    def save_reward_calibration(self, multipliers, file_path, verbose=False):
        """Saves reward multipliers to a JSON file using reward names as keys."""
        # Get names from the first runner's reward list
        # Create name -> value mapping
        calib_dict = {
            name: float(val) for name, val in zip(self.reward_names, multipliers)
        }

        with open(file_path, "w") as f:
            json.dump(calib_dict, f, indent=4)

        if verbose:
            logger.info("Reward calibration saved to %s", file_path)

    def load_reward_calibration(self, file_path, verbose=False):
        """Loads multipliers from JSON and aligns them with current reward order."""
        with open(file_path, "r") as f:
            calib_dict = json.load(f)

        # Build the multiplier array based on current reward order
        # Use reward names to ensure alignment
        multipliers = []
        for name in self.reward_names:
            if name not in calib_dict:
                raise ValueError(f"Reward '{name}' not found in calibration file!")
            multipliers.append(calib_dict[name])

        if verbose:
            logger.info("Loaded calibration for: %s", list(calib_dict.keys()))
        self.calibrate_reward_scales(np.array(multipliers), verbose=verbose)

    def close(self) -> None:
        """Clean up worker pool (no-op in sequential mode)."""
        if self._use_pool and hasattr(self, "pool"):
            self.pool.close()
            self.pool.join()

    def __del__(self):
        """Ensure pool is closed on deletion."""
        self.close()
