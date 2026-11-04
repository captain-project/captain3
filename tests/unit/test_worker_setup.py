"""Pool mode of EvolStrategiesTrainer: every worker process gets exactly one runner."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from types import SimpleNamespace

import numpy as np
import pytest

from captain.algorithms import evolution_train
from captain.algorithms.evolution_train import EvolStrategiesTrainer

N_WORKERS = 3


class TaggedRunner:
    """Minimal stand-in for an EpisodeRunner: the trainer only reads .rewards.names."""

    def __init__(self, tag: int):
        self.tag = tag
        self.rewards = SimpleNamespace(names=["r"])


def worker_tag(_: int) -> tuple[int, int | None]:
    """Process id and tag of the runner set up in the worker (None if none)."""
    time.sleep(0.02)  # let every worker pick up tasks
    runner = evolution_train._runner
    return os.getpid(), None if runner is None else runner.tag


@pytest.mark.slow
@pytest.mark.skipif((os.cpu_count() or 1) < N_WORKERS, reason="needs one CPU per worker")
def test_each_pool_worker_gets_one_runner():
    # Mapping the setup over the runners used to let a worker that started early take
    # two runners and leave another without one. It happened when workers start with
    # captain already imported (the forkserver preloads the example script, which
    # imports it), so preload it here too, and repeat to catch it
    mp.get_context("forkserver").set_forkserver_preload(["captain", __name__])
    for _ in range(5):
        trainer = EvolStrategiesTrainer(
            [TaggedRunner(i) for i in range(N_WORKERS)], initial_coeffs=np.zeros(3)
        )
        try:
            results = trainer.pool.map(worker_tag, range(20 * N_WORKERS), chunksize=1)
        finally:
            trainer.close()

        tags_by_pid: dict[int, set[int | None]] = {}
        for pid, tag in results:
            tags_by_pid.setdefault(pid, set()).add(tag)
        assert all(None not in tags for tags in tags_by_pid.values())
        assert all(len(tags) == 1 for tags in tags_by_pid.values())
        worker_tags = [next(iter(tags)) for tags in tags_by_pid.values()]
        assert len(set(worker_tags)) == len(worker_tags)  # one runner per worker
