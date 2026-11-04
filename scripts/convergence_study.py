#!/usr/bin/env python
"""How long to train: reward curves over 100 epochs for several seeds.

Trains the toy example's global agent (examples/train_policy.py) and the full advanced
setup (examples/train_policy_advanced.py) with N_EPOCHS epochs for each seed in SEEDS,
then plots each reward term over the epochs (mean and range across seeds) to
docs/images/convergence_<setup>.png and prints a table of the mean rewards in early,
middle and late epochs.

Each run takes about 15 minutes on 4 CPU cores (about 1.5 hours in total). Finished runs
(in data/convergence/) are skipped when the script is run again.

Usage:
    uv run python scripts/convergence_study.py          # train missing runs, then plot
    uv run python scripts/convergence_study.py --plot   # plot only
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples"))

import train_policy as tp  # noqa: E402
import train_policy_advanced as adv  # noqa: E402

SEEDS = [1, 2, 3]
N_EPOCHS = 100
LOG_FILE = "training_log.tsv"
OUT_DIR = tp.DATA_DIR / "convergence"
IMAGES = REPO / "docs" / "images"
WINDOWS = [(0, 10), (40, 50), (90, 100)]  # epoch ranges summarized in the table


def run_dir(setup: str, seed: int) -> Path:
    return OUT_DIR / f"{setup}_seed{seed}"


def finished(setup: str, seed: int) -> bool:
    log = run_dir(setup, seed) / LOG_FILE
    return log.exists() and len(pd.read_csv(log, sep="\t")) >= N_EPOCHS


def train_global(seed: int) -> None:
    out = run_dir("global", seed)
    out.mkdir(parents=True, exist_ok=True)
    tp.SEED, tp.N_EPOCHS = seed, N_EPOCHS
    tp.RESULTS_DIR, tp.LOG_FILE, tp.PLOT_TRAIN_FREQ = out, LOG_FILE, N_EPOCHS
    tp.main()


def train_advanced(seed: int) -> None:
    out = run_dir("advanced", seed)
    adv.SEED, adv.N_EPOCHS, adv.ANNEAL_EPOCHS = seed, N_EPOCHS, N_EPOCHS // 2
    adv.PLOT_TRAIN_FREQ = N_EPOCHS
    adv.results_dir = lambda: out
    adv.main()


def load_logs(setup: str) -> pd.DataFrame:
    logs = []
    for seed in SEEDS:
        log = pd.read_csv(run_dir(setup, seed) / LOG_FILE, sep="\t")
        logs.append(log.assign(seed=seed))
    return pd.concat(logs)


def plot(setup: str, title: str, anneal_epochs: int | None = None) -> None:
    logs = load_logs(setup)
    terms = [c for c in logs.columns if c.startswith("r_")]
    fig, axes = plt.subplots(1, len(terms), figsize=(3.4 * len(terms), 3), sharex=True)
    axes = [axes] if len(terms) == 1 else axes
    for ax, term in zip(axes, terms, strict=True):
        # running mean within each seed, then mean and range across seeds
        smooth = logs.groupby("seed")[term].transform(lambda y: y.ewm(alpha=0.2).mean())
        stats = smooth.groupby(logs["epoch"]).agg(["mean", "min", "max"])
        if anneal_epochs:
            ax.axvspan(-0.5, anneal_epochs - 0.5, color="#f1efe8", lw=0)
        ax.fill_between(stats.index, stats["min"], stats["max"], color="#9fb7cf", alpha=0.5, lw=0)
        ax.plot(stats.index, stats["mean"], color="#c8102e", lw=2)
        ax.set_title(term[2:], fontsize=10)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.tick_params(labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Reward (sum over the episode)", fontsize=9)
    note = f"{title}: mean (line) and range (band) of {len(SEEDS)} seeds, running average"
    if anneal_epochs:
        note += f"; shaded: species_worst weight annealed over epochs 0-{anneal_epochs - 1}"
    fig.text(0.01, 0.01, note, fontsize=8, color="#6b6a66")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    outfile = IMAGES / f"convergence_{setup}.png"
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    print("saved", outfile)

    print(f"\n{title} (mean across seeds of the per-epoch rewards)")
    header = " | ".join(f"epochs {a}-{b - 1}" for a, b in WINDOWS)
    print(f"| Term | {header} |")
    print("|---|" + "---|" * len(WINDOWS))
    for term in terms:
        means = [logs[(logs.epoch >= a) & (logs.epoch < b)][term].mean() for a, b in WINDOWS]
        print(f"| {term[2:]} | " + " | ".join(f"{m:.2f}" for m in means) + " |")


def main() -> None:
    if "--plot" not in sys.argv:
        for seed in SEEDS:
            if not finished("global", seed):
                train_global(seed)
            if not finished("advanced", seed):
                train_advanced(seed)
    IMAGES.mkdir(parents=True, exist_ok=True)
    plot("global", "Toy example, global agent")
    plot("advanced", "Advanced example", anneal_epochs=N_EPOCHS // 2)


if __name__ == "__main__":
    main()
