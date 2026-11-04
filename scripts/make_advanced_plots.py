#!/usr/bin/env python
"""Regenerate the tutorials/advanced_example figures and table from trained runs.

Does NOT retrain. Expects the output folders of:
    uv run python examples/train_policy_advanced.py
    (the same with EDGE_FACTOR = 0 and USE_CONNECTIVITY = False)
    (the same with EDGE_FACTOR = 0, USE_CONNECTIVITY = False and USE_WORST_SPECIES = False)
    uv run python examples/static_prioritization.py

and writes to tutorials/advanced_example/images/:
- plans_comparison.png       the three plans side by side
- rewards_by_term.png        per-term rewards over training (full setup)
- static_plan.png            the static prioritization plan
and prints a markdown table comparing the three plans, all evaluated in the same
environment (the full setup: edge effect on, same existing reserves and connectivity).

Usage:
    uv run python scripts/make_advanced_plots.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import ndimage  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
IMAGES = REPO / "tutorials" / "advanced_example" / "images"
sys.path.insert(0, str(REPO / "examples"))

import static_prioritization as sp  # noqa: E402
import train_policy_advanced as adv  # noqa: E402

RUNS = [
    ("advanced_edge0.9_conn_worst_p0.1", "A. Edge effect, connectivity,\nworst-served species"),
    ("advanced_edge0_worst_p0.1", "B. No edge effect,\nno connectivity"),
    ("advanced_edge0_p0.1", "C. Protected population\nand cost only"),
]
COLORS = {"valid": "#e6e5e1", "existing": "#52514e", "new": "#1baf7a"}
INK = "#2f2e2b"


def plan_image(plan: np.ndarray) -> np.ma.MaskedArray:
    """Masked plan grid (0 valid, 1 existing, 2 new) cropped to the study area."""
    valid_rows = np.where(~np.isnan(plan).all(axis=1))[0]
    valid_cols = np.where(~np.isnan(plan).all(axis=0))[0]
    crop = plan[valid_rows.min():valid_rows.max() + 1, valid_cols.min():valid_cols.max() + 1]
    return np.ma.masked_invalid(crop)


def draw_plan(ax, plan: np.ndarray, title: str) -> None:
    cmap = ListedColormap([COLORS["valid"], COLORS["existing"], COLORS["new"]])
    cmap.set_bad("white")
    ax.imshow(plan_image(plan), cmap=cmap, vmin=-0.5, vmax=2.5, interpolation="nearest")
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def legend_handles(existing: bool = True) -> list[Patch]:
    handles = [Patch(color=COLORS["new"], label="New protection")]
    if existing:
        handles.insert(0, Patch(color=COLORS["existing"], label="Existing reserves"))
    handles.append(Patch(color=COLORS["valid"], label="Unprotected"))
    return handles


def evaluate(plans: dict[str, np.ndarray]) -> pd.DataFrame:
    """Evaluate each final plan in the full environment, with and without edge effect.

    The plan is protected from the first step and simulated for N_TIME_STEPS steps
    (climate change included), so the plans are compared under the same conditions.
    """
    adv.EDGE_FACTOR, adv.USE_CONNECTIVITY, adv.USE_WORST_SPECIES = 0.9, True, True
    with_edge = adv.create_episode_runner()
    adv.EDGE_FACTOR = 0.0
    no_edge = adv.create_episode_runner()
    adv.EDGE_FACTOR = 0.9

    rows, cols = (np.asarray(c) for c in with_edge.env.sdms.coords)
    results = []
    for run, plan in plans.items():
        values = plan[rows, cols]
        new_cells = np.where(values == 2)[0]
        out = {"run": run}
        for label, episode in (("edge", with_edge), ("no_edge", no_edge)):
            env = episode.env
            env.reset()
            env.update_protection_matrix(new_cells)
            terms = {t.name: t for t in episode.rewards.terms}
            cost_share = []  # share of the total cost protected, at each step
            for _ in range(adv.N_TIME_STEPS):
                env.step()
                cost_share.append(-terms["cost"].calc_reward(env))
            out["cost"] = float(np.mean(cost_share))
            out[f"protected_population_{label}"] = terms["protected_population"].calc_reward(env)
            out[f"species_worst_{label}"] = terms["species_worst"].calc_reward(env)
            out["connectivity"] = terms["connectivity"].calc_reward(env)
        grid = ~np.isnan(plan) & (np.nan_to_num(plan) > 0)
        patches, n_patches = ndimage.label(grid, structure=np.ones((3, 3)))
        sizes = np.bincount(patches.ravel())[1:]
        out.update(n_patches=n_patches, isolated=int((sizes == 1).sum()),
                   protected=int(grid.sum()))
        results.append(out)
    return pd.DataFrame(results)


def print_table(df: pd.DataFrame) -> None:
    print("| Run | Protected patches | Isolated cells | Connectivity (ECA gain) "
          "| Protected population | Loss to edge effect | Worst-served species | Cost |")
    print("|---|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        loss = 1 - r.protected_population_edge / r.protected_population_no_edge
        print(f"| {r.run} | {r.n_patches} | {r.isolated} | {r.connectivity:.2f} "
              f"| {r.protected_population_edge:.3f} | {loss:.0%} "
              f"| {r.species_worst_edge:.3f} | {r.cost:.3f} |")


def plot_rewards(log_file: Path, anneal_epochs: int, outfile: Path) -> None:
    log = pd.read_csv(log_file, sep="\t")
    terms = [c[2:] for c in log.columns if c.startswith("r_")]
    fig, axes = plt.subplots(1, len(terms), figsize=(3.2 * len(terms), 3), sharex=True)
    for ax, term in zip(axes, terms, strict=True):
        y = log[f"r_{term}"]
        ax.axvspan(-0.5, anneal_epochs - 0.5, color="#f1efe8", lw=0)
        ax.scatter(log["epoch"], y, s=10, color="#9fb7cf", lw=0)
        ax.plot(log["epoch"], y.ewm(alpha=0.25).mean(), color="#c8102e", lw=2)
        ax.set_title(term, fontsize=10, color=INK)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.tick_params(labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Reward (sum over the episode)", fontsize=9)
    fig.text(0.01, 0.01, f"Shaded: epochs 0-{anneal_epochs - 1}, while the weight of "
             "species_worst is annealed from 100x to 1x", fontsize=8, color="#6b6a66")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    print("saved", outfile)


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    data_dir = adv.DATA_DIR

    plans = {run: np.load(data_dir / run / "protection_plan.npy") for run, _ in RUNS}
    fig, axes = plt.subplots(1, len(RUNS), figsize=(4 * len(RUNS), 5.6))
    for ax, (run, title) in zip(axes, RUNS, strict=True):
        draw_plan(ax, plans[run], title)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(IMAGES / "plans_comparison.png", dpi=200)
    plt.close(fig)
    print("saved", IMAGES / "plans_comparison.png")

    plot_rewards(data_dir / RUNS[0][0] / adv.LOG_FILE, adv.ANNEAL_EPOCHS,
                 IMAGES / "rewards_by_term.png")

    static_dir = sp.results_dir()
    if (static_dir / sp.MODEL_FILE).exists():
        episode = sp.create_episode_runner()
        episode.run_episode(np.load(static_dir / sp.MODEL_FILE))
        static_plan = episode.env.protection_matrix.reconstruct_grid[0]
        static_plan = np.where(np.isnan(static_plan), np.nan, 2 * (static_plan > 0))
        fig, ax = plt.subplots(figsize=(4.4, 5.6))
        draw_plan(ax, static_plan, "Static prioritization")
        fig.legend(handles=legend_handles(existing=False), loc="lower center", ncol=2,
                   frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(IMAGES / "static_plan.png", dpi=200)
        plt.close(fig)
        print("saved", IMAGES / "static_plan.png")

    print()
    print_table(evaluate(plans))


if __name__ == "__main__":
    main()
