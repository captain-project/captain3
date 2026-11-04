#!/usr/bin/env python
"""Example: static prioritization within a single time step.

When the landscape is treated as fixed (present habitat suitability, disturbance and
costs; no climate change or population dynamics), a protection plan can be optimized
as a one-shot prioritization: the whole budget is allocated within one time step, over
N_DECISIONS decisions, and the policy features are recomputed between decisions, so
later decisions can complement what is already protected.

EpisodeRunner(env_dynamics=False) skips dispersal and growth and keeps the inputs at
their initial values: rewards only reflect which cells are protected, and an episode
takes a fraction of a second, so training is fast.

Rewards (both on [0, 1] scales, so no reward calibration is needed):
- CalcRewardProtectedPopulation with a per-species target: once a species has
  PROTECTION_TARGET of its population protected, protecting more of it earns nothing,
  which pushes the plan to cover more species (complementarity)
- CalcRewardPersistentCost: share of the total cost that is protected (a penalty)

Requirements:
- Example data in DATA_DIR (see train_policy.py)
"""

import warnings

# Filter out the specific PyTorch Sparse CSR beta warning
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

import logging  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import captain as cn  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 123

# =============================================================================
# Configuration
# =============================================================================

# Data paths - the repo's own toy dataset, resolved relative to this script
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

PRESENT_SDMS_DIR = "present_sdms"
SPECIES_TRAIT_FILE = "species_tbl.csv"
DISTURBANCE_FILE = "env_layers/area_swept_disturbance.tif"
COST_FILE = "env_layers/cost.tif"
DATA_MASK = "env_layers/area_mask.npy"

# Prioritization
TARGET_PROTECTED_CELLS_FRACTION = 0.10  # Fraction of valid cells to protect
N_DECISIONS = 20  # decisions within the single time step (features updated in between)
THREAT_WEIGHTS = [1, 2, 4, 8, 16]  # LC, NT, VU, EN, CR
# Share of each species' population after which it counts as covered: protecting more of
# it earns nothing. Here equal to the share of cells protected, i.e. each species' fair share
PROTECTION_TARGET = 0.1
REWARD_WEIGHTS = np.array([1.0, 1.0])  # protected population, cost

# Training
N_EPOCHS = 40
N_PERTURBATIONS = 6  # Number of parallel episode evaluations (sequential on GPU)
N_PARALLEL_WORKERS = 4  # Number of CPUs (if not CUDA)

# Species parameters
AVG_CARRYING_CAPACITY = 100  # 'individuals' per cells (* empirical relative abundance)
DISPERSAL_RATE = 0.5
DISPERSAL_WINDOW = 3

# Output
LOG_FILE = "training_log.tsv"
MODEL_FILE = "trained_weights.npy"


def results_dir() -> Path:
    return DATA_DIR / f"static_p{TARGET_PROTECTED_CELLS_FRACTION}_t{PROTECTION_TARGET}"


# =============================================================================
# Episode Setup Function
# =============================================================================


def create_episode_runner() -> cn.EpisodeRunner:
    """Create a single-step episode runner without population dynamics."""
    mask, _ = cn.data_loader.load_map(DATA_DIR / DATA_MASK)

    # Present-day layers only: nothing changes during the (single-step) episode
    sdm = cn.load_spatial_data_from_dir(
        dir=DATA_DIR / PRESENT_SDMS_DIR, mask=mask, lower_bound=0, upper_bound=1
    )
    disturbance = cn.load_spatial_data(
        file=DATA_DIR / DISTURBANCE_FILE, mask=mask, lower_bound=0, upper_bound=1
    )
    costs = cn.load_spatial_data(file=DATA_DIR / COST_FILE, mask=mask, lower_bound=0, upper_bound=1)
    protection = cn.SpatialData(
        data=np.zeros(disturbance.shape), mask=mask, lower_bound=0, upper_bound=1
    )
    target_protected_cells = int(TARGET_PROTECTED_CELLS_FRACTION * np.nansum(mask))

    traits = cn.data_loader.load_trait_table(
        DATA_DIR / SPECIES_TRAIT_FILE, sdm.names, ref_column="species", fill_gaps=True
    )
    sdm.reset_threshold(traits["min_habitat_suitability"].to_numpy())

    sensitivity = traits["sensitivity_disturbance"].to_numpy(copy=True)[:, np.newaxis]
    growth_rates = traits["growth_rate"].to_numpy(copy=True) + 1.0
    carrying_capacity = AVG_CARRYING_CAPACITY / traits["conservation_status"].to_numpy(copy=True)
    conservation_status = traits["conservation_status"].to_numpy(copy=True) - 1
    ext_risk = cn.ExtinctionRisk(init_status=conservation_status, n_classes=5, alpha=0.5)

    disp_file = DATA_DIR / f"dispersal_d{DISPERSAL_RATE}_t{DISPERSAL_WINDOW}.npz"
    if not disp_file.exists():
        cn.grid_utils.save_dispersal_distances(
            lambda_0=DISPERSAL_RATE,
            coords=sdm._coords,
            threshold=DISPERSAL_WINDOW,
            filename=str(disp_file),
        )
    dispersal_matrix = cn.grid_utils.load_dispersal_distances(str(disp_file))

    env = cn.BioEnv(
        sdms=sdm,
        disturbance=disturbance,
        costs=costs,
        protection_matrix=protection,
        species_k=carrying_capacity,
        growth_rates=growth_rates,
        sensitivity_rates=sensitivity,
        cached_dispersal_matrix=dispersal_matrix,
        ext_risk=ext_risk,
        device=DEVICE,
    )

    feature_extractor = cn.FeatureExtractor(env, device=DEVICE)
    model = cn.CellNN(input_dim=feature_extractor.n_features, hidden_dim=16)
    policy = cn.PolicyNetwork(model, seed=SEED, device=DEVICE)

    rewards = cn.Rewards(
        reward_obj_list=[
            cn.CalcRewardProtectedPopulation(
                threat_weights=THREAT_WEIGHTS, target=PROTECTION_TARGET, device=DEVICE
            ),
            cn.CalcRewardPersistentCost(rescaler=float(1.0 / costs.data.sum())),
        ],
        reward_weights=REWARD_WEIGHTS,
    )

    # The whole budget within one time step, in N_DECISIONS decisions
    budget_manager = cn.GlobalBudgetManager(
        total_target=target_protected_cells,
        cells_per_time_step=target_protected_cells,
        feature_updates_per_time_step=N_DECISIONS,
    )

    return cn.EpisodeRunner(
        env=env,
        feature_extractor=feature_extractor,
        policy_network=policy,
        rewards=rewards,
        n_steps=1,
        budget_manager=budget_manager,
        env_dynamics=False,  # no dispersal/growth; inputs stay at their present values
    )


def rewards_cost(episode: cn.EpisodeRunner) -> float:
    """Current value of the cost term (minus the share of the total cost protected)."""
    cost_term = next(t for t in episode.rewards._reward_obj_list if t.name == "cost")
    return float(cost_term.calc_reward(episode.env))


# =============================================================================
# Main Training Loop
# =============================================================================


def main():
    print("=" * 60)
    print("CAPTAIN-3 Static prioritization (single time step)")
    print("=" * 60)
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data folder {DATA_DIR} not found (see train_policy.py)")
    out_dir = results_dir()
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nDevice: {DEVICE}")
    if DEVICE == "cuda":
        episode_runners = [create_episode_runner()]
    else:
        print(f"Creating {N_PARALLEL_WORKERS} parallel episode runners on CPU...")
        episode_runners = [create_episode_runner() for _ in range(N_PARALLEL_WORKERS)]
    episode = episode_runners[0]

    trainer = cn.EvolStrategiesTrainer(
        episode_runners,
        initial_coeffs=episode.policy.get_flat_weights(),
        scheduler=cn.LearningScheduler(initial_alpha=0.05, initial_sigma=0.3),
        epsilon_reward=0.75,
        n_perturbations=N_PERTURBATIONS,
        seed=SEED,
    )
    logger = cn.algorithms.TrainingLogger(
        trainer=trainer,
        episode=episode,
        results_dir=out_dir,
        log_file=LOG_FILE,
        weights_file=MODEL_FILE,
        plot_freq=N_EPOCHS,  # only the first epoch's map; the final plan is mapped below
    )

    print(f"\nTraining for {N_EPOCHS} epochs...")
    print("-" * 60)
    t_start = time.time()
    for epoch in range(N_EPOCHS):
        t0 = time.time()
        avg_reward, summary = trainer.train_epoch()
        logger.log_epoch(epoch, avg_reward, summary, time.time() - t0)
    print("-" * 60)
    print(f"Training complete in {time.time() - t_start:.1f}s")
    trainer.close()

    cn.plots.plot_rl_rewards(
        logger.log_path,
        title="Reward over training (static prioritization)",
        outfile=out_dir / "reward_over_training.png",
        reward_col="avg_reward",
    )

    # Plan of the trained policy
    env = episode.env
    episode.run_episode(np.load(logger.weights_path))
    cn.plots.plot_grid(
        env.protection_matrix.reconstruct_grid[0],
        title="Static prioritization",
        outfile=out_dir / "protection_plan",
        dpi=300,
        figsize=(6, 8),
    )

    # Protected share of each species' population
    protected = env.protected_cells_mask.to(env.h.dtype)
    protected_share = (env.h * protected).sum(dim=1) / env.h.sum(dim=1).clamp(min=1e-12)
    n_covered = int((protected_share >= PROTECTION_TARGET).sum())
    print(f"\nProtected cells: {int(protected.sum())}")
    print(f"Protected share of each species' population: min {protected_share.min():.1%}, "
          f"median {protected_share.median():.1%}, max {protected_share.max():.1%}")
    print(f"Species with >= {PROTECTION_TARGET:.0%} of their population protected: "
          f"{n_covered} of {env.n_species}")
    print(f"Share of the total cost: {-rewards_cost(episode):.1%}")
    print(f"Plan saved to: {out_dir / 'protection_plan.png'}")


if __name__ == "__main__":
    main()
