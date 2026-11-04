#!/usr/bin/env python
"""Example: Train a policy with a reward weight schedule (annealing).

Same data and environment as train_policy.py, with two reward terms:

- CalcRewardProtectedPopulation: IUCN-weighted protected share of each species
- CalcRewardWorstSpecies: mean protected share of the 10% worst-served species
  ("leave no species behind", a soft maximin)

The worst-served term is exactly 0 while more than 10% of the species have no
protection, which is what an untrained policy, or one that has learned only the first
term, does. It then gives no training signal: with fixed weights, training can settle
on plans that ignore it. A RewardWeightSchedule starts the term with a large weight
and lowers it to 1, so that the policy first learns to spread protection and then
moves to the balance set by REWARD_WEIGHTS. Both terms are fractions in [0, 1], so no
reward calibration is needed; with a calibration (see train_policy.py), the schedule
multiplies on top of it.

During the schedule the total reward in the training log follows the changing
weights, so compare the per-term columns (r_protected_population, r_species_worst)
across epochs; mult_<term> lists the multipliers used in each epoch.

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
FUTURE_SDMS_DIR = "future_sdms"
SPECIES_TRAIT_FILE = "species_tbl.csv"
DISTURBANCE_FILE = "env_layers/area_swept_disturbance.tif"
FUTURE_DISTURBANCE_FILE = "env_layers/future_area_swept_disturbance.tif"
COST_FILE = "env_layers/cost.tif"
FUTURE_COST_FILE = "env_layers/future_cost.tif"
DATA_MASK = "env_layers/area_mask.npy"

# Rewards: weights of the final objective (protected population, worst-served species)
REWARD_WEIGHTS = np.array([1.0, 1.0])
THREAT_WEIGHTS = [1, 2, 4, 8, 16]  # LC, NT, VU, EN, CR
WORST_FRACTION = 0.1  # share of species averaged by the worst-served term

# Training and policy parameters
N_EPOCHS = 40
N_PERTURBATIONS = 6  # Number of parallel episode evaluations (sequential on GPU)
N_PARALLEL_WORKERS = 4  # Number of CPUs (if not CUDA)
N_TIME_STEPS = 50  # Time duration of each episode
TARGET_PROTECTED_CELLS_FRACTION = 0.10  # Fraction of valid cells to be protected
CELLS_PER_STEP = 1000  # number of new cells protected at each time step

# Reward weight schedule: the worst-served term starts at 100x its weight and is lowered
# geometrically to 1x (its REWARD_WEIGHTS value) over the first half of the training.
# With a geometric schedule the multiplier is 10x halfway through.
WORST_WEIGHT_START = 100.0
ANNEAL_EPOCHS = N_EPOCHS // 2

# Species parameters
AVG_CARRYING_CAPACITY = 100  # 'individuals' per cells (* empirical relative abundance)
DISPERSAL_RATE = 0.5
DISPERSAL_WINDOW = 3

# Output
RESULTS_DIR = DATA_DIR / f"global_worst_schedule_x{WORST_WEIGHT_START:g}_e{ANNEAL_EPOCHS}"
LOG_FILE = "global_log.tsv"
MODEL_FILE = "trained_weights.npy"
PLOT_TRAIN_FREQ = 5  # plot intermediate protection results during training

os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# Episode Setup Function
# =============================================================================


def create_episode_runner() -> cn.EpisodeRunner:
    """Create an episode runner with real data (one per parallel worker)."""
    mask, _ = cn.data_loader.load_map(DATA_DIR / DATA_MASK)

    sdm = cn.load_spatial_data_from_dir(
        dir=DATA_DIR / PRESENT_SDMS_DIR,
        future_dir=DATA_DIR / FUTURE_SDMS_DIR,
        mask=mask,
        lower_bound=0,
        upper_bound=1,
        n_time_steps=N_TIME_STEPS,
    )
    disturbance = cn.load_spatial_data(
        file=DATA_DIR / DISTURBANCE_FILE,
        future_file=DATA_DIR / FUTURE_DISTURBANCE_FILE,
        mask=mask,
        lower_bound=0,
        upper_bound=1,
        n_time_steps=N_TIME_STEPS,
    )
    protection = cn.SpatialData(
        data=np.zeros(disturbance.shape), mask=mask, lower_bound=0, upper_bound=1
    )
    costs = cn.load_spatial_data(
        file=DATA_DIR / COST_FILE,
        future_file=DATA_DIR / FUTURE_COST_FILE,
        mask=mask,
        lower_bound=0,
        upper_bound=1,
        n_time_steps=N_TIME_STEPS,
    )
    target_protected_cells = int(TARGET_PROTECTED_CELLS_FRACTION * np.nansum(mask))

    traits = cn.data_loader.load_trait_table(
        DATA_DIR / SPECIES_TRAIT_FILE, sdm.names, ref_column="species", fill_gaps=True
    )
    # Per-species habitat thresholds (use the same ones at inference)
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
            coords=sdm.coords,
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

    # Default features plus the cells holding the currently worst-served species
    feature_extractor = cn.FeatureExtractor(
        env,
        feature_set=[*cn.FeatureExtractor.DEFAULT_FEATURE_SET, "worst_species_value"],
        time_rescale=N_TIME_STEPS / 2,
        worst_fraction=WORST_FRACTION,
        device=DEVICE,
    )

    model = cn.CellNN(input_dim=feature_extractor.n_features, hidden_dim=16)
    policy = cn.PolicyNetwork(model, seed=SEED, device=DEVICE)

    rewards = cn.Rewards(
        reward_obj_list=[
            cn.CalcRewardProtectedPopulation(threat_weights=THREAT_WEIGHTS, device=DEVICE),
            cn.CalcRewardWorstSpecies(worst_fraction=WORST_FRACTION),
        ],
        reward_weights=REWARD_WEIGHTS,
    )

    budget_manager = cn.GlobalBudgetManager(
        total_target=target_protected_cells,
        cells_per_time_step=CELLS_PER_STEP,
        feature_updates_per_time_step=1,
    )

    return cn.EpisodeRunner(
        env=env,
        feature_extractor=feature_extractor,
        policy_network=policy,
        rewards=rewards,
        n_steps=N_TIME_STEPS,
        budget_manager=budget_manager,
    )


# =============================================================================
# Main Training Loop
# =============================================================================


def main():
    print("=" * 60)
    print("CAPTAIN-3 Training with a reward weight schedule")
    print("=" * 60)
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data folder {DATA_DIR} not found (see train_policy.py)")

    print(f"\nDevice: {DEVICE}")
    if DEVICE == "cuda":
        episode_runners = [create_episode_runner()]
    else:
        print(f"Creating {N_PARALLEL_WORKERS} parallel episode runners on CPU...")
        episode_runners = [create_episode_runner() for _ in range(N_PARALLEL_WORKERS)]

    episode = episode_runners[0]
    print(f"  Grid: {episode.env.n_cells} cells, {episode.env.n_species} species")
    print(f"  Features: {episode.feature_extractor.n_features}")
    print(f"  Reward terms: {episode.rewards.names}")

    # Multipliers by reward term name: CalcRewardWorstSpecies is named "species_worst"
    schedule = cn.RewardWeightSchedule(
        {"species_worst": cn.FactorSchedule(start=WORST_WEIGHT_START, epochs=ANNEAL_EPOCHS)}
    )
    print(f"  Schedule: {schedule.to_dict()}")

    trainer = cn.EvolStrategiesTrainer(
        episode_runners,
        initial_coeffs=episode.policy.get_flat_weights(),
        scheduler=cn.LearningScheduler(initial_alpha=0.05, initial_sigma=0.3),
        epsilon_reward=0.75,
        n_perturbations=N_PERTURBATIONS,
        seed=SEED,
        reward_weight_schedule=schedule,  # default advantage_normalization="zscore"
    )

    logger = cn.algorithms.TrainingLogger(
        trainer=trainer,
        episode=episode,
        results_dir=RESULTS_DIR,
        log_file=LOG_FILE,
        weights_file=MODEL_FILE,
        plot_freq=PLOT_TRAIN_FREQ,
    )

    print(f"\nTraining for {N_EPOCHS} epochs...")
    print("-" * 60)
    t_start = time.time()
    for epoch in range(N_EPOCHS):
        t0 = time.time()
        avg_reward, summary = trainer.train_epoch()
        logger.log_epoch(epoch, avg_reward, summary, time.time() - t0)
        r_pop, r_worst = summary["avg_rewards_by_type"]
        print(
            f"           protected population {r_pop:.3f} | worst-served {r_worst:.3f} | "
            f"worst-served weight x{trainer.last_reward_multipliers[1]:.2f}"
        )

    print("-" * 60)
    print(f"Training complete in {time.time() - t_start:.1f}s")
    print(f"Log saved to: {logger.log_path}")
    print(f"Weights saved to: {logger.weights_path}")

    # Per-term rewards over training (the total follows the changing weights)
    for name in episode.rewards.names:
        cn.plots.plot_rl_rewards(
            logger.log_path,
            title=f"{name} over training",
            outfile=RESULTS_DIR / f"reward_{name}_over_training.png",
            reward_col=f"r_{name}",
        )
    print(f"Reward plots saved to: {RESULTS_DIR}")

    trainer.close()


if __name__ == "__main__":
    main()
