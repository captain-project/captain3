#!/usr/bin/env python
"""Example: advanced training setup with edge effects, connectivity and multiple objectives.

Uses the same toy data as train_policy.py and combines the newer CAPTAIN features:

1. Edge effect (BioEnv): disturbance spills over into protected cells at reserve
   borders, so isolated cells and thin strips support smaller populations than the
   core of a compact reserve.
2. Existing protected areas and connectivity: CalcRewardConnectivity rewards linking
   protected-area units (Equivalent Connected Area gain). The toy data has no layer
   of existing protected areas, so by default a few small stand-in reserves are seeded
   at species-richness hotspots; with real data, point EXISTING_PA_FILE to a raster
   of the existing protected areas.
3. Rewards (all on [0, 1] scales, so no reward calibration is needed):
   - CalcRewardProtectedPopulation: IUCN-weighted protected share of each species
   - CalcRewardWorstSpecies: protected share of the worst-served species
     ("leave no species behind"), with its weight annealed by a RewardWeightSchedule
   - CalcRewardConnectivity: linking the protected-area units
   - CalcRewardPersistentCost: share of the total cost that is protected (a penalty)
4. Extra policy features: "edge_exposure", "buffer_value", "pa_linking" and
   "worst_species_value", so the policy can see what the new terms respond to.
5. Output: per-term reward curves, a map of the plan, a KMZ file of the plan for
   Google Earth and a short summary (protected patches, connectivity, rewards).

Turn components off with the switches below (EDGE_FACTOR = 0, USE_CONNECTIVITY,
USE_WORST_SPECIES) to compare plans with and without them.

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
import rasterio  # noqa: E402
import torch  # noqa: E402
from scipy import ndimage  # noqa: E402

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

# --- Edge effect -------------------------------------------------------------
# Protected cells receive EDGE_FACTOR x their edge exposure as extra disturbance.
# EDGE_SOURCE: "unprotected" = exposure is the unprotected share of the neighbourhood
# (an isolated protected cell has exposure 1); "disturbance" = the mean disturbance of
# unprotected neighbours (edges facing undisturbed land cost nothing).
EDGE_FACTOR = 0.9  # 0 = no edge effect
EDGE_RADIUS = 1  # neighbourhood radius in cells (1 = the 8 surrounding cells)
EDGE_DECAY = None  # weight neighbours by exp(-distance / EDGE_DECAY); None = equal weights
EDGE_SOURCE = "unprotected"

# --- Existing protected areas and connectivity -------------------------------
# Raster of existing protected areas (> 0 = protected), relative to DATA_DIR. None =
# seed N_RESERVES stand-in reserves: (2 * RESERVE_RADIUS + 1)-cell square blocks around
# the most species-rich cells, at least RESERVE_MIN_SEPARATION cells apart, within the
# largest connected part of the study area.
EXISTING_PA_FILE = None
N_RESERVES = 6
RESERVE_RADIUS = 3
RESERVE_MIN_SEPARATION = 50
USE_CONNECTIVITY = True
CONN_RESISTANCE = "uniform"  # "uniform" (gap = cells to link) or "cost" (relative cost)
CONN_ALPHA = 40.0  # distance decay of the connectivity, in resistance units
CONN_AREA_EXPONENT = 0.5  # unit weight = area ** exponent (0 = all units count the same)
CONN_FEATURE_REFRESH = 5  # recompute the pa_linking feature every n decisions

# --- Rewards -----------------------------------------------------------------
# Final weights by reward term name (terms that are switched off are ignored)
REWARD_WEIGHTS = {
    "protected_population": 1.0,
    "species_worst": 1.0,
    "connectivity": 1.0,
    "cost": 1.0,
}
THREAT_WEIGHTS = [1, 2, 4, 8, 16]  # LC, NT, VU, EN, CR
USE_WORST_SPECIES = True
WORST_FRACTION = 0.1  # share of species averaged by the worst-served term
# The worst-served term is 0 while more than WORST_FRACTION of the species have no
# protection, so it gives no training signal at first. Its weight starts at
# WORST_WEIGHT_START x and is lowered geometrically to 1x over ANNEAL_EPOCHS epochs.
WORST_WEIGHT_START = 100.0

# --- Training ----------------------------------------------------------------
N_EPOCHS = 40
ANNEAL_EPOCHS = N_EPOCHS // 2
N_PERTURBATIONS = 6  # Number of parallel episode evaluations (sequential on GPU)
N_PARALLEL_WORKERS = 4  # Number of CPUs (if not CUDA)
N_TIME_STEPS = 50  # Time duration of each episode
TARGET_PROTECTED_CELLS_FRACTION = 0.10  # Fraction of valid cells to newly protect
CELLS_PER_STEP = 1000  # number of new cells protected at each time step

# Species parameters
AVG_CARRYING_CAPACITY = 100  # 'individuals' per cells (* empirical relative abundance)
DISPERSAL_RATE = 0.5
DISPERSAL_WINDOW = 3

# Output
LOG_FILE = "training_log.tsv"
MODEL_FILE = "trained_weights.npy"
PLOT_TRAIN_FREQ = 5  # plot intermediate protection results during training


def results_dir() -> Path:
    """Output folder named after the switches, so different setups don't overwrite."""
    name = f"advanced_edge{EDGE_FACTOR:g}"
    name += "_conn" if USE_CONNECTIVITY else ""
    name += "_worst" if USE_WORST_SPECIES else ""
    return DATA_DIR / f"{name}_p{TARGET_PROTECTED_CELLS_FRACTION}"


# =============================================================================
# Existing protected areas
# =============================================================================


def existing_protection(mask: np.ndarray, sdm: cn.SpatialData) -> np.ndarray:
    """Existing protected cells, shape (n_cells,), from EXISTING_PA_FILE or stand-ins."""
    rows, cols = (np.asarray(c) for c in sdm._coords)
    if EXISTING_PA_FILE is not None:
        pa_grid, _ = cn.data_loader.load_map(DATA_DIR / EXISTING_PA_FILE, nan_to_num=True)
        return np.asarray(pa_grid).squeeze()[rows, cols] > 0

    # Stand-in reserves at species-richness hotspots of the largest connected area
    richness = (sdm.data_min_threshold > 0).sum(dim=0).cpu().numpy()
    valid = np.zeros(sdm.shape[1:], dtype=bool)
    valid[rows, cols] = True
    parts, _ = ndimage.label(valid, structure=np.ones((3, 3)))
    main_part = parts[rows, cols] == np.argmax(np.bincount(parts[rows, cols])[1:]) + 1
    centres: list[tuple[int, int]] = []
    for i in np.argsort(-richness, kind="stable"):
        if not main_part[i]:
            continue
        r, c = rows[i], cols[i]
        if all(max(abs(r - r0), abs(c - c0)) >= RESERVE_MIN_SEPARATION for r0, c0 in centres):
            centres.append((r, c))
        if len(centres) == N_RESERVES:
            break
    protected = np.zeros(len(rows), dtype=bool)
    for r0, c0 in centres:
        protected |= (np.abs(rows - r0) <= RESERVE_RADIUS) & (np.abs(cols - c0) <= RESERVE_RADIUS)
    return protected


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
    costs = cn.load_spatial_data(
        file=DATA_DIR / COST_FILE,
        future_file=DATA_DIR / FUTURE_COST_FILE,
        mask=mask,
        lower_bound=0,
        upper_bound=1,
        n_time_steps=N_TIME_STEPS,
    )

    traits = cn.data_loader.load_trait_table(
        DATA_DIR / SPECIES_TRAIT_FILE, sdm.names, ref_column="species", fill_gaps=True
    )
    # Per-species habitat thresholds (use the same ones at inference)
    sdm.reset_threshold(traits["min_habitat_suitability"].to_numpy())

    # Existing protected areas: the protection matrix starts with them, and the budget
    # adds TARGET_PROTECTED_CELLS_FRACTION of the valid cells on top
    protected0 = existing_protection(mask, sdm)
    rows, cols = (np.asarray(c) for c in sdm._coords)
    protection_grid = np.zeros(sdm.shape[1:])
    protection_grid[rows, cols] = protected0
    protection = cn.SpatialData(data=protection_grid, mask=mask, lower_bound=0, upper_bound=1)
    target_protected_cells = int(protected0.sum()) + int(
        TARGET_PROTECTED_CELLS_FRACTION * np.nansum(mask)
    )

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
        edge_factor=EDGE_FACTOR,
        edge_radius=EDGE_RADIUS,
        edge_decay=EDGE_DECAY,
        edge_source=EDGE_SOURCE,
        device=DEVICE,
    )

    # Rewards and the policy features that let the policy see what they respond to
    feature_set = list(cn.FeatureExtractor.DEFAULT_FEATURE_SET)
    reward_terms = [cn.CalcRewardProtectedPopulation(threat_weights=THREAT_WEIGHTS, device=DEVICE)]

    cell_values = None
    if EDGE_FACTOR > 0:
        # "buffer_value" needs the value of each cell's full carrying capacity: IUCN-weighted
        # population per individual of each species (as in CalcRewardProtectedPopulation),
        # times its sensitivity (the share of carrying capacity an edge removes)
        suitability = env.sdms.data_min_threshold
        weights = torch.as_tensor(THREAT_WEIGHTS, dtype=suitability.dtype, device=env.device)
        per_capita = weights[env.ext_risk.init_status] / env.ext_risk.init_pop.clamp(min=1e-8)
        value = (per_capita[:, None] * env._species_k[:, None] * suitability).sum(dim=0)
        sens = env._species_sensitivity.mean(dim=1)
        full_value = ((per_capita * sens)[:, None] * env._species_k[:, None] * suitability).sum(0)
        cell_values = full_value / value.median().clamp(min=1e-12)
        feature_set += ["edge_exposure", "buffer_value"]

    connectivity = None
    if USE_CONNECTIVITY:
        grid = np.zeros(env.sdms.shape[1:], dtype=bool)
        grid[rows, cols] = protected0
        units, _ = ndimage.label(grid, structure=np.ones((3, 3)))  # one unit per reserve
        if CONN_RESISTANCE == "cost":
            cell_cost = env.costs.data[0].cpu().numpy()
            resistance = np.clip(cell_cost / cell_cost[~protected0].mean(), 1e-3, None)
        else:
            resistance = np.ones(len(rows))
        connectivity = cn.PAConnectivity(
            (rows, cols),
            units[rows, cols],
            resistance,
            initial_protection=protected0,
            alpha=CONN_ALPHA,
            area_exponent=CONN_AREA_EXPONENT,
            feature_refresh_every=CONN_FEATURE_REFRESH,
            device=DEVICE,
        )
        reward_terms.append(cn.CalcRewardConnectivity(connectivity))
        feature_set.append("pa_linking")

    if USE_WORST_SPECIES:
        reward_terms.append(cn.CalcRewardWorstSpecies(worst_fraction=WORST_FRACTION))
        feature_set.append("worst_species_value")

    reward_terms.append(cn.CalcRewardPersistentCost(rescaler=float(1.0 / costs.data.sum())))

    rewards = cn.Rewards(
        reward_obj_list=reward_terms,
        reward_weights=np.array([REWARD_WEIGHTS[term.name] for term in reward_terms]),
    )

    feature_extractor = cn.FeatureExtractor(
        env,
        feature_set=feature_set,
        time_rescale=N_TIME_STEPS / 2,
        cell_values=cell_values,
        connectivity=connectivity,
        worst_fraction=WORST_FRACTION,
        device=DEVICE,
    )

    model = cn.CellNN(input_dim=feature_extractor.n_features, hidden_dim=16)
    policy = cn.PolicyNetwork(model, seed=SEED, device=DEVICE)

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
# Evaluation of the trained policy
# =============================================================================


def summarize_plan(episode: cn.EpisodeRunner, weights: np.ndarray, out_dir: Path) -> None:
    """Run the trained policy once, map and export the plan and print a summary."""
    env = episode.env
    env.reset()
    protected0 = env.protected_cells_mask.cpu().numpy().astype(bool)  # existing protection
    episode.run_episode(weights)

    rows, cols = (np.asarray(c) for c in env.sdms._coords)
    protected = env.protected_cells_mask.cpu().numpy().astype(bool)

    # Plan grid: 1 = existing protection, 2 = new protection, NaN outside the study area
    plan = np.full(env.sdms.shape[1:], np.nan)
    plan[rows, cols] = np.where(protected, np.where(protected0, 1, 2), 0)
    np.save(out_dir / "protection_plan.npy", plan)
    cn.plots.plot_grid(
        plan,
        title="Existing (grey) and new (green) protection",
        outfile=out_dir / "protection_plan",
        cmap=["#52514e", "#1baf7a"],
        vmin=1,
        vmax=2,
        dpi=300,
        figsize=(6, 8),
    )

    # KMZ for Google Earth / QGIS, georeferenced like the species distribution maps
    with rasterio.open(sorted((DATA_DIR / PRESENT_SDMS_DIR).glob("*.tif"))[0]) as src:
        plan_transform, plan_crs = src.transform, src.crs
    cn.export.raster_to_kml(
        np.nan_to_num(plan).astype(int),
        plan_transform,
        plan_crs,
        out_dir / "protection_plan.kmz",
        [
            cn.export.KmlClass(1, "Existing protection", "#52514e", opacity=0.5),
            cn.export.KmlClass(2, "New protection", "#1baf7a"),
        ],
        name="CAPTAIN protection plan",
    )

    # Spatial structure of the plan: protected patches (8-neighbour connectivity)
    grid = np.zeros(env.sdms.shape[1:], dtype=bool)
    grid[rows, cols] = protected
    patches, n_patches = ndimage.label(grid, structure=np.ones((3, 3)))
    sizes = np.bincount(patches.ravel())[1:]

    print("\nTrained plan")
    print(f"  protected cells: {protected.sum()} ({protected0.sum()} existing)")
    print(f"  protected patches: {n_patches}, median size {np.median(sizes):.0f} cells, "
          f"{(sizes == 1).sum()} isolated cells")
    for term in episode.rewards._reward_obj_list:  # value of each term at the end
        print(f"  {term.name}: {float(term.calc_reward(env)):.3f}")
    print(f"Plan saved to: {out_dir / 'protection_plan.png'} and protection_plan.kmz")


# =============================================================================
# Main Training Loop
# =============================================================================


def main():
    print("=" * 60)
    print("CAPTAIN-3 Training: edge effect, connectivity, multiple objectives")
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
    print(f"  Grid: {episode.env.n_cells} cells, {episode.env.n_species} species")
    print(f"  Features: {episode.feature_extractor.feature_names}")
    print(f"  Reward terms: {episode.rewards.names}")

    schedule = None
    if USE_WORST_SPECIES:
        # Multipliers by reward term name, on top of REWARD_WEIGHTS
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
        results_dir=out_dir,
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
        terms = " | ".join(
            f"{name} {value:.3f}"
            for name, value in zip(
                episode.rewards.names, summary["avg_rewards_by_type"], strict=True
            )
        )
        print(f"           {terms}")

    print("-" * 60)
    print(f"Training complete in {time.time() - t_start:.1f}s")
    print(f"Log saved to: {logger.log_path}")
    print(f"Weights saved to: {logger.weights_path}")

    # Per-term rewards over training (the total follows the changing weights)
    for name in episode.rewards.names:
        cn.plots.plot_rl_rewards(
            logger.log_path,
            title=f"{name} over training",
            outfile=out_dir / f"reward_{name}_over_training.png",
            reward_col=f"r_{name}",
        )

    trainer.close()
    summarize_plan(episode, np.load(logger.weights_path), out_dir)


if __name__ == "__main__":
    main()
