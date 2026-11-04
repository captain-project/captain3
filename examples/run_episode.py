#!/usr/bin/env python
"""Example: Run a single episode with synthetic data.

This script demonstrates how to:
1. Create synthetic spatial data (SDMs, costs, disturbance)
2. Set up the BioEnv environment
3. Create a policy network and feature extractor
4. Run a single episode and inspect results

For real data, see train_policy.py which loads GeoTIFF files.
"""

import time

import numpy as np

import captain as cn

# Reproducibility
rng = np.random.default_rng(42)

# =============================================================================
# Configuration
# =============================================================================

N_SPECIES = 20
GRID_SIZE = (60, 60)  # (rows, cols)
N_STEPS = 16
TARGET_PROTECTED_CELLS = 500
CELLS_PER_STEP = 50

# =============================================================================
# Create Synthetic Data
# =============================================================================

print("Creating synthetic data...")

# Species distribution models (habitat suitability 0-1)
sdm_data = rng.random(size=(N_SPECIES, *GRID_SIZE))

# Mask: some cells are invalid (e.g., ocean)
mask = np.ones(GRID_SIZE)
mask[:3, :2] = np.nan  # Example: top-left corner is invalid

# Climate change: SDMs shift over time
delta_sdm = rng.normal(scale=0.01, size=(N_SPECIES, *GRID_SIZE))

sdm = cn.SpatialData(
    data=sdm_data,
    mask=mask,
    delta_per_step=delta_sdm,
    lower_bound=0,
    upper_bound=1,
)

# Disturbance layer (e.g., human pressure)
disturbance = cn.SpatialData(
    data=rng.random(size=GRID_SIZE),
    mask=mask,
    lower_bound=0,
    upper_bound=1,
)

# Protection matrix (starts empty)
protection = cn.SpatialData(
    data=np.zeros(GRID_SIZE),
    mask=mask,
    lower_bound=0,
    upper_bound=1,
)

# Cost of protection per cell
costs = cn.SpatialData(
    data=rng.random(size=GRID_SIZE) * 0.5 + 0.1,
    mask=mask,
    lower_bound=0,
    upper_bound=1,
)

# Species traits
carrying_capacity = rng.uniform(50, 150, size=N_SPECIES)
growth_rates = rng.uniform(1.0, 1.5, size=N_SPECIES)
dispersal_rate = 0.1  # Scalar for efficiency
sensitivity = rng.random(size=(N_SPECIES, 1))  # Sensitivity to disturbance

# Initial extinction risk (IUCN categories 0-4: LC, NT, VU, EN, CR)
init_status = rng.integers(low=0, high=3, size=N_SPECIES)
ext_risk = cn.ExtinctionRisk(init_status=init_status, n_classes=5, alpha=0.5)

# =============================================================================
# Create Environment
# =============================================================================

print("Creating environment...")

env = cn.BioEnv(
    sdms=sdm,
    disturbance=disturbance,
    costs=costs,
    protection_matrix=protection,
    species_k=carrying_capacity,
    growth_rates=growth_rates,
    dispersal_rates=dispersal_rate,
    sensitivity_rates=sensitivity,
    ext_risk=ext_risk,
)

print(f"  Grid: {env.n_cells} cells, {env.n_species} species")

# =============================================================================
# Create Agent Components
# =============================================================================

print("Setting up agent...")

# Feature extractor: converts env state to neural network input
feature_extractor = cn.FeatureExtractor(env)
print(f"  Features: {feature_extractor.n_features}")

# Policy network: scores cells for protection
model = cn.CellNN(input_dim=feature_extractor.n_features, hidden_dim=16)
policy = cn.PolicyNetwork(model)
print(f"  Policy parameters: {len(policy.get_flat_weights())}")

# Reward function: biodiversity outcomes vs costs
rewards = cn.Rewards(
    reward_obj_list=[
        cn.CalcRewardExtRisk(threat_weights=np.array([1, 0, -8, -16, -32])),
        cn.CalcRewardPersistentCost(rescaler=0.1),
    ],
    reward_weights=np.array([1.0, 1.0]),
)

# =============================================================================
# Create Episode Runner
# =============================================================================
# global manager (can be focused on individual regions)
budget_manager = cn.GlobalBudgetManager(
    total_target=TARGET_PROTECTED_CELLS,
    cells_per_time_step=CELLS_PER_STEP,
    feature_updates_per_time_step=1,
)

episode = cn.EpisodeRunner(
    env=env,
    feature_extractor=feature_extractor,
    policy_network=policy,
    rewards=rewards,
    n_steps=N_STEPS,
    budget_manager=budget_manager,
)

# =============================================================================
# Run Episode
# =============================================================================

print(f"\nRunning episode ({N_STEPS} steps)...")
t0 = time.time()

info, total_reward = episode.run_episode(policy.get_flat_weights())

elapsed = time.time() - t0

# =============================================================================
# Results
# =============================================================================

print(f"\nResults:")
print(f"  Total reward: {total_reward:.4f}")
print(f"  Protected cells: {info['protected_cells']}")
print(f"  Steps completed: {info['n_steps']}")
print(f"  Time: {elapsed:.2f}s ({N_STEPS / elapsed:.1f} steps/sec)")

# Reward breakdown
print(f"\nReward components:")
for name, value in zip(rewards.names, info["rewards"]):
    print(f"  {name}: {float(value):.4f}")

# Final extinction risk distribution
print(f"\nFinal extinction risk:")
risk_counts = env.ext_risk.species_per_class(env.current_ext_risk)
labels = ["LC", "NT", "VU", "EN", "CR"]
for label, count in zip(labels, risk_counts, strict=True):
    print(f"  {label}: {int(count)} species")
