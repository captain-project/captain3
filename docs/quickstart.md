# Quick Start

This guide walks through a complete training example on the toy dataset. It mirrors
[`examples/train_policy.py`](../examples/train_policy.py) and
[`examples/run_inference.py`](../examples/run_inference.py) — see those scripts for the
full, runnable versions (regional agents, reward calibration, logging, plotting, etc.)
and the [toy-example tutorial](../tutorials/toy_example/README.md) for a walkthrough.

## Download Example Data

Download the toy dataset from
[Polybox](https://polybox.ethz.ch/index.php/s/qYTHM7difqrBmo4) and unzip it to `data/`
in the repository root (or point `DATA_DIR` below to wherever you put it). It contains a
`present_sdms/` directory of habitat-suitability GeoTIFFs, a matching `future_sdms/`
directory, a species trait table (`species_tbl.csv`) and an `env_layers/` directory with
the disturbance, cost and mask layers.

## Basic Training Script

```python
"""Basic training example for CAPTAIN."""

from pathlib import Path

import numpy as np
import torch

import captain as cn

DATA_DIR = Path("data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_TIME_STEPS = 50
N_EPOCHS = 50
TARGET_PROTECTED_CELLS = 1000
CELLS_PER_STEP = 50

# 1. Load the area mask and species distribution models (present + future)
mask, _ = cn.data_loader.load_map(DATA_DIR / "env_layers" / "area_mask.npy")

sdm = cn.load_spatial_data_from_dir(
    dir=DATA_DIR / "present_sdms",
    future_dir=DATA_DIR / "future_sdms",
    mask=mask,
    lower_bound=0,
    upper_bound=1,
    n_time_steps=N_TIME_STEPS,
    min_threshold=0.05,
)

# 2. Load environmental layers (disturbance, costs), each with a future variant
disturbance = cn.load_spatial_data(
    file=DATA_DIR / "env_layers" / "area_swept_disturbance.tif",
    future_file=DATA_DIR / "env_layers" / "future_area_swept_disturbance.tif",
    mask=mask,
    lower_bound=0,
    upper_bound=1,
    n_time_steps=N_TIME_STEPS,
)

costs = cn.load_spatial_data(
    file=DATA_DIR / "env_layers" / "cost.tif",
    future_file=DATA_DIR / "env_layers" / "future_cost.tif",
    mask=mask,
    lower_bound=0,
    upper_bound=1,
    n_time_steps=N_TIME_STEPS,
)

# Protection matrix starts empty
protection = cn.SpatialData(
    data=np.zeros(disturbance.shape), mask=mask, lower_bound=0, upper_bound=1
)

# 3. Load species traits, aligned to the SDM species order
traits = cn.data_loader.load_trait_table(
    DATA_DIR / "species_tbl.csv", sdm.names, ref_column="species", fill_gaps=True
)
# Per-species minimum habitat suitability (use the same thresholds at inference)
sdm.reset_threshold(traits["min_habitat_suitability"].to_numpy())
growth_rates = traits["growth_rate"].to_numpy() + 1.0
carrying_capacity = 100 / traits["conservation_status"].to_numpy()
conservation_status = traits["conservation_status"].to_numpy() - 1

# 4. Initial extinction risk from conservation status
ext_risk = cn.ExtinctionRisk(init_status=conservation_status, n_classes=5, alpha=0.5)

# 5. Create environment
env = cn.BioEnv(
    sdms=sdm,
    disturbance=disturbance,
    costs=costs,
    protection_matrix=protection,
    species_k=carrying_capacity,
    growth_rates=growth_rates,
    sensitivity_rates=traits["sensitivity_disturbance"].to_numpy()[:, np.newaxis],
    ext_risk=ext_risk,
    device=DEVICE,
)

# 6. Create feature extractor and policy
feature_extractor = cn.FeatureExtractor(env, device=DEVICE)
model = cn.CellNN(input_dim=feature_extractor.n_features, hidden_dim=32)
policy = cn.PolicyNetwork(model, seed=42, device=DEVICE)

# 7. Create rewards
rewards = cn.Rewards([
    cn.CalcRewardExtRisk(threat_weights=[1, 0, -8, -16, -32]),
    cn.CalcRewardPersistentCost(rescaler=1.0 / costs.data.sum()),
])

# 8. Create the episode runner and budget manager (how many cells get
#    protected, and how fast)
budget_manager = cn.GlobalBudgetManager(
    total_target=TARGET_PROTECTED_CELLS,
    cells_per_time_step=CELLS_PER_STEP,
)
episode = cn.EpisodeRunner(
    env=env,
    feature_extractor=feature_extractor,
    policy_network=policy,
    rewards=rewards,
    n_steps=N_TIME_STEPS,
    budget_manager=budget_manager,
)

# 9. Create trainer and train
trainer = cn.EvolStrategiesTrainer(
    [episode],
    initial_coeffs=policy.get_flat_weights(),
    n_perturbations=4,
    seed=42,
)

print("Starting training...")
for epoch in range(N_EPOCHS):
    avg_reward, _summary = trainer.train_epoch()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Reward: {avg_reward:.4f}")

# 10. Save the trained policy weights (a flat NumPy array)
np.save("trained_policy.npy", trainer.epoch_coeff)
print("Training complete!")
```

## Running an Episode

After training, load the saved weights and run a single episode to evaluate:

```python
# Load trained weights and inject them into the policy
trained_weights = np.load("trained_policy.npy")
policy.set_flat_weights(trained_weights)

# Run episode
info, total_reward = episode.run_episode(trained_weights)

print(f"Total reward: {total_reward:.4f}")
print(f"Protected cells: {info['protected_cells']}")
```

## Next Steps

- [Training guide](training.md) - optimizer settings, reward calibration, weight
  schedules, parallel training
- [Inference outputs](inference.md) - saving features and rankings, Google Earth export
- [Architecture overview](architecture.md) - how the components fit together
- [Advanced example](../tutorials/advanced_example/README.md) - edge effects,
  connectivity, static prioritization
