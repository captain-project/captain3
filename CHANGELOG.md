# Changelog

All notable changes to CAPTAIN v.3. Versions follow [PEP 440](https://peps.python.org/pep-0440/).

## [3.0.0rc1] - 2026-11-04

First release candidate of CAPTAIN v.3. Changes since the 3.0.0 beta preview (August 2026).

### Added

- **Edge effect** in `BioEnv` (`edge_factor`, `edge_radius`, `edge_decay`, `edge_source`):
  disturbance spills over from unprotected land into the borders of protected areas.
- **Connectivity** among protected areas: `PAConnectivity` (Equivalent Connected Area over
  least-cost gaps), the `CalcRewardConnectivity` reward and the `pa_linking` policy feature.
- **Reward terms**: `CalcRewardProtectedPopulation` (threat- and range-weighted protected share
  of each species, optional per-species target), `CalcRewardWorstSpecies` (worst-served
  species), `CalcRewardLayerOverlap` (protected share of any spatial layer).
- **Policy features**: `edge_exposure`, `buffer_value`, `pa_linking`, `worst_species_value`.
- **Reward weight schedules**: `RewardWeightSchedule` and `FactorSchedule` change the weight of
  chosen reward terms over the epochs.
- **Static single-step episodes**: `EpisodeRunner(env_dynamics=False)` for one-shot
  prioritization without population dynamics.
- **Google Earth export** of protection plans as KMZ/KML: `cn.export.raster_to_kml`,
  `cn.export.geotiff_to_kml`.
- **Public accessors** for what the examples used to read from private attributes:
  `SpatialData.coords`, `.nonzero_cells_mask`, `.to_grid()`; `BioEnv.species_k`,
  `.growth_rates`, `.sensitivity_rates`, `.carrying_capacity`, `.species_traits`,
  `.get_trait()`, `.step_num`, `.env_step_num`, `.edge_factor`; `ExtinctionRisk.n_classes`,
  `.init_status_counts`; `Rewards.terms`, `.get_term()`; `CalcRewardConnectivity.connectivity`.
- Examples `train_policy_advanced.py`, `static_prioritization.py`,
  `train_policy_weight_schedule.py`; tutorials on rewards and an advanced example; a quick
  start, training guide, inference guide and architecture overview in `docs/`.
- Advice on how many epochs to train, from a 100-epoch study with three seeds
  (`scripts/convergence_study.py`, "How long to train" in `docs/training.md`).
- Test suite (unit and integration) and continuous integration (lint, type check, tests on
  Python 3.11–3.13).

### Changed

- The evolution strategies trainer standardizes the rewards of each epoch's perturbations
  before the update (`advantage_normalization="zscore"`), so the step size no longer grows
  with the reward scale. Pass `advantage_normalization=None` for the previous update.
  Training results differ from the preview.
- `LearningScheduler` keeps sigma at or above `min_sigma` also while adapting it to the
  similarity of the perturbations' plans.
- Budget managers round the number of cells protected per decision up, so the whole
  protection target is reached.
- Species in the trait table without a species distribution map are dropped with a warning
  instead of raising an error.
- `CalcRewardSpecieValue` is renamed `CalcRewardSpeciesValue`; the old name still works with a
  `DeprecationWarning` and will be removed in 3.1.
- `FeatureExtractor(trait_features=...)` raises `NotImplementedError` (the features were never
  filled); pass trait-based layers as `static_features`.
- `load_map` reports multi-channel rasters through `logging` instead of `print`, and no longer
  changes the process-wide warning filters.
- `FeatureExtractorCNN`, `SpatialCNN` and `CellCNNPolicy` are marked experimental: their API
  may change in a minor release.
- The package is distributed as `captain3` (the import name is still `captain`), and its
  version follows PEP 440 (`3.0.0rc1` instead of `3.0.0 beta - <date>`).

### Removed

- Experimental classes that were not part of the documented API: `BridgeEpisodeRunner`,
  `CalcRewardDistFromTarget`, `CalcRewardEdgeEffect`, `grid_utils.DistanceFromTarget`,
  `BioEnv(distance_calculator=...)` and `BioEnv.get_distance_from_target`.

### Fixed

- With several CPU workers, a worker could start without an episode runner, and training
  stopped at the first epoch with "Worker not initialized".
- `SpatialData(multi_per_step=...)` raised a `TypeError` for 2-D input;
  `SpatialData.reset_threshold` crashed on an integer threshold.
- `EvolStrategiesTrainer` raised an error at interpreter shutdown when its construction had
  failed.
- The example scripts no longer contain hard-coded paths, and the Google Earth export
  takes its georeference from the species distribution maps.
