# Reward Functions

This tutorial explains the reward terms available in CAPTAIN
([`captain/agents/rewards.py`](../../captain/agents/rewards.py)), what each one
measures, and how to choose between them. For a runnable example that combines several
of them on the toy data (with edge effects and connectivity), see the
[advanced example](../advanced_example/README.md). It covers:

- how reward terms are combined, accumulated and calibrated, and how to change their
  weights over training (reward weight schedules)
- extinction risk (`CalcRewardExtRisk`)
- protection cost (`CalcRewardPersistentCost`)
- species value (`CalcRewardSpecieValue`)
- protected population (`CalcRewardProtectedPopulation`)
- connectivity among protected areas (`CalcRewardConnectivity`)
- worst-served species (`CalcRewardWorstSpecies`) and spatial layers such as trails
  (`CalcRewardLayerOverlap`)

## 1. How rewards are combined

Reward terms are passed to `cn.Rewards`, which evaluates every term at each
time step of an episode:

```python
rewards = cn.Rewards(
    reward_obj_list=[
        cn.CalcRewardProtectedPopulation(threat_weights=[1, 2, 4, 8, 16]),
        cn.CalcRewardPersistentCost(rescaler=float(1.0 / costs.data.sum())),
    ],
    reward_weights=np.array([1.0, 1.0]),
)
```

- **Accumulation.** By default (`cumulative_reward=True`) the per-step values
  of each term are summed over the episode. For terms that measure a *change*
  (extinction risk) the sum is the total change; for terms that measure a
  *state* (cost, species value, protected population) it is the state summed
  over time, so protecting good cells early earns more than protecting them
  late.
- **Weighting.** The episode reward is
  `sum(term totals * reward_weights * calibration)`. `reward_weights` express
  your priorities; the calibration multipliers put the terms on comparable
  scales.
- **Calibration.** `EvolStrategiesTrainer.get_reward_calibrated_weights()` runs
  probe episodes and derives one multiplier per term from the spread of its
  values, saved as `reward_calibration.json` and loaded with
  `trainer.load_reward_calibration(...)`. The file is keyed by reward `name`, so
  it must be regenerated when the list of terms changes. Calibrate against the
  global agent only (see [toy example §1](../toy_example/README.md)).
  The probes perturb the *untrained* policy, so a term that hardly varies for an
  untrained policy (e.g. `CalcRewardWorstSpecies`, which is 0 while many species are
  unprotected) gets a very large multiplier and can dominate the trained objective.
  For such terms prefer a reward weight schedule (below).
- **Sign.** Every term takes `positive` and `rescaler`; `positive=False` flips
  the sign (used by the cost term to make it a penalty).
- **Reward scale and step size.** By default
  (`EvolStrategiesTrainer(..., advantage_normalization="zscore")`) the rewards of the
  perturbations are standardized within each epoch: the step size then depends only on
  the learning rate and sigma, while the weights and calibration of the terms still
  decide which perturbations count as better. `"rank"` uses centred ranks instead.
  With `advantage_normalization=None` the update is proportional to each
  perturbation's reward minus a running baseline, so the step size grows with the
  scale of the total reward (calibration typically multiplies rewards by 10-50x,
  which can make early epochs unstable).

### Reward weight schedules

Some terms give no training signal at the start. `CalcRewardWorstSpecies(worst_fraction=0.1)`
is exactly 0 while more than 10% of the species have no protection, and so is every
plan that concentrates on the species favoured by the other terms. With fixed weights,
evolution strategies then see no difference between perturbations for that term, and
training can settle on plans that ignore it. A `RewardWeightSchedule` multiplies chosen
terms by a factor that changes over the epochs, on top of `reward_weights` and the
calibration:

```python
schedule = cn.RewardWeightSchedule({
    # term name -> factor from 100x at epoch 0 to 1x at epoch 100 (then 1x)
    "species_worst": cn.FactorSchedule(start=100.0, epochs=100),
})
trainer = cn.EvolStrategiesTrainer(runners, weights, reward_weight_schedule=schedule)
```

- **Shape.** `shape="geometric"` (default) lowers the factor by the same ratio each
  epoch (100x, 10x halfway, 1x at the end); `shape="linear"` changes it in equal
  steps. `end` (default 1) is the factor after the schedule; with `end=1` the
  objective at the end of training is the one set by `reward_weights`. A factor can
  also grow (`start < end`), e.g. to bring in a term after the policy has learned the
  others.
- **Multi-objective use.** Start the flat term high so the policy first learns to
  spread protection (e.g. so that every species gets some), then lower it to its
  final weight: training then moves towards the balance of the final objective from
  plans where the term varies. Give the schedule enough epochs, and keep training
  for a while after it ends.
- **Workers and checkpoints.** The trainer sends the multipliers of each epoch with
  every episode, so all worker processes use them, and `save_checkpoint` stores the
  epoch so a resumed run continues the schedule.
- **Logs.** With a schedule, `TrainingLogger` adds `mult_<term>` columns with the
  multipliers of each epoch. The total reward (`avg_reward`) follows the changing
  weights, so compare the per-term columns (`r_<term>`) across epochs.
- **Advantages.** Use the default `advantage_normalization="zscore"` (or `"rank"`): with
  `None` the running reward baseline mixes reward scales across epochs.

See [`examples/train_policy_weight_schedule.py`](../../examples/train_policy_weight_schedule.py)
and the [advanced example](../advanced_example/README.md), which anneal the weight of
a worst-served species term from 100x to 1x over the first half of the training.

## 2. Extinction risk: `CalcRewardExtRisk`

Rewards species moving to lower extinction-risk classes and penalizes moves to
higher ones.

```python
cn.CalcRewardExtRisk(threat_weights=np.array([1, 0, -8, -16, -32]))
#                                      LC  NT  VU   EN   CR
```

**How it works.** Species are re-classified at every step by
`cn.ExtinctionRisk`:

```
status = initial IUCN status + bucket(population decline) - bucket(gain in protected share)
```

Both quantities are compared with the start of the episode and binned with
thresholds `(k / n_classes) ** (1 / alpha)`; with `alpha = 0.5` a species moves
one class when its decline, or the gain in the fraction of its population
inside protected areas, crosses 4%, 16%, 36% or 64%. The reward at each step is
the change in the number of species per class, weighted by `threat_weights`
and divided by the number of species; summed over the episode this is the
final minus the initial class composition.

**Use it when** you want an outcome in Red List terms (how many species end up
in each category).

**Limitations.**
- The signal is discrete: it only changes when a species crosses a threshold.
  Small but systematic differences between protection plans (e.g. the edge
  effect below, or which cells of a species' range get protected) are mostly
  invisible to it.
- Its scale depends on the number of species changing class, so it needs
  calibration against the other terms.

## 3. Protection cost: `CalcRewardPersistentCost`

Penalizes the cost of the protected area.

```python
cn.CalcRewardPersistentCost(rescaler=float(1.0 / costs.data.sum()))
```

**How it works.** At each step: `-(sum over cells of cost * protection) *
rescaler`. With `rescaler = 1 / total cost`, a step's value is minus the
fraction of the total landscape cost that is currently protected. Because the
cost is paid at every step a cell stays protected ("persistent"), protecting
expensive cells early costs more than protecting them late. Existing protected
areas add a constant that does not affect which policy is better.

**Use it when** cells differ in acquisition or opportunity cost and you want
the agent to prefer cheaper cells for the same conservation gain. Note that the
*number* of protected cells is set by the budget manager, not by this term: it
only shifts *which* cells are chosen.

## 4. Species value: `CalcRewardSpecieValue`

Rewards the amount of a species trait (an ecosystem-service value such as
pollination, biomass or carbon) held by the populations.

```python
cn.CalcRewardSpecieValue(
    trait_name="pollination",  # a numeric column of the species trait table
    protected_value=True,
    species_range=False,
)
```

**How it works.** `reward = sum over species of trait_s * amount_s`, where
`amount_s` depends on the mode:

| `protected_value` | `species_range` | `amount_s` |
|---|---|---|
| `False` | - | mean population per cell over the whole landscape |
| `True` | `False` | population in protected cells / number of protected cells |
| `True` | `True` | number of protected cells where the species occurs / number of protected cells |

Amounts are **absolute**: abundant, widespread species contribute the most,
which is what you want for a service that scales with biomass or abundance.
The `species_range=True` mode only counts presence, so it does not respond to
changes in population size within protected cells. Set `rescaler` so the term
has a sensible scale, e.g. `1 / sum(mean population * trait)`.

**Use it when** the objective is a trait-weighted service rather than species
persistence.

## 5. Protected population: `CalcRewardProtectedPopulation`

Rewards protecting as much of each species' population as possible, with
fixed weights by extinction-risk category and a preference for small-range
species.

```python
cn.CalcRewardProtectedPopulation(
    threat_weights=[1, 2, 4, 8, 16],  # LC, NT, VU, EN, CR; None = equal weights
    range_exponent=1.0,               # gamma
    target=None,                      # optional cap on the protected fraction, e.g. 0.3
)
```

**How it works.** For each species, with `P_s` the population in protected
cells and `N_s` the species' total population at the start of the episode:

```
score_s = min(P_s, cap_s) / N_s ** gamma        cap_s = target * N_s (N_s if no target)
reward  = sum(w_s * score_s) / sum(w_s * cap_s / N_s ** gamma)
```

- **Weights `w_s`** come from `threat_weights`, indexed by each species'
  *initial* status. Species are not re-classified during the episode, so the
  weighting is fixed and the reward does not depend on the extinction-risk
  thresholds.
- **`range_exponent` (gamma)** sets how strongly small-range species are
  favoured. With `gamma = 1` (default) the score is the protected *fraction* of
  each species' population, so every species counts equally regardless of its
  range, and a protected cell is worth about `1 / range` for each species in
  it. With `gamma = 0` the score is the absolute protected population (range
  ignored); values in between soften the preference.
- **`target`** saturates the score: once a species has `target` of its
  population protected, more protection of that species earns nothing, which
  pushes the agent to spread protection across species (complementarity).
- The denominator is the maximum achievable value, so the reward lies in
  `[0, 1]` (per step).

**Example.** Three species with ranges of 10, 100 and 1000 cells (one
individual per cell), equal weights, 30 protected cells:

| Protected cells per species | gamma = 1 | gamma = 0 |
|---|---|---|
| 10, 10, 10 | (1 + 0.1 + 0.01) / 3 = **0.370** | 30 / 1110 = 0.027 |
| 1, 5, 24 | (0.1 + 0.05 + 0.024) / 3 = **0.058** | 30 / 1110 = 0.027 |

With `gamma = 1` protecting the small-range species is strongly preferred; with
`gamma = 0` the two plans tie because they protect the same area.

**Use it when** the objective is to secure the populations of as many species
as possible, prioritizing threatened and range-restricted ones. Because
`P_s` is read from the simulated populations, every change in carrying
capacity inside protected cells registers continuously, including the edge
effect.

**Caveats.**
- With `gamma = 1`, species with very small (possibly spurious) ranges get very
  high per-cell value; check the quality of their distribution data.
- Without a `target` the reward is linear: fully protecting one species is
  worth as much as protecting 10% of ten species.
- Protected population can exceed the initial population (protection removes
  disturbance and populations grow); it is capped at `N_s` (or `target * N_s`).

## 6. Connectivity: `CalcRewardConnectivity`

Rewards linking protected-area units (e.g. the existing protected areas)
through newly protected land.

```python
conn = cn.PAConnectivity(
    env.sdms._coords,
    units,               # unit id per cell: 0 = none, k = unit k (e.g. labelled PAs)
    resistance,          # cost of crossing each unprotected cell (e.g. relative cost)
    initial_protection=env.protected_cells_mask,
    alpha=20.0,          # distance decay, in resistance units
    area_exponent=0.5,   # unit weight = area ** exponent
)
cn.CalcRewardConnectivity(conn)
```

**How it works.** The landscape is a graph of cells with 8-neighbour moves.
Entering an unprotected cell costs its resistance, protected cells are free, so
the gap between two units is the least-cost distance through the land that
still needs protecting (0 once they are linked). With unit weights
`w_i = area_i ** area_exponent`, the Equivalent Connected Area is

```
ECA = sqrt(sum_ij w_i w_j exp(-gap_ij / alpha))
```

and the reward is the ECA gain scaled to `[0, 1]`: 0 for the initial
protection, 1 when all units are connected. Every cell that shortens a gap
raises it, so the signal is continuous rather than "connected or not".

- **`alpha`** sets how far apart units can be and still count as partly
  connected; gaps much larger than `alpha` contribute almost nothing.
- **`area_exponent`**: 1 is the standard ECA (large units dominate); 0 weighs
  every unit the same; values in between let small reserves count.
- Use the same layer for **resistance** as for the cost reward: the gaps then
  follow the cheapest links, and the cost reward decides whether a link is worth
  its price.

**The `pa_linking` feature.** A cell-based policy cannot see that a cell is
part of a missing link far away. `FeatureExtractor(..., feature_set=[...,
"pa_linking"], connectivity=conn)` provides it: for each pair of unconnected
units, cells on the least-cost link get the pair's weight `w_i w_j
exp(-gap_ij / alpha)`, cells off it a share that decays with how much longer
the link through them is (`beta`), summed and scaled to `[0, 1]`. It is
recomputed during the episode (every `feature_refresh_every` decisions).

**Use it when** protected areas should be linked into a network. There is no
minimum corridor width: narrow links are penalised through the edge effect
instead (section 8).

## 7. Worst-served species and spatial layers

**`CalcRewardWorstSpecies`: leave no species behind.** A mean over species (as in
`CalcRewardProtectedPopulation`) can be raised by protecting already well-covered
species while a few are left with nothing. This term rewards the mean protected
fraction of the worst-served share of species, a soft maximin:

```python
cn.CalcRewardWorstSpecies(worst_fraction=0.1)   # mean of the lowest 10% of fractions
```

Fractions are `min(P_s, cap_s) / cap_s` (with an optional `target`, as above), and
which species are worst-served is re-evaluated at every call, so lifting one hands
the reward to the next. The tail is chosen by count, whatever the species' status;
`threat_weights=[1, 2, 4, 8, 16]` optionally weights the mean within it by initial
IUCN status, so lifting a threatened species in the tail counts more while poorly
protected LC species stay in it. (Choosing the tail as a share of total weight
instead, by IUCN status or range size, makes its size depend on how well the heavily
weighted species are already protected and can swallow most species.) `worst_fraction = 1 / n_species`
is a pure maximin (only the single worst species), which changes in steps and gives
evolution strategies little signal; 5-20% is smoother. The optional
`"worst_species_value"` feature (`FeatureExtractor(..., worst_fraction=...)`) shows the
policy which cells hold the currently worst-served species. Combine it with
`CalcRewardProtectedPopulation`: on its own it ignores everything but the worst-off.
The term is exactly 0 while more than `worst_fraction` of the species are unprotected,
which gives no training signal. If the other terms concentrate protection on a few
species, anneal its weight with a [reward weight schedule](#reward-weight-schedules)
instead of fixing it (or calibrating it).

**`CalcRewardLayerOverlap`: protect a spatial layer.** Rewards the share of a
non-negative layer inside protected cells, `sum(layer * protected) / sum(layer)` in
`[0, 1]`: e.g. the length of a hiking trail (layer = share of each cell crossed by
it), carbon stock or scenic value. Add the layer as a static feature so the policy
can see it.

```python
cn.CalcRewardLayerOverlap(trail_per_cell, name="trail")
```

## 8. Rewards and the edge effect

`BioEnv(edge_factor=..., edge_radius=1)` lets disturbance spill over into
protected cells: a protected cell receives `edge_factor` times its edge
exposure, so isolated cells and thin corridors support smaller populations
than the core of a compact reserve. The exposure is a mean over the neighbours
within `edge_radius` cells (the cell itself excluded) of:

- `edge_source="disturbance"` (default): the neighbour's disturbance if it is
  unprotected, so edges facing undisturbed land cost nothing;
- `edge_source="unprotected"`: 1 if the neighbour is unprotected, regardless of
  land use, i.e. one minus the mean protection of the neighbourhood. An
  isolated protected cell has exposure 1.

With `edge_decay`, neighbours are weighted by `exp(-distance / edge_decay)`
instead of equally, so with a larger `edge_radius` the effect reaches deeper
into a reserve and fades with distance from its border. The agent only learns
to avoid edge-heavy layouts if a reward term responds to the edge effect:

| Term | Responds to the edge effect? |
|---|---|
| `CalcRewardExtRisk` | Only when the change pushes a species across a class threshold (weak, noisy) |
| `CalcRewardPersistentCost` | No (cost does not depend on populations) |
| `CalcRewardSpecieValue` | Yes with `species_range=False`; no with `species_range=True` |
| `CalcRewardProtectedPopulation` | Yes, continuously |
| `CalcRewardConnectivity` | No (a link counts once its cells are protected) |
| `CalcRewardWorstSpecies` | Yes, through the worst-served species' populations |
| `CalcRewardLayerOverlap` | No (counts protected cells only) |

In a test on the toy data (2,000 cells protected either as compact 5x5 blocks
or as scattered single cells, 5 random replicates, 50 time steps,
`edge_factor = 0.9`), the edge effect lowered `CalcRewardProtectedPopulation` by
about 2% for compact layouts and 5% for scattered ones with
`edge_source="disturbance"`, and by 15% vs. 51% with `edge_source="unprotected"`,
consistently across replicates. `CalcRewardExtRisk` did not separate the two
layouts (the same few class changes, or none, in both).

## 9. Choosing rewards

| Term | Measures | Signal | Scale |
|---|---|---|---|
| `CalcRewardExtRisk` | Red List class changes | Discrete (thresholds) | Depends on species counts |
| `CalcRewardPersistentCost` | Cost of protected cells | Continuous | Set by `rescaler` |
| `CalcRewardSpecieValue` | Trait-weighted absolute amounts | Continuous | Set by `rescaler` |
| `CalcRewardProtectedPopulation` | Risk-weighted protected fraction of each species | Continuous | `[0, 1]` per step |
| `CalcRewardConnectivity` | ECA gain among protected-area units | Continuous | `[0, 1]` per step |
| `CalcRewardWorstSpecies` | Protected fraction of the worst-served species | Continuous (soft maximin) | `[0, 1]` per step |
| `CalcRewardLayerOverlap` | Protected share of a spatial layer | Continuous | `[0, 1]` per step |

A typical biodiversity setup combines one biodiversity term
(`CalcRewardProtectedPopulation` or `CalcRewardExtRisk`) with
`CalcRewardPersistentCost`, plus `CalcRewardSpecieValue` when an ecosystem
service matters and `CalcRewardConnectivity` to link protected areas. Recalibrate
whenever the set of terms changes (terms already on `[0, 1]` scales usually need no
calibration).
