"""Reward calculation for conservation planning.

This module provides reward functions for evaluating conservation strategies,
including extinction risk changes and protection costs.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from captain.environment.bioenv import BioEnv
    from captain.utils.connectivity import PAConnectivity

logger = logging.getLogger(__name__)


class CalcReward:
    """Base class for reward calculation.

    Subclass this to implement custom reward functions.

    Attributes:
        name: Identifier for this reward component.
        rescaler: Scaling factor applied to reward.
    """

    def __init__(
            self,
            name: str = "base",
            rescaler: float = 1.0,
            positive: bool = True,
    ):
        """Initialize reward calculator.

        Args:
            name: Reward identifier.
            rescaler: Scaling factor (multiplied by reward).
            positive: If False, negate the rescaler.
        """
        self._name = name
        self._rescaler = rescaler if positive else -rescaler
        self._positive = positive

    @property
    def name(self) -> str:
        """Reward name."""
        return self._name

    def calc_reward(self, env: BioEnv) -> float:
        """Calculate reward from environment state.

        Args:
            env: Current environment.

        Returns:
            Reward value.
        """
        return 0.0

    def reset(self) -> None:
        """Reset any internal state (called at episode start)."""


class CalcRewardPersistentCost(CalcReward):
    """Reward based on cumulative protection costs.

    Penalizes expensive protection actions by computing the dot product
    of costs and protection levels.
    """

    def __init__(
            self,
            name: str = "cost",
            rescaler: float = 1.0,
            positive: bool = False,
    ):
        """Initialize cost reward.

        Args:
            name: Reward identifier.
            rescaler: Scaling factor (typically 1/total_budget).
            positive: If False, cost is a penalty (default).
        """
        super().__init__(name, rescaler, positive)

    def calc_reward(self, env: BioEnv) -> float:
        """Calculate cost penalty.

        Args:
            env: Current environment.

        Returns:
            Scaled cost penalty (negative if positive=False).
        """
        # Flatten and compute dot product
        costs = env.costs.data.flatten()
        protection = env.protection_matrix.data.flatten()
        reward = torch.dot(costs, protection).item()
        return reward * self._rescaler


class CalcRewardExtRisk(CalcReward):
    """Reward based on changes in species extinction risk.

    Tracks species movement between IUCN threat categories and applies
    weighted rewards/penalties for conservation outcomes.
    """

    def __init__(
            self,
            name: str = "extinction_risk",
            rescaler: float = 1.0,
            positive: bool = True,
            threat_weights: np.ndarray | torch.Tensor | list | None = None,
            device: torch.device | str = "cpu",
    ):
        """Initialize extinction risk reward.

        Args:
            name: Reward identifier.
            rescaler: Scaling factor.
            positive: If True, improvements yield positive reward.
            threat_weights: Weights per threat category [LC, NT, VU, EN, CR].
                           Typically [1, 0, -8, -16, -32] to penalize extinctions.
            device: PyTorch device.

        Raises:
            ValueError: If threat_weights not provided.
        """
        super().__init__(name, rescaler, positive)

        if threat_weights is None:
            raise ValueError(
                "threat_weights must be specified, e.g. [1, 0, -8, -16, -32]"
            )

        self.device = torch.device(device)

        if isinstance(threat_weights, (list, np.ndarray)):
            threat_weights = torch.tensor(threat_weights, dtype=torch.float32)
        self._threat_weights = threat_weights.to(self.device)

        self._previous_status_counts: torch.Tensor | None = None

    def calc_reward(self, env: BioEnv) -> float:
        """Calculate reward from extinction risk changes.

        Args:
            env: Current environment.

        Returns:
            Weighted sum of species shifts between risk categories.
        """
        # Ensure weights are on the same device as env
        if self._threat_weights.device != env.device:
            self._threat_weights = self._threat_weights.to(env.device)

        # Get current counts per risk class
        current_counts = env.ext_risk.species_per_class(env.current_ext_risk)

        # Initialize if first step
        if self._previous_status_counts is None:
            self._previous_status_counts = env.ext_risk.init_status_counts.clone()

        # Ensure previous counts on same device
        if self._previous_status_counts.device != current_counts.device:
            self._previous_status_counts = self._previous_status_counts.to(
                current_counts.device
            )

        # Calculate shift
        diff = current_counts - self._previous_status_counts

        # Apply weights and normalize
        weighted_diff = (diff * self._threat_weights) / env.n_species

        # Update snapshot for next step
        self._previous_status_counts.copy_(current_counts)

        return weighted_diff.sum().item() * self._rescaler

    def reset(self) -> None:
        """Reset tracking state."""
        self._previous_status_counts = None

    def to(self, device: torch.device | str) -> CalcRewardExtRisk:
        """Move to specified device.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        self.device = torch.device(device)
        self._threat_weights = self._threat_weights.to(self.device)
        if self._previous_status_counts is not None:
            self._previous_status_counts = self._previous_status_counts.to(self.device)
        return self


class CalcRewardSpeciesValue(CalcReward):
    """Reward based on the total amount of a species 'value' trait.

    Each species contributes its trait value (e.g. biomass, pollination) times its
    population, either over the whole study area or only within protected cells.
    """

    def __init__(
            self,
            trait_name: str,
            name: str | None = None,
            rescaler: float = 1.0,
            positive: bool = True,
            device: torch.device | str = "cpu",
            trait_column_indx: int | None = None,  # if None set based on trait_name
            protected_value: bool = False,  # If true only calculate the total value within protected areas
            species_range: bool = False,
            # True: weigh species only based on presence/absence; False: weigh by population
    ):
        """Initialize species value reward.

        Args:
            trait_name: Numeric column of the species trait table (BioEnv species_traits)
                that defines each species' value.
            name: Reward identifier (default: trait_name).
            rescaler: Scaling factor.
            positive: If True, more value yields a higher reward.
            device: PyTorch device.
            trait_column_indx: Index of the trait among the numeric columns; if None, it
                is looked up from trait_name.
            protected_value: If True, only count the value within protected cells (per
                protected cell); otherwise the mean population over all cells.
            species_range: With protected_value, weigh species by their protected range
                (presence) instead of their protected population.
        """
        if name is None:
            name = trait_name
        super().__init__(name, rescaler, positive)
        self.device = torch.device(device)
        self._trait_name = trait_name
        self._trait_column_indx = trait_column_indx
        self._protected_value = protected_value
        self._species_range = species_range

    def calc_reward(self, env: BioEnv) -> float:
        trait_values = env.get_trait(
            self._trait_name if self._trait_column_indx is None else self._trait_column_indx
        )

        # average value
        if self._protected_value:
            if self._species_range:
                # protected range per species (i.e. does not weigh more species
                # with higher carrying capacity)
                protected_amount_per_species = env.protected_range / torch.clamp(
                    env.protected_cells_mask.sum(), min=1.0
                )
            else:
                # Result is the average abundance per masked cell for each species
                # (i.e. weighs more species with higher carrying capacity all else equal
                # e.g. carbon or biomass)
                protected_amount_per_species = env.protected_population / torch.clamp(
                    env.protected_cells_mask.sum(), min=1.0
                )

            # Dot product with value (species-specific trait)
            reward = torch.dot(trait_values, protected_amount_per_species)
        else:
            reward = torch.dot(trait_values, env.h.mean(dim=1))
        return reward.item() * self._rescaler


class CalcRewardSpecieValue(CalcRewardSpeciesValue):
    """Deprecated alias of CalcRewardSpeciesValue (to be removed in 3.1)."""

    def __init__(self, *args: Any, **kwargs: Any):
        warnings.warn(
            "CalcRewardSpecieValue is deprecated, use CalcRewardSpeciesValue",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class CalcRewardProtectedPopulation(CalcReward):
    """Reward for protecting each species' population, favouring small-range species.

    Each species scores its protected population relative to its initial total
    population, raised to ``range_exponent`` (gamma):

        score_s = min(protected_pop_s, cap_s) / init_pop_s ** gamma
        cap_s = target * init_pop_s  (init_pop_s if target is None)

    With gamma = 1 (default) the score is the protected fraction of the species'
    population, so a protected cell is worth ~1/range for each species in it and
    protecting small-range species counts more than the same area of widespread
    ones. gamma = 0 sums absolute protected populations (range size ignored);
    values in between soften the preference for small ranges.

    Scores are weighted by fixed per-category weights, indexed by each species'
    *initial* extinction-risk status (species are not re-classified during the
    episode), and normalized by the maximum achievable value, so the reward is
    in [0, 1]. Protected population is read from the simulated populations, so
    anything lowering carrying capacity in protected cells (e.g. the BioEnv
    edge_factor) lowers the reward.
    """

    def __init__(
            self,
            name: str = "protected_population",
            rescaler: float = 1.0,
            positive: bool = True,
            threat_weights: np.ndarray | torch.Tensor | list | None = None,
            range_exponent: float = 1.0,
            target: float | None = None,
            device: torch.device | str = "cpu",
    ):
        """Initialize protected population reward.

        Args:
            name: Reward identifier.
            rescaler: Scaling factor.
            positive: If True, more protected population yields higher reward.
            threat_weights: Non-negative weight per initial risk category
                [LC, NT, VU, EN, CR], e.g. [1, 2, 4, 8, 16]. None = equal weights.
            range_exponent: Exponent gamma of the initial population in the
                denominator (1 = protected fraction, 0 = absolute population).
            target: Optional protected fraction per species (0, 1] beyond which
                further protection of that species earns no reward.
            device: PyTorch device.

        Raises:
            ValueError: If weights are negative or sum to zero, range_exponent is
                negative, or target is outside (0, 1].
        """
        super().__init__(name, rescaler, positive)
        self.device = torch.device(device)

        if threat_weights is not None:
            threat_weights = torch.as_tensor(threat_weights, dtype=torch.float32)
            if torch.any(threat_weights < 0) or threat_weights.sum() <= 0:
                raise ValueError("threat_weights must be non-negative with a positive sum")
            threat_weights = threat_weights.to(self.device)
        self._threat_weights = threat_weights

        if range_exponent < 0:
            raise ValueError(f"range_exponent must be >= 0, got {range_exponent}")
        if target is not None and not 0.0 < target <= 1.0:
            raise ValueError(f"target must be in (0, 1], got {target}")
        self._range_exponent = float(range_exponent)
        self._target = target

    def calc_reward(self, env: BioEnv) -> float:
        """Calculate the weighted, range-normalized protected population.

        Args:
            env: Current environment.

        Returns:
            Reward in [0, 1] (times rescaler).
        """
        init_pop = env.ext_risk.init_pop
        if init_pop is None:
            raise RuntimeError("Initial populations not set, call env.reset() first")
        init_pop = init_pop.to(env.device)
        protected_pop = env.protected_population

        # Fixed per-species weights from the initial risk status
        if self._threat_weights is None:
            weights = torch.ones_like(init_pop)
        else:
            if self._threat_weights.device != env.device:
                self._threat_weights = self._threat_weights.to(env.device)
            status = env.ext_risk.init_status.to(env.device)
            if status.max() >= len(self._threat_weights):
                raise ValueError(
                    f"threat_weights has {len(self._threat_weights)} entries but "
                    f"initial status reaches class {status.max().item()}"
                )
            weights = self._threat_weights[status]

        # Species absent at the start cannot be protected
        present = init_pop > 0
        weights = weights * present
        init_pop = torch.where(present, init_pop, torch.ones_like(init_pop))

        cap = init_pop if self._target is None else self._target * init_pop
        denom = init_pop**self._range_exponent
        score = torch.minimum(protected_pop, cap) / denom
        max_score = cap / denom

        reward = (weights * score).sum() / (weights * max_score).sum().clamp(min=1e-12)
        return reward.item() * self._rescaler

    def to(self, device: torch.device | str) -> CalcRewardProtectedPopulation:
        """Move to specified device.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        self.device = torch.device(device)
        if self._threat_weights is not None:
            self._threat_weights = self._threat_weights.to(self.device)
        return self


def protected_fractions(env: BioEnv, target: float | None = None) -> torch.Tensor:
    """Protected fraction of each species' initial population, shape (n_species,).

    min(protected_pop, cap) / cap with cap = target * initial population (the initial
    population if target is None), so values are in [0, 1]. Species absent at the
    start (initial population 0) get NaN.
    """
    init_pop = env.ext_risk.init_pop
    if init_pop is None:
        raise RuntimeError("Initial populations not set, call env.reset() first")
    init_pop = init_pop.to(env.device)
    cap = init_pop if target is None else target * init_pop
    frac = torch.minimum(env.protected_population, cap) / cap.clamp(min=1e-12)
    return torch.where(init_pop > 0, frac, torch.full_like(frac, float("nan")))


def worst_species(fractions: torch.Tensor, worst_fraction: float) -> torch.Tensor:
    """Indices of the worst-served species: the lowest ceil(worst_fraction * n) fractions.

    NaN entries (species absent at the start) are never selected.
    """
    valid = ~torch.isnan(fractions)
    k = max(1, int(np.ceil(worst_fraction * int(valid.sum()))))
    ranked = torch.where(valid, fractions, torch.full_like(fractions, float("inf")))
    return torch.topk(ranked, k, largest=False).indices


def tail_threat_weights(
        env: BioEnv, species: torch.Tensor, threat_weights: torch.Tensor
) -> torch.Tensor:
    """IUCN weights (by initial status) of the given species, shape (len(species),)."""
    status = env.ext_risk.init_status.to(env.device)[species]
    threat_weights = threat_weights.to(env.device)
    if status.max() >= len(threat_weights):
        raise ValueError(
            f"threat_weights has {len(threat_weights)} entries but initial status reaches "
            f"class {status.max().item()}"
        )
    return threat_weights[status]


class CalcRewardWorstSpecies(CalcReward):
    """Reward for the worst-served species ("leave no species behind").

    The mean protected fraction (as in CalcRewardProtectedPopulation with gamma = 1) of
    the worst-served share of species: a soft maximin. The worst-served are the
    worst_fraction of species with the lowest fractions, chosen by count whatever their
    status; worst_fraction = 1 / n_species is a pure maximin (the single worst species),
    larger values give a smoother signal. With threat_weights, the mean over these
    species is weighted by their initial IUCN status (the selection is not), so lifting
    a threatened species in the tail counts more without excluding the others. Which
    species are worst-served is re-evaluated every call, so protecting one of them
    shifts the reward to the next. Reward in [0, 1].
    """

    def __init__(
            self,
            name: str = "species_worst",
            rescaler: float = 1.0,
            positive: bool = True,
            worst_fraction: float = 0.1,
            target: float | None = None,
            threat_weights: np.ndarray | torch.Tensor | list | None = None,
    ):
        """Initialize worst-served species reward.

        Args:
            name: Reward identifier.
            rescaler: Scaling factor.
            positive: If True, better protection of the worst-served yields more reward.
            worst_fraction: Share (0, 1] of species averaged over (the lowest fractions).
            target: Optional protected fraction (0, 1] at which a species counts as
                fully protected (as in CalcRewardProtectedPopulation).
            threat_weights: Optional non-negative weight per initial risk category
                [LC, NT, VU, EN, CR] for the mean within the tail. None = unweighted.

        Raises:
            ValueError: If worst_fraction or target is outside (0, 1], or weights are
                negative or sum to zero.
        """
        super().__init__(name, rescaler, positive)
        if not 0.0 < worst_fraction <= 1.0:
            raise ValueError(f"worst_fraction must be in (0, 1], got {worst_fraction}")
        if target is not None and not 0.0 < target <= 1.0:
            raise ValueError(f"target must be in (0, 1], got {target}")
        self._worst_fraction = float(worst_fraction)
        self._target = target
        if threat_weights is not None:
            threat_weights = torch.as_tensor(threat_weights, dtype=torch.float32)
            if torch.any(threat_weights < 0) or threat_weights.sum() <= 0:
                raise ValueError("threat_weights must be non-negative with a positive sum")
        self._threat_weights = threat_weights

    def calc_reward(self, env: BioEnv) -> float:
        """(Weighted) mean protected fraction of the worst-served species (times rescaler)."""
        fractions = protected_fractions(env, self._target)
        worst = worst_species(fractions, self._worst_fraction)
        if self._threat_weights is None:
            return fractions[worst].mean().item() * self._rescaler
        weights = tail_threat_weights(env, worst, self._threat_weights)
        reward = (weights * fractions[worst]).sum() / weights.sum().clamp(min=1e-12)
        return reward.item() * self._rescaler


class CalcRewardLayerOverlap(CalcReward):
    """Reward for protecting the cells of a spatial layer (e.g. a trail).

    Share of the layer's total value inside protected cells:
    sum(layer * protected) / sum(layer), in [0, 1]. With a trail layer holding the
    fraction of each cell crossed by the trail, this is the share of trail length that
    is protected.
    """

    def __init__(
            self,
            layer: np.ndarray | torch.Tensor,
            name: str = "layer_overlap",
            rescaler: float = 1.0,
            positive: bool = True,
    ):
        """Initialize layer overlap reward.

        Args:
            layer: Non-negative value per cell, shape (n_cells,).
            name: Reward identifier.
            rescaler: Scaling factor.
            positive: If True, more protected layer value yields more reward.

        Raises:
            ValueError: If the layer has negative values or sums to zero.
        """
        super().__init__(name, rescaler, positive)
        layer = torch.as_tensor(layer, dtype=torch.float32).reshape(-1)
        if torch.any(layer < 0) or layer.sum() <= 0:
            raise ValueError("layer must be non-negative with a positive sum")
        self._layer = layer / layer.sum()

    def calc_reward(self, env: BioEnv) -> float:
        """Protected share of the layer (times rescaler)."""
        if self._layer.device != env.device:
            self._layer = self._layer.to(env.device)
        protected = env.protected_cells_mask.to(self._layer.dtype)
        return torch.dot(self._layer, protected).item() * self._rescaler


class CalcRewardConnectivity(CalcReward):
    """Reward for connecting protected-area units (Equivalent Connected Area gain).

    Uses a PAConnectivity instance: units (e.g. the existing protected areas) are linked
    through least-cost paths over unprotected cells, and the reward is the gain in
    Equivalent Connected Area scaled to [0, 1] (0 = initial protection, 1 = all units
    connected). It increases continuously as gaps between units shrink, not only when
    a link is completed.
    """

    def __init__(
            self,
            connectivity: PAConnectivity,
            name: str = "connectivity",
            rescaler: float = 1.0,
            positive: bool = True,
    ):
        """Initialize connectivity reward.

        Args:
            connectivity: PAConnectivity defining units, resistance and decay.
            name: Reward identifier.
            rescaler: Scaling factor.
            positive: If True, better connectivity yields higher reward.
        """
        super().__init__(name, rescaler, positive)
        self._connectivity = connectivity

    @property
    def connectivity(self) -> PAConnectivity:
        """The PAConnectivity instance the reward is computed with."""
        return self._connectivity

    def calc_reward(self, env: BioEnv) -> float:
        """ECA gain of the current protection, in [0, 1] (times rescaler)."""
        return self._connectivity.eca_gain(env.protected_cells_mask) * self._rescaler
