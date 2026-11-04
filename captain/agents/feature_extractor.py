"""Feature extraction from environment state for policy network input.

This module provides the FeatureExtractor class that transforms raw environment
state into normalized features suitable for neural network input.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from captain.agents.rewards import protected_fractions, tail_threat_weights, worst_species
from captain.utils import grid_utils
from captain.utils.grid_utils import scipy_sparse_to_torch

if TYPE_CHECKING:
    from captain.environment.bioenv import BioEnv
    from captain.utils.connectivity import PAConnectivity

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract and normalize features from environment state.

    Transforms raw environment observations into normalized feature vectors
    suitable for policy network input. Features include time, disturbance,
    population metrics, extinction risk, costs, and protection status.

    Attributes:
        n_features: Total number of features.
        feature_names: List of feature names.
        device: PyTorch device.

    Example:
        >>> extractor = FeatureExtractor(env, device="cuda")
        >>> features = extractor.observe(env)  # shape: (n_features, n_cells)
        >>> actions = policy(features)
    """

    # Feature constants
    CLIP_RANGE: float = 5.0
    DEFAULT_TIME_RESCALE: float = 10.0
    DEFAULT_FEATURE_SET: tuple[str, ...] = (
        "time",
        "disturbance",
        "disturbance_conv",
        "species_richness",
        "total_population",
        "current_ext_risk",
        "cost",
        "protection_matrix",
        "protection_matrix_conv",
    )
    # Optional dynamic features (add them to feature_set):
    #   "edge_exposure": env.edge_exposure(), the edge exposure each cell has or would
    #       have if protected now (reduces K by sensitivity * edge_factor * exposure)
    #   "buffer_value": env.edge_buffer(cell_values), the value gained by protected
    #       neighbours whose edge exposure drops if the cell is protected
    #   "pa_linking": connectivity.linking_value(protection), importance of the cell for
    #       linking protected-area units that are not yet connected (PAConnectivity)
    #   "worst_species_value": share of the currently worst-served species' populations
    #       (worst_fraction, as in CalcRewardWorstSpecies) held by each cell, scaled to [0, 1]
    OPTIONAL_FEATURES: tuple[str, ...] = (
        "edge_exposure", "buffer_value", "pa_linking", "worst_species_value"
    )

    def __init__(
            self,
            env: BioEnv,
            feature_set: list[str] | None = None,
            trait_features: list[str] | None = None,  # <- not fully implemented
            static_features: np.ndarray | torch.Tensor | None = None,
            static_features_names: list[str] | None = None,
            convolution: int = 5,
            keys_to_reset: list[str] | None = None,
            time_rescale: float = DEFAULT_TIME_RESCALE,
            device: torch.device | str | None = None,
            cell_values: np.ndarray | torch.Tensor | None = None,
            connectivity: PAConnectivity | None = None,
            worst_fraction: float = 0.1,
            worst_target: float | None = None,
            worst_threat_weights: np.ndarray | torch.Tensor | list | None = None,
    ):
        """Initialize feature extractor.

        Args:
            env: Environment to extract features from.
            feature_set: List of features to extract (default: all).
            convolution: Neighborhood radius for convolution features.
            keys_to_reset: Features that should not be z-score normalized.
            time_rescale: Divisor for time normalization.
            device: PyTorch device (default: same as env).
            cell_values: Per-cell value of the full carrying capacity, shape
                (n_cells,), required by the "buffer_value" feature.
            connectivity: PAConnectivity, required by the "pa_linking" feature.
            worst_fraction: Share of species counted as worst-served by the
                "worst_species_value" feature (match CalcRewardWorstSpecies).
            worst_target: Protected fraction at which a species counts as fully
                protected for that feature (match CalcRewardWorstSpecies).
            worst_threat_weights: IUCN weights of the worst-served species in that
                feature (match CalcRewardWorstSpecies); None = unweighted.
        """
        self.device = torch.device(device) if device else env.device

        # Feature configuration
        if feature_set is None:
            self._feature_set = self.default_feature_set
        else:
            self._feature_set = list(feature_set)

        # Build feature name list (expand current_ext_risk to per-class features)
        self._feature_names = list(self._feature_set)

        if "current_ext_risk" in self._feature_set:
            risk_names = [f"ext_risk_{i}" for i in range(env.ext_risk._n_classes)]
            idx = self._feature_set.index("current_ext_risk")
            self._feature_names[idx: idx + 1] = risk_names
            self._n_risk_classes = env.ext_risk._n_classes
        else:
            self._n_risk_classes = 0

        if trait_features is not None:
            self._feature_set = self._feature_set + trait_features
            self._feature_names = self._feature_names + trait_features
            self._use_trait_features = True
        else:
            self._use_trait_features = False

        if isinstance(static_features, (np.ndarray, torch.Tensor)):
            self._static_features = torch.as_tensor(static_features).to(self.device)
            if static_features_names is None:
                self._feature_names = self._feature_names + [
                    f"static_{k}" for k in range(self._static_features.size(0))
                ]
            else:
                self._feature_names = self._feature_names + static_features_names
        else:
            self._static_features = None

        if "buffer_value" in self._feature_set and cell_values is None:
            raise ValueError("The 'buffer_value' feature requires cell_values")
        if "pa_linking" in self._feature_set and connectivity is None:
            raise ValueError("The 'pa_linking' feature requires connectivity")
        self._connectivity = connectivity
        self._worst_fraction = worst_fraction
        self._worst_target = worst_target
        self._worst_threat_weights = (
            None if worst_threat_weights is None
            else torch.as_tensor(worst_threat_weights, dtype=torch.float32)
        )
        self._cell_values = (
            None
            if cell_values is None
            else torch.as_tensor(cell_values, dtype=torch.float32).to(self.device)
        )

        # Feature map for quick lookup
        self.feature_map = {name: i for i, name in enumerate(self._feature_names)}
        self._n_features = len(self._feature_names)

        # Convolution matrix for neighborhood features
        self.convolution = convolution
        scipy_conv = grid_utils.compute_convolution_matrix(
            env.disturbance._coords, radius=int(self.convolution - 1 / 2)
        )
        self._conv_matrix = scipy_sparse_to_torch(scipy_conv, self.device)

        # Feature buffer
        self.features = torch.zeros(
            (self._n_features, env.n_cells),
            device=self.device,
            dtype=torch.float32,
        )

        # Rescaling configuration
        if keys_to_reset is None:
            # these features are not rescaled by mean/std -> set_rescaler()
            self.keys_to_reset = [
                "time",
                "disturbance",
                "disturbance_conv",
                "protection_matrix",
                "protection_matrix_conv",
                "edge_exposure",
                "buffer_value",
                "pa_linking",
                "worst_species_value",
            ]
        else:
            self.keys_to_reset = list(keys_to_reset)

        self._features_to_reset = [
            self.feature_map[k] for k in self.keys_to_reset if k in self.feature_map
        ]

        # time is rescaled differently
        self._time_rescale = time_rescale

        # Initialize rescaler
        self.set_rescaler(env)

    def reset_static_features(
            self,
            static_features: np.ndarray | torch.Tensor,
            indices: int | list[int] | np.ndarray | torch.Tensor | None = None,
    ) -> None:
        """
        Resets all or a subset of static features.

        Args:
            static_features: The new feature values (array or tensor).
            indices: A single index or collection of indices pointing to the channels
                     to update. If None, the entire feature matrix is reset.
        """
        # 1. Convert incoming features to a PyTorch tensor on the correct device/dtype
        new_features = torch.as_tensor(
            static_features, device=self.device, dtype=self._static_features.dtype
        )

        # Case A: Full Reset (No indices provided)
        if indices is None:
            if self._static_features.shape != new_features.shape:
                raise ValueError(
                    f"Shape mismatch for full reset. "
                    f"Expected {self._static_features.shape}, got {new_features.shape}."
                )
            # Using .copy_() performs an in-place update. This is much safer than
            # reassignment because it preserves memory references if other parts of
            # your code are pointing to self._static_features.
            self._static_features.copy_(new_features)
            return

        # Case B: Single Index (e.g., resetting just channel 0)
        if isinstance(indices, (int, np.integer)):
            target_shape = self._static_features[indices].shape

            # If they passed a flat 1D array for a single slice, try to automatically
            # reshape/unsqueeze it to match the target slice shape.
            if new_features.shape != target_shape:
                try:
                    new_features = new_features.view(target_shape)
                except RuntimeError:
                    raise ValueError(
                        f"Cannot cast incoming shape {new_features.shape} to match "
                        f"the target channel slice shape {target_shape}."
                    )
            self._static_features[indices] = new_features

        # Case C: Multiple Indices (e.g., resetting channels [0, 2, 5])
        else:
            # Convert indices cleanly to a PyTorch LongTensor on the correct device
            idx_tensor = torch.as_tensor(indices, dtype=torch.long, device=self.device)
            target_shape = self._static_features[idx_tensor].shape

            if new_features.shape != target_shape:
                raise ValueError(
                    f"Shape mismatch for indices {indices}. "
                    f"Expected shape {target_shape}, got {new_features.shape}."
                )
            self._static_features[idx_tensor] = new_features

    @property
    def n_features(self) -> int:
        """Number of features."""
        return self._n_features

    @property
    def feature_names(self) -> list[str]:
        """Feature names, in the order of the feature dimension."""
        return list(self._feature_names)

    @property
    def default_feature_set(self):
        return list(self.DEFAULT_FEATURE_SET)

    def extract_features(self, env: BioEnv) -> torch.Tensor:
        """Extract raw features from environment.

        Args:
            env: Environment to extract features from.

        Returns:
            Feature tensor of shape (n_features, n_cells).
        """
        obs = self.features

        if "time" in self._feature_set:
            idx = self.feature_map["time"]
            obs[idx].fill_(env._env_step_num / self._time_rescale)

        if "disturbance" in self._feature_set:
            idx = self.feature_map["disturbance"]
            obs[idx] = env.disturbance.data.mean(dim=0)

        if "disturbance_conv" in self._feature_set:
            idx = self.feature_map["disturbance_conv"]
            disturbance_mean = env.disturbance.data.mean(dim=0)
            # Sparse matrix multiplication: (n_cells,) @ sparse(n_cells, n_cells)
            # Note: For MPS, sparse matrix is on CPU
            sparse_device = self._conv_matrix.device
            if sparse_device != self.device:
                # MPS path: transfer to CPU for sparse op, then back
                disturbance_cpu = disturbance_mean.to(sparse_device)
                result = torch.sparse.mm(
                    self._conv_matrix, disturbance_cpu.unsqueeze(1)
                ).squeeze(1)
                obs[idx] = result.to(self.device)
            else:
                obs[idx] = torch.sparse.mm(
                    self._conv_matrix, disturbance_mean.unsqueeze(1)
                ).squeeze(1)

        if "species_richness" in self._feature_set:
            idx = self.feature_map["species_richness"]
            obs[idx] = (env.h > 1).sum(dim=0).float()

        if "total_population" in self._feature_set:
            idx = self.feature_map["total_population"]
            pop_sum = env.h.sum(dim=0)
            obs[idx] = torch.log1p(pop_sum)

        if "protection_matrix" in self._feature_set:
            idx = self.feature_map["protection_matrix"]
            obs[idx] = env.protected_cells_mask.float()

        if "protection_matrix_conv" in self._feature_set:
            idx = self.feature_map["protection_matrix_conv"]
            protection_m = env.protected_cells_mask.float()
            # Sparse matrix multiplication: (n_cells,) @ sparse(n_cells, n_cells)
            # Note: For MPS, sparse matrix is on CPU
            sparse_device = self._conv_matrix.device
            if sparse_device != self.device:
                # MPS path: transfer to CPU for sparse op, then back
                disturbance_cpu = protection_m.to(sparse_device)
                result = torch.sparse.mm(
                    self._conv_matrix, disturbance_cpu.unsqueeze(1)
                ).squeeze(1)
                obs[idx] = result.to(self.device)
            else:
                obs[idx] = torch.sparse.mm(
                    self._conv_matrix, protection_m.unsqueeze(1)
                ).squeeze(1)

        if "current_ext_risk" in self._feature_set:
            start = self.feature_map["ext_risk_0"]
            end = start + self._n_risk_classes
            risk_view = obs[start:end]
            risk_view.zero_()

            # One-hot encoding of risk per cell weighted by species presence
            current_risk = env.current_ext_risk  # (n_species,)
            presence = (env.h > 1).float()  # (n_species, n_cells)

            # Scatter add: for each species, add its presence to its risk class row
            for risk_class in range(self._n_risk_classes):
                mask = current_risk == risk_class
                if mask.any():
                    risk_view[risk_class] = presence[mask].sum(dim=0)

        if "cost" in self._feature_set:
            idx = self.feature_map["cost"]
            obs[idx] = env.costs.data.mean(dim=0)

        if "edge_exposure" in self._feature_set:
            obs[self.feature_map["edge_exposure"]] = env.edge_exposure()

        if "buffer_value" in self._feature_set:
            obs[self.feature_map["buffer_value"]] = env.edge_buffer(self._cell_values)

        if "pa_linking" in self._feature_set:
            obs[self.feature_map["pa_linking"]] = self._connectivity.linking_value(
                env.protected_cells_mask
            ).to(self.device)

        if "worst_species_value" in self._feature_set:
            worst = worst_species(protected_fractions(env, self._worst_target), self._worst_fraction)
            init_pop = env.ext_risk.init_pop.to(env.device)[worst].clamp(min=1e-12)
            k_full = env._species_k[worst, None] * env.sdms.data_min_threshold[worst]
            share = k_full / init_pop[:, None]
            if self._worst_threat_weights is not None:
                share = share * tail_threat_weights(env, worst, self._worst_threat_weights)[:, None]
            value = share.sum(dim=0)
            obs[self.feature_map["worst_species_value"]] = (
                value / value.max().clamp(min=1e-12)
            ).to(self.device)

        if self._static_features is not None:
            # add static features
            obs[-self._static_features.size(0):] = self._static_features

        return obs

    def observe(self, env: BioEnv) -> torch.Tensor:
        """Extract and normalize features from environment.

        Args:
            env: Environment to observe.

        Returns:
            Normalized feature tensor of shape (n_features, n_cells).
        """
        raw_obs = self.extract_features(env)
        return self.transform(raw_obs)

    def transform(self, obs: torch.Tensor) -> torch.Tensor:
        """Apply z-score normalization with outlier clipping.

        Args:
            obs: Raw observation tensor.

        Returns:
            Normalized tensor clipped to [-CLIP_RANGE, CLIP_RANGE].
        """
        z = (obs - self._rescaler_mean) / self._rescaler_std
        return torch.clamp(z, -self.CLIP_RANGE, self.CLIP_RANGE)

    def set_rescaler(self, env: BioEnv) -> None:
        """Compute normalization statistics from current environment state.

        Args:
            env: Environment to compute statistics from.
        """
        obs = self.extract_features(env)

        # Compute mean and std per feature
        self._rescaler_std = torch.std(obs, dim=1, keepdim=True)
        self._rescaler_std = torch.maximum(
            self._rescaler_std,
            torch.ones_like(self._rescaler_std),
        )
        self._rescaler_mean = torch.mean(obs, dim=1, keepdim=True)

        # Don't rescale certain features
        self._rescaler_std[self._features_to_reset] = 1.0
        self._rescaler_mean[self._features_to_reset] = 0.0

    def plot_features(
            self,
            env: BioEnv,
            rescale: bool = True,
            outdir: str | Path | None = None,
            figsize=(5, 6),
    ) -> None:
        """Plot all features as spatial grids.

        Args:
            env: Environment to visualize.
            rescale: If True, show normalized features.
        """
        from captain.utils import plots

        obs = self.observe(env) if rescale else self.extract_features(env)

        obs_np = obs.cpu().numpy()
        feature_3d = grid_utils.reconstruct_grid(
            obs_np, env.disturbance._coords, env.disturbance._data_shape[1:]
        )
        for i in range(self._n_features):
            plots.plot_grid(
                feature_3d[i],
                title=self._feature_names[i],
                outfile=os.path.join(str(outdir), f"f_{self._feature_names[i]}.png"),
                figsize=figsize,
            )

    def to(self, device: torch.device | str) -> FeatureExtractor:
        """Move extractor to specified device.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        self.device = torch.device(device)
        self.features = self.features.to(self.device)
        self._conv_matrix = self._conv_matrix.to(self.device)
        self._rescaler_std = self._rescaler_std.to(self.device)
        self._rescaler_mean = self._rescaler_mean.to(self.device)
        if self._cell_values is not None:
            self._cell_values = self._cell_values.to(self.device)
        return self
