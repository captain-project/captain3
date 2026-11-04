"""Least-cost connectivity among protected-area units.

Used by CalcRewardConnectivity (reward) and the "pa_linking" feature of
FeatureExtractor, so that both see the same connectivity measure.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

# Tiny cost for crossing protected cells, so that zero-cost moves stay edges of the graph
_EPS = 1e-9


class PAConnectivity:
    """Equivalent Connected Area (ECA) of protected-area units under least-cost distances.

    The landscape is a graph of valid cells with 8-neighbour moves. Entering a cell costs
    its resistance if it is unprotected and nothing if it is protected, so the gap
    between two units is the least total resistance of the unprotected cells needed to
    link them (0 once they are connected through protected cells). With unit weights
    w_i = area_i ** area_exponent:

        ECA = sqrt(sum_ij w_i w_j exp(-gap_ij / alpha))

    which ranges from ECA_0 (initial protection) to sum_i w_i (all units connected).

    Args:
        coords: (rows, cols) grid indices of the valid cells, e.g. SpatialData._coords.
        units: Unit id per cell, shape (n_cells,): 0 = not part of a unit, k > 0 = unit k
            (e.g. connected components of the existing protected areas).
        resistance: Cost of crossing each unprotected cell, shape (n_cells,), > 0.
        initial_protection: Protected cells defining the baseline ECA_0, shape (n_cells,).
        alpha: Distance decay of connection probability, in resistance units.
        area_exponent: 1 = standard ECA (large units dominate), 0 = all units equal.
        beta: Tolerance (resistance units) for cells near, but not on, the best link in
            linking_value().
        feature_refresh_every: linking_value() recomputes distances only every n-th call
            and otherwise reuses the last result (newly protected cells set to 0).
        device: Device of the returned tensors.
    """

    def __init__(
            self,
            coords: tuple[np.ndarray, np.ndarray],
            units: np.ndarray | torch.Tensor,
            resistance: np.ndarray | torch.Tensor,
            initial_protection: np.ndarray | torch.Tensor,
            alpha: float,
            area_exponent: float = 1.0,
            beta: float = 1.0,
            feature_refresh_every: int = 1,
            device: torch.device | str = "cpu",
    ):
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be > 0")
        if feature_refresh_every < 1:
            raise ValueError("feature_refresh_every must be >= 1")
        self.device = torch.device(device)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.feature_refresh_every = int(feature_refresh_every)

        units = np.asarray(torch.as_tensor(units).cpu(), dtype=int)
        self._unit_ids = np.unique(units[units > 0])
        if len(self._unit_ids) < 2:
            raise ValueError("At least two protected-area units are needed")
        self._unit_cells = [np.flatnonzero(units == k) for k in self._unit_ids]
        area = np.array([len(c) for c in self._unit_cells], dtype=float)
        self._w = area**area_exponent
        self.n_units = len(self._unit_ids)

        self._resistance = np.asarray(torch.as_tensor(resistance).cpu(), dtype=float)
        if np.any(self._resistance <= 0):
            raise ValueError("resistance must be > 0")
        self._build_graph(coords)

        self._cache_key: np.ndarray | None = None
        self._dist = self._gaps = None
        self._n_feature_calls = 0
        self._feature_cache: np.ndarray | None = None
        self._feature_protection: np.ndarray | None = None

        self._eca_max = self._w.sum()
        self._eca_0 = self.eca(initial_protection)

    def _build_graph(self, coords: tuple[np.ndarray, np.ndarray]) -> None:
        rows, cols = (np.asarray(c, dtype=int) for c in coords)
        grid = -np.ones((rows.max() + 2, cols.max() + 2), dtype=int)
        grid[rows, cols] = np.arange(len(rows))
        src, dst = [], []
        for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            r2, c2 = rows + dr, cols + dc
            ok = (r2 >= 0) & (c2 >= 0)
            nb = grid[r2[ok], c2[ok]]
            keep = nb >= 0
            src.append(np.flatnonzero(ok)[keep])
            dst.append(nb[keep])
        src, dst = np.concatenate(src), np.concatenate(dst)
        n = len(rows)
        # directed edges; weight of an edge = cost of entering its destination cell
        self._graph = sparse.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
        self._graph.sort_indices()
        self._edge_dst = self._graph.indices.copy()
        self.n_cells = n

    def _node_cost(self, protected: np.ndarray) -> np.ndarray:
        return np.where(protected, _EPS, self._resistance)

    def _update(self, protected: np.ndarray) -> None:
        """Least-cost distances from every unit and pairwise gaps (cached per protection)."""
        if self._cache_key is not None and np.array_equal(protected, self._cache_key):
            return
        cost = self._node_cost(protected)
        self._graph.data = cost[self._edge_dst]
        dist = np.stack(
            [dijkstra(self._graph, indices=cells, min_only=True) for cells in self._unit_cells]
        )
        gaps = np.stack([dist[:, cells].min(axis=1) for cells in self._unit_cells], axis=1)
        gaps = np.minimum(gaps, gaps.T)
        gaps[gaps < 1e-6] = 0.0
        self._dist, self._gaps, self._cache_key = dist, gaps, protected.copy()

    @staticmethod
    def _as_bool(protected: np.ndarray | torch.Tensor) -> np.ndarray:
        return np.asarray(torch.as_tensor(protected).cpu()).reshape(-1) > 0

    def gaps(self, protected: np.ndarray | torch.Tensor) -> np.ndarray:
        """Pairwise least-cost gaps between units, shape (n_units, n_units)."""
        self._update(self._as_bool(protected))
        return self._gaps.copy()

    def eca(self, protected: np.ndarray | torch.Tensor) -> float:
        """Equivalent Connected Area under the given protection."""
        self._update(self._as_bool(protected))
        p = np.exp(-self._gaps / self.alpha)
        return float(np.sqrt(self._w @ p @ self._w))

    def eca_gain(self, protected: np.ndarray | torch.Tensor) -> float:
        """ECA gain scaled to [0, 1]: 0 = initial protection, 1 = all units connected."""
        denom = self._eca_max - self._eca_0
        if denom <= 0:
            return 1.0
        return (self.eca(protected) - self._eca_0) / denom

    def linking_value(self, protected: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Importance of each cell for linking units that are not yet connected.

        For each unconnected pair (i, j), cells on the least-cost link get weight
        w_i w_j exp(-gap_ij / alpha), and cells off it exp(-excess / beta) of that, where
        excess is how much longer the best link through the cell is. Summed over pairs,
        scaled to [0, 1] by the maximum, 0 for protected cells and when all units are
        connected. Shape (n_cells,).
        """
        prot = self._as_bool(protected)
        # protection only grows within an episode: cells unprotected again = new episode
        reset = self._feature_protection is not None and bool(
            np.any(self._feature_protection & ~prot)
        )
        if self._feature_cache is None or reset:
            self._n_feature_calls = 0
        recompute = self._n_feature_calls % self.feature_refresh_every == 0
        self._n_feature_calls += 1
        if recompute:
            self._update(prot)
            cost = self._node_cost(prot)
            value = np.zeros(self.n_cells)
            for i in range(self.n_units):
                for j in range(i + 1, self.n_units):
                    gap = self._gaps[i, j]
                    if gap <= 0 or not np.isfinite(gap):
                        continue
                    through = self._dist[i] + self._dist[j] - cost  # best link via each cell
                    excess = np.clip(through - gap, 0, None)
                    weight = self._w[i] * self._w[j] * np.exp(-gap / self.alpha)
                    value += weight * np.exp(-excess / self.beta)
            self._feature_cache = value / value.max() if value.max() > 0 else value
            self._feature_protection = prot.copy()
        value = np.where(prot, 0.0, self._feature_cache)
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def reset_feature_cache(self) -> None:
        """Force linking_value() to recompute at its next call."""
        self._feature_cache = None
        self._feature_protection = None
        self._n_feature_calls = 0
