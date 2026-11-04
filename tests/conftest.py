"""Shared pytest fixtures for CAPTAIN tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture(scope="session")
def device():
    """Return available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def rng():
    """Return a seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def torch_rng():
    """Return a seeded PyTorch generator for reproducible tests."""
    gen = torch.Generator()
    gen.manual_seed(42)
    return gen


@pytest.fixture
def small_grid_params(rng):
    """Parameters for a small test grid (5 species, 10x10)."""
    n_species = 5
    grid_x, grid_y = 10, 10

    # Create grid with some masked cells
    mask = np.ones((grid_x, grid_y))
    mask[:2, :2] = np.nan  # Mask corner

    return {
        "n_species": n_species,
        "grid_shape": (grid_x, grid_y),
        "mask": mask,
        "sdms": rng.random((n_species, grid_x, grid_y)),
        "disturbance": rng.random((1, grid_x, grid_y)),
        "costs": rng.random((1, grid_x, grid_y)),
        "growth_rates": rng.uniform(1.0, 1.5, size=n_species),
        "dispersal_rate": 0.5,
        "species_k": rng.uniform(50, 100, size=n_species),
        "sensitivity": rng.random((n_species, 1)),
        "init_status": rng.integers(0, 5, size=n_species),
    }


@pytest.fixture
def medium_grid_params(rng):
    """Parameters for a medium test grid (50 species, 50x50)."""
    n_species = 50
    grid_x, grid_y = 50, 50

    mask = np.ones((grid_x, grid_y))
    mask[:5, :5] = np.nan

    return {
        "n_species": n_species,
        "grid_shape": (grid_x, grid_y),
        "mask": mask,
        "sdms": rng.random((n_species, grid_x, grid_y)),
        "disturbance": rng.random((1, grid_x, grid_y)),
        "costs": rng.random((1, grid_x, grid_y)),
        "growth_rates": rng.uniform(1.0, 1.5, size=n_species),
        "dispersal_rate": 0.3,
        "species_k": rng.uniform(50, 100, size=n_species),
        "sensitivity": rng.random((n_species, 1)),
        "init_status": rng.integers(0, 5, size=n_species),
    }


@pytest.fixture
def small_spatial_data(small_grid_params, rng):
    """Create a small SpatialData instance for testing."""
    from captain.data.spatial_data import SpatialData

    return SpatialData(
        data=small_grid_params["sdms"],
        mask=small_grid_params["mask"],
        delta_per_step=rng.normal(scale=0.01, size=small_grid_params["sdms"].shape),
        lower_bound=0.0,
        upper_bound=1.0,
    )


@pytest.fixture
def small_env(small_grid_params, rng):
    """Create a small BioEnv instance for testing."""
    from captain.data.extinction_risk import ExtinctionRisk
    from captain.data.spatial_data import SpatialData
    from captain.environment.bioenv import BioEnv

    p = small_grid_params
    mask = p["mask"]

    sdms = SpatialData(data=p["sdms"], mask=mask)
    disturbance = SpatialData(data=p["disturbance"], mask=mask)
    costs = SpatialData(data=p["costs"], mask=mask)
    protection = SpatialData(
        data=np.zeros_like(p["disturbance"]),
        mask=mask,
    )

    ext_risk = ExtinctionRisk(
        init_status=p["init_status"],
        n_classes=5,
    )

    return BioEnv(
        sdms=sdms,
        disturbance=disturbance,
        costs=costs,
        protection_matrix=protection,
        growth_rates=p["growth_rates"],
        dispersal_rates=p["dispersal_rate"],
        species_k=p["species_k"],
        sensitivity_rates=p["sensitivity"],
        ext_risk=ext_risk,
    )


@pytest.fixture
def small_feature_extractor(small_env):
    """Create a FeatureExtractor for the small environment."""
    from captain.agents.feature_extractor import FeatureExtractor

    return FeatureExtractor(small_env)


@pytest.fixture
def small_policy(small_feature_extractor):
    """Create a PolicyNetwork for the small environment."""
    from captain.agents.policy_network import CellNN, PolicyNetwork

    model = CellNN(
        input_dim=small_feature_extractor._n_features,
        hidden_dim=16,
    )
    return PolicyNetwork(model)


@pytest.fixture
def small_rewards():
    """Create a Rewards instance for testing."""
    from captain.agents.rewards import (
        CalcRewardExtRisk,
        CalcRewardPersistentCost,
        Rewards,
    )

    return Rewards([
        CalcRewardExtRisk(
            threat_weights=np.array([1, 0, -8, -16, -32]),
            rescaler=1.0,
        ),
        CalcRewardPersistentCost(rescaler=0.1),
    ])
