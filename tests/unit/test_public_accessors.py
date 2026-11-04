"""Public accessors used by the examples instead of private attributes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from captain.agents.reward_aggregator import Rewards
from captain.agents.rewards import (
    CalcRewardConnectivity,
    CalcRewardPersistentCost,
    CalcRewardProtectedPopulation,
)
from captain.data.extinction_risk import ExtinctionRisk
from captain.data.spatial_data import SpatialData
from captain.environment.bioenv import BioEnv
from captain.utils.connectivity import PAConnectivity


class TestSpatialDataAccessors:
    def test_coords_match_valid_cells(self, small_spatial_data, small_grid_params):
        rows, cols = small_spatial_data.coords
        mask = small_grid_params["mask"]
        assert len(rows) == small_spatial_data.data.shape[1]
        assert np.all(~np.isnan(mask[rows, cols]))

    def test_nonzero_cells_mask_follows_updates(self, small_grid_params):
        layer = SpatialData(
            data=np.zeros((1, *small_grid_params["grid_shape"])), mask=small_grid_params["mask"]
        )
        assert not layer.nonzero_cells_mask.any()
        layer.update_cell_values([0, 3], 1.0)
        assert layer.nonzero_cells_mask.nonzero().flatten().tolist() == [0, 3]

    def test_to_grid_matches_reconstruct_grid(self, small_spatial_data):
        values = small_spatial_data.data
        np.testing.assert_allclose(
            small_spatial_data.to_grid(values), small_spatial_data.reconstruct_grid, equal_nan=True
        )
        # 1-D input gives a 2-D grid, NaN outside the valid cells
        grid = small_spatial_data.to_grid(values[0])
        assert grid.shape == small_spatial_data.shape[1:]
        np.testing.assert_allclose(
            grid, small_spatial_data.reconstruct_grid[0], equal_nan=True
        )


class TestBioEnvAccessors:
    def test_species_parameters(self, small_env, small_grid_params):
        np.testing.assert_allclose(small_env.species_k.numpy(), small_grid_params["species_k"],
                                   rtol=1e-6)
        np.testing.assert_allclose(small_env.growth_rates.numpy(),
                                   small_grid_params["growth_rates"], rtol=1e-6)
        assert small_env.sensitivity_rates.shape == (small_env.n_species, 1)
        assert small_env.carrying_capacity.shape == (small_env.n_species, small_env.n_cells)
        assert small_env.edge_factor == 0.0

    def test_step_counters(self, small_env):
        small_env.reset()
        assert small_env.step_num == 0 and small_env.env_step_num == 0
        small_env.step()
        small_env.step(env_step=False)
        assert small_env.step_num == 2 and small_env.env_step_num == 1

    def test_traits(self, small_env, small_grid_params):
        assert small_env.species_traits is None
        with pytest.raises(KeyError):
            small_env.get_trait("biomass")

        p = small_grid_params
        mask = p["mask"]
        traits = pd.DataFrame({
            "species": [f"sp{i}" for i in range(p["n_species"])],
            "biomass": np.arange(p["n_species"], dtype=float),
        })
        env = BioEnv(
            sdms=SpatialData(data=p["sdms"], mask=mask),
            disturbance=SpatialData(data=p["disturbance"], mask=mask),
            costs=SpatialData(data=p["costs"], mask=mask),
            protection_matrix=SpatialData(data=np.zeros_like(p["disturbance"]), mask=mask),
            growth_rates=p["growth_rates"],
            species_k=p["species_k"],
            sensitivity_rates=p["sensitivity"],
            ext_risk=ExtinctionRisk(init_status=p["init_status"], n_classes=5),
            species_traits=traits,
        )
        assert env.species_traits is traits
        expected = torch.arange(p["n_species"], dtype=env.get_trait("biomass").dtype)
        assert torch.equal(env.get_trait("biomass"), expected)
        assert torch.equal(env.get_trait(env.trait_map["biomass"]), expected)
        with pytest.raises(KeyError):
            env.get_trait("species")  # not numeric


class TestExtinctionRiskAccessors:
    def test_classes_and_initial_counts(self):
        risk = ExtinctionRisk(init_status=np.array([0, 0, 2, 4]), n_classes=5)
        assert risk.n_classes == 5
        assert risk.init_status_counts.tolist() == [2, 0, 1, 0, 1]


class TestRewardsAccessors:
    def test_terms_and_get_term(self):
        cost = CalcRewardPersistentCost()
        protpop = CalcRewardProtectedPopulation()
        rewards = Rewards([protpop, cost])
        assert rewards.terms == [protpop, cost]
        assert rewards.get_term("cost") is cost
        with pytest.raises(KeyError):
            rewards.get_term("connectivity")

    def test_connectivity_property(self, small_env):
        coords = small_env.sdms.coords
        units = np.zeros(small_env.n_cells, dtype=int)
        units[[0, small_env.n_cells - 1]] = [1, 2]
        conn = PAConnectivity(
            coords, units, np.ones(small_env.n_cells), initial_protection=units > 0, alpha=5.0
        )
        assert CalcRewardConnectivity(conn).connectivity is conn


class TestApiChanges:
    def test_species_value_alias_warns(self):
        from captain.agents.rewards import CalcRewardSpecieValue, CalcRewardSpeciesValue

        with pytest.warns(DeprecationWarning):
            old = CalcRewardSpecieValue(trait_name="biomass")
        assert isinstance(old, CalcRewardSpeciesValue)
        assert old.name == "biomass"

    def test_trait_features_not_implemented(self, small_env):
        from captain.agents.feature_extractor import FeatureExtractor

        with pytest.raises(NotImplementedError):
            FeatureExtractor(small_env, trait_features=["biomass"])
