"""Agent components: policy network, feature extraction, and rewards."""

from __future__ import annotations

from captain.agents.feature_extractor import FeatureExtractor
from captain.agents.feature_extractor_cnn import FeatureExtractorCNN
from captain.agents.policy_network import CellCNNPolicy, CellNN, PolicyNetwork
from captain.agents.reward_aggregator import NoRewards, Rewards
from captain.agents.rewards import (
    CalcReward,
    CalcRewardConnectivity,
    CalcRewardDistFromTarget,
    CalcRewardExtRisk,
    CalcRewardLayerOverlap,
    CalcRewardPersistentCost,
    CalcRewardProtectedPopulation,
    CalcRewardWorstSpecies,
)
from captain.agents.spatial_cnn import SpatialCNN

__all__ = [
    "FeatureExtractor",
    "FeatureExtractorCNN",
    "CellNN",
    "CellCNNPolicy",
    "PolicyNetwork",
    "SpatialCNN",
    "CalcReward",
    "CalcRewardExtRisk",
    "CalcRewardPersistentCost",
    "CalcRewardDistFromTarget",
    "CalcRewardConnectivity",
    "CalcRewardProtectedPopulation",
    "CalcRewardWorstSpecies",
    "CalcRewardLayerOverlap",
    "Rewards",
    "NoRewards",
]
