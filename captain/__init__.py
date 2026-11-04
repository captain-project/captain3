"""CAPTAIN: Conservation Area Prioritization Through Artificial INtelligence."""

from __future__ import annotations

__version__ = "3.0.0rc1"

# Data structures
# Agents
from captain.agents.feature_extractor import FeatureExtractor
from captain.agents.feature_extractor_cnn import FeatureExtractorCNN
from captain.agents.policy_network import (
    CellCNNPolicy,
    CellNN,
    PolicyNetwork,
    RegionalPolicyNetwork,
)
from captain.agents.reward_aggregator import NoRewards, Rewards
from captain.agents.rewards import (
    CalcReward,
    CalcRewardConnectivity,
    CalcRewardExtRisk,
    CalcRewardLayerOverlap,
    CalcRewardPersistentCost,
    CalcRewardProtectedPopulation,
    CalcRewardSpeciesValue,
    CalcRewardSpecieValue,
    CalcRewardWorstSpecies,
)
from captain.agents.spatial_cnn import SpatialCNN
from captain.algorithms import TrainingLogger
from captain.algorithms.budget_manager import (
    GlobalBudgetManager,
    NoBudgetManager,
    RegionalBudgetManager,
)

# Algorithms
from captain.algorithms.episode import EpisodeRunner
from captain.algorithms.evolution_train import EvolStrategiesTrainer
from captain.algorithms.inference_recorder import InferenceRecorder
from captain.algorithms.scheduler import FactorSchedule, LearningScheduler, RewardWeightSchedule
from captain.data.extinction_risk import ExtinctionRisk, ExtinctionRiskStatic
from captain.data.spatial_data import (
    SpatialData,
    StochasticSpatialData,
    load_spatial_data,
    load_spatial_data_from_dir,
)

# Environment
from captain.environment.bioenv import BioEnv

# Utilities
from captain.utils import data_loader, export, grid_utils, plots
from captain.utils.connectivity import PAConnectivity

__all__ = [
    # Version
    "__version__",
    # Data
    "SpatialData",
    "StochasticSpatialData",
    "ExtinctionRisk",
    "ExtinctionRiskStatic",
    "load_spatial_data",
    "load_spatial_data_from_dir",
    # Environment
    "BioEnv",
    # Agents
    "FeatureExtractor",
    "FeatureExtractorCNN",
    "CellNN",
    "CellCNNPolicy",
    "PolicyNetwork",
    "RegionalPolicyNetwork",
    "SpatialCNN",
    "CalcReward",
    "CalcRewardExtRisk",
    "CalcRewardPersistentCost",
    "CalcRewardSpecieValue",
    "CalcRewardSpeciesValue",
    "CalcRewardConnectivity",
    "CalcRewardProtectedPopulation",
    "CalcRewardWorstSpecies",
    "CalcRewardLayerOverlap",
    "Rewards",
    "NoRewards",
    # Algorithms
    "EpisodeRunner",
    "InferenceRecorder",
    "EvolStrategiesTrainer",
    "LearningScheduler",
    "FactorSchedule",
    "RewardWeightSchedule",
    "TrainingLogger",
    "GlobalBudgetManager",
    "RegionalBudgetManager",
    "NoBudgetManager",
    # Utilities
    "grid_utils",
    "PAConnectivity",
    "plots",
    "data_loader",
    "export",
]
