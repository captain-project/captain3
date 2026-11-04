# CAPTAIN v.3.0 – Release Candidate

**Conservation Area Prioritization Through Artificial INtelligence**

[![Status](https://img.shields.io/badge/status-release%20candidate%203.0.0rc1-blue)](https://github.com/captain-project/captain3/blob/main/CHANGELOG.md)
[![CI](https://github.com/captain-project/captain3/actions/workflows/ci.yml/badge.svg)](https://github.com/captain-project/captain3/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)


## What is CAPTAIN?

CAPTAIN is a reinforcement learning system for optimizing conservation and restoration strategies in space and time.

### Key Features in v.3
⚙️ **Engineered for Scale:** Complete rewrite for high performance. Full GPU support and optimized sparse matrix operations allow for analysis at larger scales and finer resolutions.

🌱 **Dynamic Environment Scenarios:** Advanced support for time-evolving scenarios, incorporating climate change projections and dynamic implementation costs.  

🧩 **Modular & Customizable:** A flexible architecture allowing for highly tailored conservation policies and seamless integration of custom spatial data.

🎯 **Multi-objective Optimization:** Full support to quantify synergies and trade-offs between competing conservation and restoration targets.

🤖 **Decentralized Regional Agents:** Features regional agents designed to optimize spatially-distributed conservation strategies.

🧱 **Edge Effects:** Disturbance can spill over from unprotected land into the borders of protected areas, so that compact reserves support larger populations than scattered cells.

🔗 **Connectivity:** Reward and policy features that link protected areas, based on the equivalent connected area of the reserve network and least-cost paths between reserves.

⚖️ **Species-level Rewards:** Reward the protected share of each species' population (weighted by threat status and range size), the worst-served species, or the overlap with any spatial layer, and change the reward weights during training with weight schedules. See the [rewards tutorial](https://github.com/captain-project/captain3/blob/main/tutorials/rewards/README.md).

📍 **Static Prioritization:** One-shot protection plans within a single time step, without population dynamics, for fast spatial prioritization.

🌍 **Google Earth Export:** Export protection plans as KMZ polygons to view them on satellite imagery in Google Earth or QGIS.

📘 **Hands-on Tutorials:** A step-by-step [toy-example walkthrough](https://github.com/captain-project/captain3/blob/main/tutorials/toy_example/README.md) covering data setup, reward calibration, training, and global vs. regional agents on a small toy dataset, and an [advanced example](https://github.com/captain-project/captain3/blob/main/tutorials/advanced_example/README.md) with edge effects, existing protected areas, connectivity and static prioritization. The [training guide](https://github.com/captain-project/captain3/blob/main/docs/training.md) explains the optimizer and its settings.

### What's new in 3.0.0rc1

Since the August preview: edge effects, connectivity, new reward terms and reward weight schedules, static single-step episodes and KMZ export (see above), with new example scripts and tutorials, a [quick start](https://github.com/captain-project/captain3/blob/main/docs/quickstart.md), a test suite and continuous integration.

Changes that affect existing code and runs:
- The evolution strategies trainer now standardizes the rewards of each epoch's perturbations before the update (`advantage_normalization="zscore"`), so the step size no longer grows with the reward scale. Pass `advantage_normalization=None` to `EvolStrategiesTrainer` for the previous behaviour.
- Budget managers round the number of cells protected per decision up, so the full protection target is reached.
- `CalcRewardSpecieValue` is now `CalcRewardSpeciesValue` (the old name still works, with a deprecation warning).
- The package is installed as `captain3`; you still `import captain`.

See the [changelog](https://github.com/captain-project/captain3/blob/main/CHANGELOG.md) for the full list.


### ⚠️ Notice
**This repository contains release candidate 3.0.0rc1 of CAPTAIN v.3.**

* **Status:** The API is now stable for 3.0: until the 3.0.0 release, it will change only to fix bugs. The CNN feature extractor (`FeatureExtractorCNN`, `SpatialCNN`, `CellCNNPolicy`) is experimental.
* **Development:** Primary development occurs in a private repository; this public repository is updated with each release.
* **Support:** Please report bugs via GitHub Issues, but note that feature requests may be prioritized based on our internal development roadmap.


## Installation


#### 1. Install `uv` 
`uv` is a high-performance Python package manager that ensures consistent environments across Windows, macOS, and Linux.

* **macOS / Linux:**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
* **Windows (PowerShell):**

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

#### 2. Set up the Environment
Once `uv` is installed, clone the repository and sync the dependencies. This will automatically create a virtual environment (`.venv`) and install the correct version of PyTorch.

```bash
# Clone and enter the repository
git clone https://github.com/captain-project/captain3
cd captain3

# Sync dependencies and create virtual environment
uv sync

# Check the installation
uv run python -c "import captain; print(captain.__version__)"
```

To run the test suite, install the development dependencies first:

```bash
uv sync --extra dev
uv run pytest
```

To use CAPTAIN as a package in another project, install it from this repository:

```bash
uv add "captain3 @ git+https://github.com/captain-project/captain3@v3.0.0rc1"
```

#### 🛠️ Troubleshooting - Windows-Specific Setup


If you are on Windows, you may need to perform two quick steps to ensure uv and Python work correctly:

Enable Long Paths: Windows has a default 260-character limit for file paths. High-performance libraries like PyTorch often exceed this.

Run PowerShell as Administrator and execute:

```PowerShell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Execution Policy: If PowerShell blocks the uv command, run:

```PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Examples

Working example scripts are in the [examples folder](https://github.com/captain-project/captain3/blob/main/examples)
- [run_episode.py](https://github.com/captain-project/captain3/blob/main/examples/run_episode.py) - Run one episode on synthetic data (no download needed)
- [plot\_input\_data.py](https://github.com/captain-project/captain3/blob/main/examples/plot_input_data.py) - Visualize spatial and time-varying input data
- [calibrate_rewards.py](https://github.com/captain-project/captain3/blob/main/examples/calibrate_rewards.py) - Calibrate the scales of the reward terms (run once before training)
- [train_policy.py](https://github.com/captain-project/captain3/blob/main/examples/train_policy.py) - Full training loop with real data
- [run_inference.py](https://github.com/captain-project/captain3/blob/main/examples/run_inference.py) - Load a trained model and perform optimization, plot the results
- [train_policy_advanced.py](https://github.com/captain-project/captain3/blob/main/examples/train_policy_advanced.py) - Edge effects, existing protected areas, connectivity and a worst-served species reward
- [static_prioritization.py](https://github.com/captain-project/captain3/blob/main/examples/static_prioritization.py) - One-shot plan within a single time step
- [train_policy_weight_schedule.py](https://github.com/captain-project/captain3/blob/main/examples/train_policy_weight_schedule.py) - Anneal the weight of a reward term during training

Example data can be downloaded [here](https://polybox.ethz.ch/index.php/s/WKdbHHGj3ayL9w9). 
A pre-trained model is available [here](https://polybox.ethz.ch/index.php/s/wZ5AMXPdzboZSm2). 

For a short training script, see the [quick start](https://github.com/captain-project/captain3/blob/main/docs/quickstart.md).
For a quicker, self-contained walkthrough using a small toy dataset, see the
[toy-example tutorial](https://github.com/captain-project/captain3/blob/main/tutorials/toy_example/README.md) —
covers data setup, reward calibration, training, and running inference with
both a single global agent and coordinated regional agents. The
[advanced example](https://github.com/captain-project/captain3/blob/main/tutorials/advanced_example/README.md) and the
[rewards tutorial](https://github.com/captain-project/captain3/blob/main/tutorials/rewards/README.md) build on it.

<img width="50%" alt="animated_map" src="https://github.com/user-attachments/assets/9ef85156-9a76-4a32-8c29-d5960481a7cb" />

## Project Structure

```
captain3/
├── captain/            # Main package
│   ├── agents/         # Policy network, feature extraction, rewards
│   ├── algorithms/     # Evolution strategies trainer, episode runner
│   ├── data/           # SpatialData, ExtinctionRisk classes
│   ├── environment/    # BioEnv simulation engine
│   └── utils/          # Utilities, data loading, connectivity, KMZ export
├── docs/               # Quick start, training and inference guides
├── examples/           # Usage examples
├── scripts/            # Data-preparation and figure scripts for the tutorials
├── tests/              # Unit and integration tests
└── tutorials/          # Toy example, rewards and advanced example tutorials
```

## Citation

If you use CAPTAIN v.3, please cite:

```bibtex
@software{captain_3_2026,
  title = {CAPTAIN v.3: Conservation Area Prioritization Through Artificial INtelligence},
  version = {3.0.0rc1},
  year = {2026},
  url = {https://github.com/captain-project/captain3}
}

@article{silvestro2022improving,
  title={Improving biodiversity protection through artificial intelligence},
  author={Silvestro, Daniele and Goria, Stefano and Sterner, Thomas and Antonelli, Alexandre},
  journal={Nature sustainability},
  volume={5},
  number={5},
  pages={415--424},
  year={2022},
  publisher={Nature Publishing Group UK London}
}

@article{silvestro2025using,
  title={Using artificial intelligence to optimize ecological restoration for climate and biodiversity},
  author={Silvestro, Daniele and Goria, Stefano and Rau, E-ping and Ferreira de Lima, Renato Augusto and Groom, Ben and Jacobsson, Piotr and Sterner, Thomas and Antonelli, Alexandre},
  journal={bioRxiv},
  pages={2025--01},
  year={2025},
}
```



## License

This project is licensed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International, see [full license](https://github.com/captain-project/captain2/blob/main/CAPTAIN-License.pdf) for detail.

For commercial licensing inquiries or permission to deviate from these terms, please contact the development team.
