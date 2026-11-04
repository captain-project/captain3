# CAPTAIN v.3.0 – Preview

**Conservation Area Prioritization Through Artificial INtelligence**

[![Status](https://img.shields.io/badge/status-Technical%20Preview-orange)](https://github.com/captain-project/captain3preview)
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

⚖️ **Species-level Rewards:** Reward the protected share of each species' population (weighted by threat status and range size), the worst-served species, or the overlap with any spatial layer, and change the reward weights during training with weight schedules. See the [rewards tutorial](https://github.com/captain-project/captain3preview/blob/main/tutorials/rewards/README.md).

📍 **Static Prioritization:** One-shot protection plans within a single time step, without population dynamics, for fast spatial prioritization.

🌍 **Google Earth Export:** Export protection plans as KMZ polygons to view them on satellite imagery in Google Earth or QGIS.

📘 **Hands-on Tutorials:** A step-by-step [toy-example walkthrough](https://github.com/captain-project/captain3preview/blob/main/tutorials/toy_example/README.md) covering data setup, reward calibration, training, and global vs. regional agents on a small toy dataset, and an [advanced example](https://github.com/captain-project/captain3preview/blob/main/tutorials/advanced_example/README.md) with edge effects, existing protected areas, connectivity and static prioritization. The [training guide](https://github.com/captain-project/captain3preview/blob/main/docs/training.md) explains the optimizer and its settings.

### What's new (November 2026)

New features: edge effects, connectivity, new reward terms and reward weight schedules, static single-step episodes and KMZ export (see above), with three new example scripts: [train_policy_advanced.py](https://github.com/captain-project/captain3preview/blob/main/examples/train_policy_advanced.py), [static_prioritization.py](https://github.com/captain-project/captain3preview/blob/main/examples/static_prioritization.py) and [train_policy_weight_schedule.py](https://github.com/captain-project/captain3preview/blob/main/examples/train_policy_weight_schedule.py).

Changes that affect existing runs:
- The evolution strategies trainer now standardizes the rewards of each epoch's perturbations before the update (`advantage_normalization="zscore"`), so the step size no longer grows with the reward scale. Pass `advantage_normalization=None` to `EvolStrategiesTrainer` for the previous behaviour.
- `LearningScheduler` keeps sigma at or above `min_sigma` also while adapting it to the similarity of the perturbations' plans.
- Budget managers round the number of cells protected per decision up, so the full protection target is reached.
- Species in the trait table that have no species distribution map are dropped with a warning instead of raising an error.


### ⚠️ Notice
**This repository contains a preview of CAPTAIN v.3.**

* **Status:** This is an active beta. The core simulation engine and training loops are functional, but the API and internal architecture are subject to change.
* **Development:** Primary development occurs in a private repository. This public mirror is provided for demonstration, testing, and early-access feedback.
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
git clone https://github.com/captain-project/captain3preview
cd captain3preview

# Sync dependencies and create virtual environment
uv sync
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

Working example scripts are in the [examples folder](https://github.com/captain-project/captain3preview/blob/main/examples)
- [plot\_input\_data.py](https://github.com/captain-project/captain3preview/blob/main/examples/plot_input_data.py) - Visualize spatial and time-varying input data
- [calibrate_rewards.py](https://github.com/captain-project/captain3preview/blob/main/examples/calibrate_rewards.py) - Calibrate the scales of the reward terms (run once before training)
- [train_policy.py](https://github.com/captain-project/captain3preview/blob/main/examples/train_policy.py) - Full training loop with real data
- [run_inference.py](https://github.com/captain-project/captain3preview/blob/main/examples/run_inference.py) - Load a trained model and perform optimization, plot the results
- [train_policy_advanced.py](https://github.com/captain-project/captain3preview/blob/main/examples/train_policy_advanced.py) - Edge effects, existing protected areas, connectivity and a worst-served species reward
- [static_prioritization.py](https://github.com/captain-project/captain3preview/blob/main/examples/static_prioritization.py) - One-shot plan within a single time step
- [train_policy_weight_schedule.py](https://github.com/captain-project/captain3preview/blob/main/examples/train_policy_weight_schedule.py) - Anneal the weight of a reward term during training

Example data can be downloaded [here](https://polybox.ethz.ch/index.php/s/WKdbHHGj3ayL9w9). 
A pre-trained model is available [here](https://polybox.ethz.ch/index.php/s/wZ5AMXPdzboZSm2). 

For a quicker, self-contained walkthrough using a small toy dataset, see the
[toy-example tutorial](https://github.com/captain-project/captain3preview/blob/main/tutorials/toy_example/README.md) —
covers data setup, reward calibration, training, and running inference with
both a single global agent and coordinated regional agents. The
[advanced example](https://github.com/captain-project/captain3preview/blob/main/tutorials/advanced_example/README.md) and the
[rewards tutorial](https://github.com/captain-project/captain3preview/blob/main/tutorials/rewards/README.md) build on it.

<img width="50%" alt="animated_map" src="https://github.com/user-attachments/assets/9ef85156-9a76-4a32-8c29-d5960481a7cb" />

## Project Structure

```
captain3preview/
├── captain/            # Main package
│   ├── agents/         # Policy network, feature extraction, rewards
│   ├── algorithms/     # Evolution strategies trainer, episode runner
│   ├── data/           # SpatialData, ExtinctionRisk classes
│   ├── environment/    # BioEnv simulation engine
│   └── utils/          # Utilities, data loading, connectivity, KMZ export
├── docs/               # Training and inference guides
├── examples/           # Usage examples
├── scripts/            # Data-preparation and figure scripts for the tutorials
└── tutorials/          # Toy example, rewards and advanced example tutorials
```

## Citation

If you use CAPTAIN v.3, please cite:

```bibtex
@software{captain_3_2026,
  title = {CAPTAIN v.3 beta: Conservation Area Prioritization Through Artificial INtelligence},
  year = {2026},
  url = {https://github.com/captain-project/}
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
