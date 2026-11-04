## CAPTAIN 3 Examples

Working example scripts are in the `examples/` folder:
- **[plot\_input\_data.py](https://github.com/captain-project/captain3preview/blob/main/examples/plot_input_data.py)** - Visualize spatial and time-varying input data
- **[calibrate_rewards.py](https://github.com/captain-project/captain3preview/blob/main/examples/calibrate_rewards.py)** - Calibrate the scales of the reward terms against the global agent (run once before training)
- **[train_policy.py](https://github.com/captain-project/captain3preview/blob/main/examples/train_policy.py)** - Full training loop with real data
- **[run_inference.py](https://github.com/captain-project/captain3preview/blob/main/examples/run_inference.py)** - Load a trained model and perform optimization. Exports the plan as `protection_plan.kmz` for Google Earth, and with `SAVE_XAI_DATA = True` it also saves features, policy scores and cell rankings for xAI analyses (see [docs/inference.md](https://github.com/captain-project/captain3preview/blob/main/docs/inference.md))
- **[train_policy_advanced.py](https://github.com/captain-project/captain3preview/blob/main/examples/train_policy_advanced.py)** - Edge effects, existing protected areas, connectivity among reserves and a worst-served species reward with a weight schedule (see the [advanced example](https://github.com/captain-project/captain3preview/blob/main/tutorials/advanced_example/README.md))
- **[static_prioritization.py](https://github.com/captain-project/captain3preview/blob/main/examples/static_prioritization.py)** - A one-shot plan within a single time step, without population dynamics
- **[train_policy_weight_schedule.py](https://github.com/captain-project/captain3preview/blob/main/examples/train_policy_weight_schedule.py)** - Protected population + worst-served species rewards, annealing the weight of the worst-served term (see the [rewards tutorial](https://github.com/captain-project/captain3preview/blob/main/tutorials/rewards/README.md#reward-weight-schedules))

The advanced examples run on the toy dataset of the
[toy-example tutorial](https://github.com/captain-project/captain3preview/blob/main/tutorials/toy_example/README.md); see the
[training guide](https://github.com/captain-project/captain3preview/blob/main/docs/training.md) for the optimizer settings they use.

Example data can be downloaded [here](https://polybox.ethz.ch/index.php/s/WKdbHHGj3ayL9w9). 
A pre-trained model is available [here](https://polybox.ethz.ch/index.php/s/wZ5AMXPdzboZSm2). 

For a quicker, self-contained walkthrough using a small toy dataset — including
reward calibration and global vs. regional agents — see the
[toy-example tutorial](https://github.com/captain-project/captain3preview/blob/main/tutorials/toy_example/README.md).

The scripts generate a number of plots, included the optimal configuration of protected areas under the given policy constraints and objectives.

<img width="75%" alt="example_protection_through_time" src="https://github.com/user-attachments/assets/f8afeb71-6bc8-472a-ae29-e616d1ae272d" />


To run a script e.g.:

```bash
cd examples
# Sync dependencies and create virtual environment
uv run plot_input_data.py
```

CAPTAIN 3 can also be run from the python console. In a Terminal window, browse to the captain3preview directory and launch Python using:

```
uv run python
```

You can then import CAPTAIN using:

```python
>>> import captain as cn
>>> cn.__version__
```
