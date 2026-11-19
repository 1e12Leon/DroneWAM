# DroneWAM: Efficient World Action Model for Drone Visual Navigation

**Liang Yao**, **Fan Liu**, **Hongbo Lu**, **Wei Xu**, **Jianyu Jiang**, **Yijun Shen**, **Chuanyi Zhang**, **Pai Peng**

Paper: [arXiv](https://arxiv.org/abs/2609.33148)  
Code and data: Coming soon.

## News

- **2026/9/27:** The preprint of our paper *DroneWAM: Efficient World Action Model for Drone Visual Navigation* is available on arXiv.




## Contents

```text
configs/           Reproducible example configurations
src/dronewam/      Models, training, evaluation, and command-line tools
pyproject.toml     Package metadata and dependencies
```

Model weights, datasets, experiment outputs, and repository history are not
included.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Data format

Each trajectory directory must contain sequential RGB frames and one state log:

```text
trajectory_0001/
  images/000000.png
  images/000001.png
  state_log.csv
```

The state log columns are:

```text
frame_id,timestamp,pos_x,pos_y,pos_z,roll,pitch,yaw
```

Create trajectory-level train and validation manifests with:

```bash
dronewam-build-manifest --dataset-root ./dataset --output-dir ./data
```

Generated manifests contain local dataset locations and are ignored by Git.

The model predicts ego-frame actions in the order
`[dx, dy, dz, droll, dpitch, dyaw]`.

## Configuration

- `configs/base.yaml`: base world-action model training.
- `configs/adaptive.yaml`: iterative planner and joint rollout-depth gate.
- `configs/token_ae.yaml`: optional latent-token autoencoder training.

All paths are repository-relative examples. Place local checkpoints under
`weights/`, or edit the YAML files. The encoder and TokenAE checkpoints are not
distributed with this repository.

## Training and evaluation

Train the TokenAE when a compatible checkpoint is not already available:

```bash
dronewam-train-token-ae --config configs/token_ae.yaml
```

Train and evaluate the base model:

```bash
dronewam-train --config configs/base.yaml
dronewam-eval \
  --config configs/base.yaml \
  --checkpoint outputs/base/latest.pt \
  --output outputs/base/metrics.json
```

Train the adaptive planner and the final joint horizon gate from a base model:

```bash
dronewam-train-adaptive \
  --config configs/adaptive.yaml \
  --stage planner \
  --base-checkpoint outputs/base/latest.pt \
  --output outputs/adaptive/planner.pt

dronewam-train-adaptive \
  --config configs/adaptive.yaml \
  --stage joint-horizon \
  --base-checkpoint outputs/base/latest.pt \
  --adaptive-checkpoint outputs/adaptive/planner.pt \
  --output outputs/adaptive/joint_horizon.pt
```

Evaluate directly, or first create a cache on the training calibration subset
and fit the optional rollout-depth bias:

```bash
dronewam-eval-joint-horizon \
  --config configs/adaptive.yaml \
  --base-checkpoint outputs/base/latest.pt \
  --joint-checkpoint outputs/adaptive/joint_horizon.pt \
  --split train --calibration-subset \
  --cache-output outputs/adaptive/calibration_cache.pt \
  --output outputs/adaptive/calibration_metrics.json

dronewam-calibrate-joint-horizon \
  --cache outputs/adaptive/calibration_cache.pt \
  --output outputs/adaptive/candidate_bias.json

dronewam-eval-joint-horizon \
  --config configs/adaptive.yaml \
  --base-checkpoint outputs/base/latest.pt \
  --joint-checkpoint outputs/adaptive/joint_horizon.pt \
  --candidate-bias outputs/adaptive/candidate_bias.json \
  --output outputs/adaptive/metrics.json
```

For fixed-depth observation-only inference, see `dronewam-rollout --help`.

## Cite

If you find this work useful, please cite our paper as:

```bibtex
@misc{yao2026dronewam,
      title={DroneWAM: Efficient World Action Model for Drone Visual Navigation},
      author={Liang Yao and Fan Liu and Hongbo Lu and Wei Xu and Jianyu Jiang and Yijun Shen and Chuanyi Zhang and Pai Peng},
      year={2026},
      eprint={2609.33148},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2609.33148},
}
```
