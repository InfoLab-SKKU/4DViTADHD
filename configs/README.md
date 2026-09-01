# configs

This folder contains configuration files used to run experiments for the 4DViTADHD project (Multimodal Intermediate Fusion for ADHD Diagnosis using 4D Vision Transformer).

Purpose
- Centralize experiment settings (dataset, model, training, preprocessing, augmentation, and evaluation).
- Make experiments reproducible by keeping hyperparameters and file paths in version-controlled YAML/JSON config files.

How the configs are organized
- dataset/      : dataset-specific settings (paths, preprocessing parameters, splits)
- model/        : model architecture and modality/fusion-related settings
- training/     : optimizer, learning-rate schedule, batch size, number of epochs, checkpointing
- augmentation/ : data augmentation parameters and pipelines
- eval/         : evaluation settings (metrics, saved checkpoint to evaluate)
- examples/     : example or experiment-level config files that combine the above parts (if present)

Common fields you will find
- name: Human-readable config name or experiment id
- seed: Random seed for reproducibility
- data:
  - root: Path to dataset root
  - modalities: Which modalities are used (e.g., 'rgb', 'depth', 'motion', 'clinical')
  - preprocess: resize/cropping/normalization settings
- model:
  - type: Model class or shorthand (e.g., `4d_vit`, `fusion_net`)
  - hidden_dim, heads, depth: Transformer parameters
  - pretrained: path or boolean indicating use of pretrained weights
- training:
  - optimizer: e.g., `adamw`
  - lr: base learning rate
  - batch_size
  - epochs
  - weight_decay
  - checkpoint_freq
- augment:
  - flip: boolean
  - color_jitter: strength
  - temporal_sampling: parameters for 4D input sampling
- eval:
  - metrics: list of metrics to compute (accuracy, AUC, f1, sensitivity, specificity)
  - checkpoint: which checkpoint to use for evaluation

How to use a config
1. Pick or create a config YAML/JSON in this folder (or an experiment-level config that imports parts).
2. Run the training script with a `--config` flag. Example (adjust script names as needed):

```bash
python train.py --config configs/examples/experiment_01.yaml
```

If the repository uses a different CLI, look for scripts like `train.py`, `run_experiment.py`, or for a README at the repo root describing the exact command.

Creating a new config
- Start from an existing example in this folder and change:
  - dataset paths and split names
  - model parameters when trying different architectures
  - training hyperparameters for tuning
- Use clear, descriptive file names like `datasetX_4dvit_fusion_lr1e-4.yaml`.
- Add a brief header comment describing the experiment purpose, author, and date.

Notes & conventions
- Prefer relative paths (project-root relative) for dataset/model paths where possible so configs remain portable.
- Keep one canonical config per experiment to make reproduction straightforward — include seed and exact commit SHA in your experiment logs.
- If configs are split into small files (dataset/model/training), document how they are composed. If the repo supports `hydra` or another config composition tool, follow the project's conventions (check repo root README or code for details).

Questions or additions
- If you want, I can:
  - list the config files currently in this folder and summarize each one
  - convert JSON configs to YAML (or vice versa)
  - add a small example config tuned for quick local testing

