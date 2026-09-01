# Models

This document describes the model components, checkpoints, and how to run training and inference for the 4DViTADHD project ("Multimodal Intermediate Fusion for ADHD Diagnosis using 4D Vision Transformer"). It is intended as a quick reference for developers and users.

## Overview

The repository implements a multimodal intermediate-fusion architecture based on a 4D Vision Transformer (4DViT) backbone. The architecture is designed to consume spatio-temporal medical imaging data (4D: 3D volumes across time) and optional non-image modalities (tabular/clinical features) and produce a diagnostic classification score.

High-level components:
- 4DViT backbone: tokenizes spatio-temporal patches and applies transformer blocks to learn joint spatio-temporal representations.
- Intermediate Fusion module: fuses image-derived features with other modality embeddings at intermediate transformer layers.
- Classification head: lightweight MLP that maps fused features to output logits (binary or multi-class diagnosis).

## Files and expected checkpoints

- models/
  - checkpoint files (recommended location): models/checkpoints/
  - naming convention (recommended): 4dvit_{split|fold}_{epoch|best}.pth (e.g. `4dvit_best.pth`)

If you run training with the provided scripts, final and best checkpoints will be written to models/checkpoints/. Place any pretrained weights you want to use there and point the training/inference scripts to the file path.

## How to load a checkpoint (PyTorch)

```python
import torch
from models.model import FourDViT  # adjust import to actual model class path

ckpt_path = 'models/checkpoints/4dvit_best.pth'
state = torch.load(ckpt_path, map_location='cpu')
model = FourDViT(**state.get('model_args', {}))
model.load_state_dict(state['model_state'])
model.eval()
```

Notes:
- Some checkpoints may store the full training state (optimizer, epoch). When only `model_state` is present, pass the correct model constructor args.
- If the repository uses a wrapper or Lightning, adapt the loading pattern (e.g., pl.LightningModule.load_from_checkpoint).

## Input format and shapes

- Imaging input: expected as a tensor of shape (B, C, T, D, H, W) or converted into a sequence of spatio-temporal patches, depending on the preprocessing pipeline. Typical shapes will vary by dataset; the code includes preprocessing utilities in the `data/` or `datasets/` modules.
- Tabular/clinical features: expected as (B, F) where F is the number of features.

Before inference, run the same normalization and resampling pipeline used in training. See the data preprocessing notebook/scripts for exact transforms.

## Training recommendations

- Loss: cross-entropy for multi-class or BCEWithLogitsLoss for binary classification.
- Optimizer: AdamW with weight decay (e.g., 1e-4).
- LR schedule: cosine annealing or step LR. Typical starting LR: 1e-4 — 5e-5 depending on batch size.
- Batch size: constrained by GPU memory due to 4D volumes. Use gradient accumulation when needed.
- Regularization: dropout in transformer heads and label smoothing when appropriate.
- Early stopping: monitor validation AUC or balanced accuracy.

Hyperparameters used in experiments (example):
- epochs: 100
- lr: 2e-4 with warmup for first 5 epochs
- weight_decay: 1e-4
- scheduler: CosineAnnealingLR

Adjust these to your dataset and compute budget.

## Inference

Use the inference script (e.g., scripts/inference.py) to run a checkpoint on hold-out data. Example:

```bash
python scripts/inference.py \
  --checkpoint models/checkpoints/4dvit_best.pth \
  --data-dir /path/to/preprocessed/data \
  --output results/predictions.csv \
  --batch-size 2
```

The inference script should apply the same preprocessing pipeline as training and save per-sample predictions and scores.

## Evaluation metrics

Recommended metrics for diagnosis tasks:
- Area under ROC curve (AUC)
- Accuracy, balanced accuracy
- Precision, recall, F1-score
- Confusion matrix

Report cross-validation or test-set averages and standard deviations when applicable.

## Reproducibility

- Set random seeds for PyTorch, NumPy, and Python's random module.
- Log full training args and environment (CUDA, PyTorch version).
- Save model checkpoints and a copy of the training config for each experiment.

## Notes and troubleshooting

- If inference runs out of memory, reduce batch size or use mixed precision (AMP).
- If training is unstable, try reducing learning rate, increasing weight decay, or clipping gradients.

## Citation

If you use this code, please cite: "Multimodal Intermediate Fusion for ADHD Diagnosis using 4D Vision Transformer" (InfoLab-SKKU).

---

## About this folder (what each file does)

This repository's models/ folder contains the model implementations, a small builder utility, and helper utilities. Below is a quick map and notes to help you understand how pieces fit together.

- models/__init__.py
  - Re-exports the project builder: `from .build import build_model`. Import `build_model` for config-driven construction.

- models/build.py
  - A thin factory that reads the project configuration and instantiates the correct model class (e.g., cross-scale MIFSM/Vit variants). Use `build_model(config, args)` when running experiments from the project's training scripts.

- models/VitADHD.py
  - Implementation of a cross-scale Vision Transformer variant used by the project. Class name: `VitADHD` (import as `from models.VitADHD import VitADHD`).
  - Default constructor values are defined in the file — check the source for `embed_dim`, `depths`, `num_heads`, etc.
  - Expected input for this file: standard image tensor shapes (B, C, H, W) after preprocessing.

- models/4D_VitADHD.py
  - Adapted implementation for 4D fMRI data. Class name: `VitADHD` (import as `from models.4D_VitADHD import VitADHD`).
  - Defaults (as implemented): img_size=(96, 96, 96, 20), patch_size=(4,4,4), in_chans=1, num_classes=2, embed_dim=24, depths=[2,2,6,2], num_heads=[3,6,12,24].
  - Expected input shape for this model: (B, C, D, H, W, T) i.e. channel-first volumes with an explicit time dimension T.
  - The 4D model adapts patch embedding and patch merging to preserve/handle the temporal axis.

- models/readme.md
  - This file — quick reference for the folder.

- models/utils/
  - Helper modules used across training and evaluation. Key files:
    - data_module.py — data loading and preprocessing utilities (large, contains dataset / dataloader logic used by training/inference).
    - losses.py — custom loss functions and wrappers used by experiments.
    - lr_scheduler.py — learning rate scheduling helpers used by training scripts.
    - metrics.py — evaluation metric helpers (AUC, accuracy wrappers, etc.).
    - seed_creation.py — reproducible-seed helpers.
    - neptune_utils.py — optional logging integration helpers.
    - parser.py — argument/config parsing helpers.
    - data_preprocess_and_load/ — additional preprocessing scripts and loaders (dataset-specific).

If you need to adapt loading or modify the training loop, start by reading `models/utils/data_module.py` and `models/build.py` to see how the model is constructed and how data is fed into it.

## Quick examples (copy-paste)

- Import and construct the 4D model with defaults and load checkpoint:

```python
import torch
from models.4D_VitADHD import VitADHD

ckpt_path = 'models/checkpoints/4dvit_best.pth'
ckpt = torch.load(ckpt_path, map_location='cpu')
# construct model using implementation defaults (adjust sizes to your data)
model = VitADHD(img_size=(96,96,96,20), patch_size=(4,4,4), in_chans=1, num_classes=2)

# checkpoint formats in this repo may vary; common keys:
state = ckpt.get('model_state', ckpt.get('state_dict', ckpt))
# when checkpoint has a full state dict from a wrapper, you might need to strip prefixes:
try:
    model.load_state_dict(state)
except RuntimeError:
    # try removing typical wrapper prefixes like 'module.'
    new_state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(new_state)

model.eval()
```

- Using the repo builder (when running with a config object):

```python
from models import build_model
# config should follow the project's config structure (see configs/)
model = build_model(config, args)
```

## Where to look next

- models/4D_VitADHD.py and models/VitADHD.py contain the canonical implementations; open them to inspect constructor arguments and adjust `img_size`, `patch_size`, `embed_dim`, and `num_classes` to your dataset.
- If you want to add robust checkpoint-loading utilities, consider adding `models/utils/checkpoint_loader.py` that:
  - detects `model_state` vs `state_dict` vs raw dict
  - strips common prefixes (module., model.)
  - supports loading Lightning checkpoints (`pl.LightningModule.load_from_checkpoint`)

---

If you want, I can now:
- add the suggested `models/utils/checkpoint_loader.py` file and a short usage example, or
- append a small diagram/ASCII flow showing how data -> model -> output flows in the 4D model.

Tell me which you prefer and I'll add it.