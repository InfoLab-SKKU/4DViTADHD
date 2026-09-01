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

If you'd like, I can also:
- add an example inference notebook under models/ or notebooks/ showing end-to-end loading and prediction, or
- create a template checkpoint loader function in models/utils.py so downstream code can load weights robustly.

Tell me which option you prefer and I'll add it next.