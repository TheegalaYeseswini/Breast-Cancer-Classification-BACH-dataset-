# DSResNet50 — BACH Breast Cancer Histology Classifier

A Kaggle-optimised PyTorch training notebook for 4-class breast cancer histopathology image classification on the [BACH dataset](https://iciar2018-challenge.grand-challenge.org/). Built around a custom **DSResNet50** architecture: a ResNet-50 backbone where all inner 3×3 convolutions are replaced with depthwise separable convolutions, optionally augmented with Squeeze-Excitation attention and Stochastic Depth regularisation.

---

## Task

Classify whole-slide microscopy images into one of four classes:

| Class | Description |
|---|---|
| **Normal** | Healthy breast tissue |
| **Benign** | Non-cancerous abnormality |
| **InSitu** | Carcinoma confined within ducts/lobules |
| **Invasive** | Invasive carcinoma (highest clinical priority) |

---

## Architecture: DSResNet50

Follows the standard ResNet-50 layer plan (`3-4-6-3` blocks) with three key modifications:

- **Depthwise Separable Convolutions** replace every inner 3×3 conv in the bottleneck blocks, reducing parameter count while preserving receptive field. The stem uses a regular 7×7 conv (cross-channel RGB patterns cannot be learned by a depth-wise filter with only 3 channels).
- **Squeeze-Excitation (SE) blocks** are appended after each bottleneck to perform channel-wise attention (`USE_SE = True` by default, `reduction = 16`).
- **Drop Path / Stochastic Depth** rates are distributed linearly across all 16 blocks from `0` (first block) to `STOCHASTIC_DEPTH_RATE` (last block), following the schedule from the original paper.

```
Input (3×H×W)
  └─ Stem: Conv7×7 → BN → ReLU → MaxPool
      └─ Layer 1–4: DSBottleneck blocks × [3, 4, 6, 3]
          each block: 1×1 Conv → DS Conv3×3 → 1×1 Conv → SE → DropPath → Residual
      └─ AdaptiveAvgPool → Dropout → Linear(2048, 4)
```

---

## Key Training Features

### Regularisation
- **Label smoothing** — `ε = 0.10`
- **Dropout** — `p = 0.45` before the final linear layer
- **Stochastic Depth** — linearly distributed drop-path rate, max `0.10`
- **Mixup / CutMix** — applied with `MIX_PROB` (disabled by default at `BATCH_SIZE=1`; `randperm(1)` is always the identity)

### Class Imbalance
- Per-class weights derived from inverse frequency and passed directly to `CrossEntropyLoss`. All four classes are weighted purely by their sample counts — no per-class manual adjustments.

### Optimiser & Schedule
- **AdamW** — `lr = 7e-4`, `weight_decay = 5e-4`, `betas = (0.9, 0.999)`
- **Warmup + Cosine Annealing** — 5-epoch linear warmup, then cosine decay to `min_lr = 1e-6` over 120 epochs

### Mixed Precision & Gradient Accumulation
- AMP via `torch.amp.autocast` and `GradScaler`
- Effective batch size = 16 via 16 gradient accumulation steps (`BATCH_SIZE = 1`)
- Gradient clipping at norm 1.0

---

## Augmentation Pipeline

Augmentations were deliberately curated for high-resolution (~2048 px) histopathology slides. Several transforms present in generic pipelines were explicitly removed:

| Removed | Reason |
|---|---|
| ElasticTransform / GridDistortion | Warps glandular architecture — the key InSitu vs Invasive discriminator |
| ISONoise | Camera sensor noise is absent from slide scanners |
| MotionBlur | Not physically present in microscopy |
| GridDropout | Erases ~500×500 px blocks at 2048 px — destroys diagnostic regions |
| GaussNoise `var_limit > 20` | Higher limits erase chromatin texture |

Retained augmentations: horizontal/vertical flip, 90° rotation, shift-scale-rotate, colour jitter, Gaussian noise, Gaussian blur, sharpening, small CoarseDropout.

**Test-Time Augmentation (TTA):** 8-fold — 4 rotations × 2 flips — averaged at inference.

---

## Checkpointing

The `CheckpointManager` writes to `/kaggle/working/checkpoints/`:

| File | Contents | When saved |
|---|---|---|
| `checkpoint_latest.pth` | Full training state | End of every epoch |
| `checkpoint_best.pth` | Full training state | When val accuracy improves |
| `checkpoint_epoch_NNNN.pth` | Full training state | Every 10 epochs |
| `checkpoint_panic.pth` | Full training state | Every 50 optimiser steps (mid-epoch) |
| `model_weights_best.pth` | Weights only (no optimiser) | When val accuracy improves |
| `model_best_epNNNN.pth` | Weights only, unique per epoch | When val accuracy improves |

`load_latest()` compares the epoch number stored in `checkpoint_latest.pth` and `checkpoint_panic.pth` and resumes from whichever is more recent.

### SIGTERM Handling

Kaggle sends `SIGTERM` ~60 seconds before killing a kernel. A signal handler sets a global flag; the training loop checks this flag at the start of every batch and immediately writes a panic checkpoint before raising `SystemExit`. This makes interrupted long runs fully resumable.

---

## Evaluation Metrics

Reported after each validation epoch and in a final `best_metrics.json`:

- Per-class Precision, Recall, F1 (via `classification_report`)
- Cohen's Kappa
- Matthews Correlation Coefficient (MCC)
- ROC-AUC (macro and weighted, one-vs-rest)
- Raw and row-normalised confusion matrices

Plots saved to `/kaggle/working/logs/`: `training_curves.png`, `confusion_matrix.png`, `roc_curves.png`.

---

## Configuration

All hyperparameters live in the `Config` class at the top of the notebook. Key fields:

```python
class Config:
    CLASSES      = ["Benign", "InSitu", "Invasive", "Normal"]
    EPOCHS       = 120
    BATCH_SIZE   = 1
    ACCUMULATION_STEPS = 16   # effective batch = 16
    LR           = 7e-4
    WEIGHT_DECAY = 5e-4
    DROPOUT      = 0.45
    USE_SE       = True
    STOCHASTIC_DEPTH_RATE  = 0.10
    LABEL_SMOOTHING        = 0.10
    PATIENCE               = 30
    INTRA_EPOCH_SAVE_STEPS = 50   # panic checkpoint frequency
```

---


## Inference
Final val_acc (best ckpt): 46.25%  |  best ever: 46.25%

==============================================================
  EVALUATION METRICS
==============================================================
  Class         Precision     Recall         F1   Support
  ----------------------------------------------------------
  Benign           0.7143     0.5000     0.5882        20
  InSitu           0.3548     0.5500     0.4314        20
  Invasive         0.5000     0.2000     0.2857        20
  Normal           0.4444     0.6000     0.5106        20
  ----------------------------------------------------------
  macro avg        0.5034     0.4625     0.4540        80
  weighted avg     0.5034     0.4625     0.4540        80

  Cohen Kappa     : 0.2833
  MCC             : 0.2943
  ROC-AUC macro   : 0.6566
  ROC-AUC weighted: 0.6566
==============================================================

### From weights-only file (recommended)

```python
model, meta = load_model_for_inference(
    "/kaggle/working/checkpoints/model_weights_best.pth"
)
result = predict_image(model, "path/to/image.tif", cfg, device, tta=True)
# {'class': 'Invasive', 'confidence': 0.91, 'probabilities': {...}}
```

### Resuming training from a specific checkpoint

```python
model, optimizer, scheduler, scaler, start_epoch, t_hist, v_hist = \
    resume_training_checkpoint("checkpoints/checkpoint_best.pth")
```

### ONNX export

```python
export_onnx(
    weights_path="checkpoints/model_weights_best.pth",
    out_path="model_best.onnx",
    input_h=512, input_w=512,   # spatial size is dynamic at runtime
)
```

Exported with `dynamic_axes` on batch, height, and width — the ONNX model accepts arbitrary spatial resolution at inference.

---

## Requirements

| Package | Role |
|---|---|
| `torch` ≥ 2.x | Core framework (CUDA recommended) |
| `albumentations` | Augmentation pipeline |
| `opencv-python` | Image loading |
| `scikit-learn` | Metrics |
| `seaborn` | Confusion matrix plots |
| `Pillow` | Fallback image loading |

Missing packages are auto-installed at cell startup via `pip_install`.

---

## Dataset Layout

The dataset auto-discovery function walks `/kaggle/input` and finds the first directory containing all four class subfolders:

```
/kaggle/input/
  └─ <dataset-name>/
      ├─ Benign/
      │   ├─ b001.tif
      │   └─ ...
      ├─ InSitu/
      ├─ Invasive/
      └─ Normal/
```

Supported image extensions: `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`, `.bmp`.

An 80/20 stratified split is applied per class (seeded, shuffled within class before splitting).

---

## Downloading Weights from Kaggle

After training, run the **"Save to Kaggle Output"** cell. It copies:
- All `model_best_epNNNN.pth` backups
- `model_weights_best.pth`
- `checkpoint_best.pth`

to `/kaggle/working/` (the top-level Output directory). From the Kaggle UI: **Output tab → right sidebar → download**.
