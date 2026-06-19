# Petals Flower Classifier

[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/tpu-getting-started)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)

A flower classification solution for the Kaggle **Petals to the Metal** competition. Uses transfer learning with ConvNeXt and EfficientNet backbones via `timm`, along with modern training techniques like MixUp/CutMix, EMA, and progressive resizing.

---

## Features

- **Multiple backbones**: ConvNeXt-Tiny, EfficientNet-B0 (easily extensible via `timm`)
- **Advanced augmentations**: RandomResizedCrop, ColorJitter, RandomErasing, vertical flips
- **MixUp / CutMix**: Stochastic interpolation and region masking for regularization
- **Exponential Moving Average (EMA)**: Smoothed model weights for better generalization
- **Automatic mixed precision (AMP)**: Faster training with `torch.cuda.amp`
- **Warmup + Cosine annealing LR schedule**: Stable training from the first epoch
- **Label smoothing**: Cross-entropy with 0.1 smoothing
- **Early stopping**: Patience-based stopping to prevent overfitting
- **TFRecord extraction**: Parallelized extraction of Kaggle TFRecord datasets
- **Kaggle-ready CLI**: Flags to run on Kaggle or local environments

---

## Results

After training:

| Backbone | Val Accuracy | Notes |
|----------|-------------|-------|
| ConvNeXt-Tiny | 94.1% | Without augmentations |
| ConvNeXt-Tiny | 93.9% | With augmentations |
| EfficientNet-B0 | 93.3% | With EMA, MixUp/CutMix |

---

## Project Structure

```
petals-flower-classifier/
├── src/
│   ├── __init__.py
│   ├── config.py        # Configuration (dataclasses)
│   ├── dataset.py       # Dataset classes, transforms, loaders
│   ├── model_factory.py # Model creation via timm
│   ├── trainer.py       # Training loop, MixUp/CutMix, EMA, validation
│   ├── inference.py     # Prediction and submission generation
│   └── utils.py         # TFRecord reader/extraction utilities
├── notebooks/           # Original Kaggle notebooks
├── models/              # Saved model checkpoints
├── data/                # Extracted images (gitignored)
├── tests/               # Unit tests
├── requirements.txt
├── run_training.py      # Main entry point
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-username/petals-flower-classifier.git
cd petals-flower-classifier

python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

pip install -r requirements.txt
```

---

## Usage

### 1. Download the competition data

Place the competition TFRecord files from [Kaggle](https://www.kaggle.com/competitions/tpu-getting-started/data) into a local directory, or use `--kaggle` to use Kaggle paths when running on the platform.

### 2. Extract TFRecords (optional, done automatically if missing)

```bash
python run_training.py --extract
```

### 3. Train the model

```bash
# Train with EfficientNet-B0 (default)
python run_training.py

# Train with ConvNeXt-Tiny
python run_training.py --model convnext_tiny

# Custom settings
python run_training.py \
    --model efficientnet_b0 \
    --batch-size 64 \
    --epochs 50 \
    --lr 3e-4

# Disable MixUp/CutMix or augmentations
python run_training.py --no-mixup --no-augment
```

### 4. Run inference only

```bash
python run_training.py --predict-only --checkpoint models/best_model_efficientnet_b0_ema.pth
```

### 5. Kaggle environment

```bash
python run_training.py --kaggle --extract
```

---

## Configuration

All hyperparameters are defined as dataclasses in `src/config.py`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `efficientnet_b0` | Backbone architecture |
| `batch_size` | 64 | Per-batch size |
| `epochs` | 50 | Max training epochs |
| `lr` | 3e-4 | Peak learning rate |
| `weight_decay` | 1e-4 | AdamW weight decay |
| `label_smoothing` | 0.1 | Label smoothing factor |
| `mixup_alpha` | 0.2 | Beta distribution alpha for MixUp |
| `cutmix_alpha` | 1.0 | Beta distribution alpha for CutMix |
| `ema_decay` | 0.999 | EMA smoothing factor |
| `patience` | 10 | Early stopping patience |
| `warmup_epochs` | 3 | Linear warmup duration |

---

## License

MIT
