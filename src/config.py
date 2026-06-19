from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class TrainingConfig:
    image_size: int = 224
    num_classes: int = 104
    batch_size: int = 64
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True

    model_name: str = "efficientnet_b0"
    pretrained: bool = True

    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    epochs: int = 50
    warmup_epochs: int = 3

    head_epochs: int = 5
    head_lr: float = 3e-4

    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mixup_prob: float = 0.7

    ema_decay: float = 0.999
    patience: int = 10

    train_tfrecord_pattern: str = ""
    val_tfrecord_pattern: str = ""
    test_tfrecord_pattern: str = ""

    train_image_dir: str = "data/train_images"
    val_image_dir: str = "data/val_images"
    test_image_dir: str = "data/test_images"
    output_dir: str = "models"
    submission_path: str = "submission.csv"

    seed: int = 42

    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    train_augmentation: bool = True
    random_erasing_p: float = 0.25


@dataclass
class KaggleConfig(TrainingConfig):
    train_tfrecord_pattern: str = "/kaggle/input/competitions/tpu-getting-started/tfrecords-jpeg-224x224/train/*.tfrec"
    val_tfrecord_pattern: str = "/kaggle/input/competitions/tpu-getting-started/tfrecords-jpeg-224x224/val/*.tfrec"
    test_tfrecord_pattern: str = "/kaggle/input/competitions/tpu-getting-started/tfrecords-jpeg-224x224/test/*.tfrec"
    train_image_dir: str = "/kaggle/working/train_images"
    val_image_dir: str = "/kaggle/working/val_images"
    test_image_dir: str = "/kaggle/working/test_images"
    output_dir: str = "/kaggle/working"
    submission_path: str = "/kaggle/working/submission.csv"
