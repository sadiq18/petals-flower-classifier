import argparse
import os

import torch
import torch.nn as nn

from src.config import TrainingConfig, KaggleConfig
from src.dataset import create_train_val_loaders, create_test_loader
from src.model_factory import build_model
from src.trainer import train_model, validate
from src.inference import predict, save_submission
from src.utils import extract_tfrecords


def parse_args():
    parser = argparse.ArgumentParser(description="Petals Flower Classifier")
    parser.add_argument("--model", type=str, default="efficientnet_b0",
                        choices=["convnext_tiny", "efficientnet_b0"],
                        help="Model architecture")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable training augmentation")
    parser.add_argument("--no-mixup", action="store_true",
                        help="Disable Mixup/CutMix")
    parser.add_argument("--kaggle", action="store_true",
                        help="Use Kaggle paths")
    parser.add_argument("--extract", action="store_true",
                        help="Extract TFRecords before training")
    parser.add_argument("--predict-only", action="store_true",
                        help="Run inference only, skip training")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint for inference")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.kaggle:
        cfg = KaggleConfig()
    else:
        cfg = TrainingConfig()

    cfg.model_name = args.model
    cfg.batch_size = args.batch_size
    cfg.epochs = args.epochs
    cfg.lr = args.lr
    cfg.head_epochs = args.head_epochs
    cfg.train_augmentation = not args.no_augment

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Config: {cfg}")

    if args.extract:
        print("\n=== Extracting TFRecords ===")
        extract_tfrecords(cfg.train_tfrecord_pattern, cfg.train_image_dir, is_test=False)
        extract_tfrecords(cfg.val_tfrecord_pattern, cfg.val_image_dir, is_test=False)
        extract_tfrecords(cfg.test_tfrecord_pattern, cfg.test_image_dir, is_test=True)

    if not args.predict_only:
        print("\n=== Creating Data Loaders ===")
        train_loader, val_loader, class_names = create_train_val_loaders(
            train_image_dir=cfg.train_image_dir,
            val_image_dir=cfg.val_image_dir,
            image_size=cfg.image_size,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.persistent_workers,
            augment=cfg.train_augmentation,
        )
        cfg.num_classes = len(class_names)
        print(f"Found {cfg.num_classes} classes")

        print("\n=== Building Model ===")
        model = build_model(
            model_name=cfg.model_name,
            num_classes=cfg.num_classes,
            pretrained=cfg.pretrained,
        )
        model = model.to(device)

        from timm.utils import ModelEma
        model_ema = ModelEma(model, decay=cfg.ema_decay, device=device)

        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "classifier" in name:
                param.requires_grad = True

        head_optimizer = torch.optim.AdamW(
            model.classifier.parameters(),
            lr=cfg.head_lr,
            weight_decay=cfg.weight_decay,
        )
        head_scaler = torch.cuda.amp.GradScaler()

        print("\n=== Training Classifier Head ===")
        for epoch in range(cfg.head_epochs):
            from src.trainer import train_one_epoch
            train_loss = train_one_epoch(
                model, train_loader,
                nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing),
                head_optimizer, head_scaler, device,
                model_ema=model_ema,
            )
            val_acc = validate(model, val_loader, device)
            print(f"[HEAD] Epoch {epoch + 1} | Loss: {train_loss:.4f} | Val: {val_acc:.4f}")

        print("\n=== Full Fine-Tuning ===")
        for param in model.parameters():
            param.requires_grad = True

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.1, total_iters=cfg.warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=cfg.epochs - cfg.warmup_epochs
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[cfg.warmup_epochs],
        )

        scaler = torch.cuda.amp.GradScaler()

        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

        os.makedirs(cfg.output_dir, exist_ok=True)
        save_path = os.path.join(cfg.output_dir, f"best_model_{cfg.model_name}.pth")
        save_path_ema = os.path.join(cfg.output_dir, f"best_model_{cfg.model_name}_ema.pth")

        best_acc, best_acc_ema = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            num_epochs=cfg.epochs,
            model_ema=model_ema,
            use_mixup=not args.no_mixup,
            mixup_alpha=cfg.mixup_alpha,
            cutmix_alpha=cfg.cutmix_alpha,
            mixup_prob=cfg.mixup_prob,
            patience=cfg.patience,
            save_path=save_path,
            save_path_ema=save_path_ema,
        )

        print(f"\nBest Val Acc: {best_acc:.4f}")
        print(f"Best EMA Val Acc: {best_acc_ema:.4f}")

    print("\n=== Running Inference ===")
    if args.predict_only:
        import glob as pyglob
        from src.dataset import get_val_transform

        test_image_dir = cfg.test_image_dir
        if not os.path.exists(test_image_dir) or not pyglob.glob(os.path.join(test_image_dir, "*.jpg")):
            print(f"No test images found in {test_image_dir}, extracting...")
            extract_tfrecords(cfg.test_tfrecord_pattern, test_image_dir, is_test=True)

        model = build_model(
            model_name=cfg.model_name,
            num_classes=cfg.num_classes,
            pretrained=False,
        )
        checkpoint = args.checkpoint
        if checkpoint is None:
            checkpoint = os.path.join(cfg.output_dir, f"best_model_{cfg.model_name}_ema.pth")
        if not os.path.exists(checkpoint):
            checkpoint = os.path.join(cfg.output_dir, f"best_model_{cfg.model_name}.pth")

        print(f"Loading checkpoint: {checkpoint}")
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model = model.to(device)

        test_loader = create_test_loader(
            test_image_dir=test_image_dir,
            image_size=cfg.image_size,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )

    elif not args.predict_only:
        model_for_inference = model
        if best_acc_ema > best_acc:
            model_for_inference = model_ema.ema
            print("Using EMA model for inference")

        model_for_inference.eval()
        checkpoint_path = os.path.join(cfg.output_dir, f"best_model_{cfg.model_name}_ema.pth")
        if os.path.exists(checkpoint_path):
            torch.save(model_for_inference.state_dict(), checkpoint_path)

        test_loader = create_test_loader(
            test_image_dir=cfg.test_image_dir,
            image_size=cfg.image_size,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )

    df = predict(model, test_loader, device)
    save_submission(df, cfg.submission_path)


if __name__ == "__main__":
    main()
