import random

import numpy as np
import torch
import torch.nn as nn
from timm.utils import ModelEma
from tqdm import tqdm


def mixup(images: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0)).to(images.device)

    mixed_images = lam * images + (1 - lam) * images[index]
    labels_a, labels_b = labels, labels[index]

    return mixed_images, labels_a, labels_b, lam


def cutmix(images: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size, _, H, W = images.size()
    index = torch.randperm(batch_size).to(images.device)

    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    labels_a, labels_b = labels, labels[index]

    return images, labels_a, labels_b, lam


def mixup_cutmix(
    images: torch.Tensor,
    labels: torch.Tensor,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
    mixup_prob: float = 0.7,
):
    if random.random() < mixup_prob:
        return mixup(images, labels, alpha=mixup_alpha)
    else:
        return cutmix(images, labels, alpha=cutmix_alpha)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    model_ema: ModelEma | None = None,
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
    mixup_prob: float = 0.7,
) -> float:
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)

        if use_mixup:
            images, labels_a, labels_b, lam = mixup_cutmix(
                images, labels,
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                mixup_prob=mixup_prob,
            )

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)

            if use_mixup:
                loss = (
                    lam * criterion(outputs, labels_a)
                    + (1 - lam) * criterion(outputs, labels_b)
                )
            else:
                loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if model_ema is not None:
            model_ema.update(model)

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Validating"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    num_epochs: int,
    model_ema: ModelEma | None = None,
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
    mixup_prob: float = 0.7,
    patience: int = 10,
    save_path: str = "best_model.pth",
    save_path_ema: str = "best_model_ema.pth",
):
    best_acc = 0.0
    best_acc_ema = 0.0
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            model_ema=model_ema,
            use_mixup=use_mixup,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            mixup_prob=mixup_prob,
        )

        val_acc = validate(model, val_loader, device)
        print(f"Epoch {epoch + 1:2d}/{num_epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

        if model_ema is not None:
            val_acc_ema = validate(model_ema.ema, val_loader, device)
            print(f"{'':>12}EMA Val Acc: {val_acc_ema:.4f}")

        if scheduler is not None:
            scheduler.step()

        improved = False

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

        if model_ema is not None and val_acc_ema > best_acc_ema:
            best_acc_ema = val_acc_ema
            torch.save(model_ema.ema.state_dict(), save_path_ema)
            improved = True
        elif model_ema is None and val_acc > best_acc:
            improved = True

        if improved:
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    return best_acc, best_acc_ema
