import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


def get_train_transform(image_size: int = 224, augment: bool = True):
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=0.25),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


def get_val_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def create_train_val_loaders(
    train_image_dir: str,
    val_image_dir: str,
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    augment: bool = True,
):
    train_dataset = ImageFolder(
        root=train_image_dir,
        transform=get_train_transform(image_size, augment=augment),
    )

    val_dataset = ImageFolder(
        root=val_image_dir,
        transform=get_val_transform(image_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return train_loader, val_loader, train_dataset.classes


class TestDataset(Dataset):
    def __init__(self, folder: str, transform=None):
        self.paths = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".jpg")
        ])
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        img_id = os.path.basename(path).replace(".jpg", "")
        return img, img_id


def create_test_loader(
    test_image_dir: str,
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 2,
):
    transform = get_val_transform(image_size)

    test_dataset = TestDataset(folder=test_image_dir, transform=transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return test_loader
