import timm
import torch.nn as nn


def build_model(
    model_name: str = "convnext_tiny",
    num_classes: int = 104,
    pretrained: bool = True,
) -> nn.Module:
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model
