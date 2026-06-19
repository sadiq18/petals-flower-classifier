import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def predict(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    ids = []
    preds = []

    for images, img_ids in tqdm(test_loader, desc="Predicting"):
        images = images.to(device)

        outputs = model(images)
        pred = outputs.argmax(dim=1).cpu().numpy()

        ids.extend(img_ids)
        preds.extend(pred)

    return pd.DataFrame({"id": ids, "label": preds})


def save_submission(
    df: pd.DataFrame,
    output_path: str = "submission.csv",
) -> None:
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(df.head())
