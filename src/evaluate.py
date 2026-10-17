from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data import FundusDataset, discover_validation_samples
from src.metrics import classification_metrics, save_confusion_matrix, save_json
from src.model import IMAGE_SIZE, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate on the official PALM validation labels.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-xlsx", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/checkpoints/best_model.pth")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/official_validation"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    image_size = int(checkpoint.get("image_size", IMAGE_SIZE))
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    samples = discover_validation_samples(args.images_dir, args.labels_xlsx)
    dataset = FundusDataset(samples, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    targets: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    names: list[str] = []
    with torch.no_grad():
        for images, labels, filenames in tqdm(loader, desc="official validation"):
            logits = model(images.to(device, non_blocking=True))
            positive_probabilities = torch.softmax(logits, dim=1)[:, 1]
            batch_predictions = logits.argmax(dim=1)
            targets.extend(labels.tolist())
            predictions.extend(batch_predictions.cpu().tolist())
            probabilities.extend(positive_probabilities.cpu().tolist())
            names.extend(filenames)

    metrics = classification_metrics(targets, predictions, probabilities)
    save_json(metrics, args.output_dir / "metrics.json")
    save_confusion_matrix(metrics["confusion_matrix"], args.output_dir / "confusion_matrix.png")
    pd.DataFrame(
        {
            "image": names,
            "true_label": targets,
            "predicted_label": predictions,
            "pm_probability": probabilities,
        }
    ).to_csv(args.output_dir / "predictions.csv", index=False)

    print(f"Device: {device}")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "support"):
        print(f"{key}: {metrics[key]}")
    print(f"confusion_matrix: {metrics['confusion_matrix']}")


if __name__ == "__main__":
    main()
