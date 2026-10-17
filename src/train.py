from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data import CLASS_NAMES, FundusDataset, discover_training_samples
from src.metrics import (
    classification_metrics,
    save_confusion_matrix,
    save_json,
    save_training_history,
)
from src.model import IMAGE_SIZE, MODEL_NAME, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a binary PM classifier on PALM images.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.82, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []

    context = torch.enable_grad if training else torch.no_grad
    with context():
        progress = tqdm(loader, desc="train" if training else "validate", leave=False)
        for images, labels, _ in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            positive_probabilities = torch.softmax(logits.detach(), dim=1)[:, 1]
            batch_predictions = logits.detach().argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            targets.extend(labels.cpu().tolist())
            predictions.extend(batch_predictions.cpu().tolist())
            probabilities.extend(positive_probabilities.cpu().tolist())
            progress.set_postfix(loss=f"{loss.item():.4f}")

    metrics = classification_metrics(targets, predictions, probabilities)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_training_samples(args.data_dir)
    labels = [sample.label for sample in samples]
    train_indices, val_indices = train_test_split(
        np.arange(len(samples)),
        test_size=args.val_size,
        random_state=args.seed,
        stratify=labels,
    )
    train_samples = [samples[index] for index in train_indices]
    val_samples = [samples[index] for index in val_indices]
    train_transform, eval_transform = make_transforms()
    train_dataset = FundusDataset(train_samples, train_transform)
    val_dataset = FundusDataset(val_samples, eval_transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(pretrained=not args.no_pretrained).to(device)
    counts = Counter(sample.label for sample in train_samples)
    class_weights = torch.tensor(
        [len(train_samples) / (2 * counts[index]) for index in range(2)],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    print(f"Device: {device}")
    print(f"Samples: train={len(train_samples)}, validation={len(val_samples)}")
    print(f"Class distribution: {dict(Counter(labels))}")

    history: list[dict[str, float]] = []
    best_f1 = -1.0
    best_metrics: dict[str, object] = {}
    checkpoint_path = checkpoint_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer, scaler=scaler
        )
        val_metrics = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": float(train_metrics["loss"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "val_f1": float(val_metrics["f1"]),
            "val_roc_auc": float(val_metrics["roc_auc"]),
        }
        history.append(row)
        print(
            "train loss={train_loss:.4f} acc={train_accuracy:.4f} | "
            "val loss={val_loss:.4f} acc={val_accuracy:.4f} "
            "f1={val_f1:.4f} auc={val_roc_auc:.4f}".format(**row)
        )

        if float(val_metrics["f1"]) > best_f1:
            best_f1 = float(val_metrics["f1"])
            best_metrics = val_metrics
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": MODEL_NAME,
                    "class_names": CLASS_NAMES,
                    "image_size": IMAGE_SIZE,
                    "label_definition": {"0": "Non-PM (N/H)", "1": "PM (P)"},
                    "seed": args.seed,
                    "epoch": epoch,
                    "internal_validation_metrics": val_metrics,
                },
                checkpoint_path,
            )

    history_path = args.output_dir / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    save_json(best_metrics, args.output_dir / "internal_validation_metrics.json")
    save_confusion_matrix(
        best_metrics["confusion_matrix"], args.output_dir / "internal_confusion_matrix.png"
    )
    save_training_history(history, args.output_dir / "training_history.png")
    print(f"\nBest checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
