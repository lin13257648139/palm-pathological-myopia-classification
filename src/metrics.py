from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def classification_metrics(
    targets: list[int], predictions: list[int], probabilities: list[float]
) -> dict[str, object]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        average="binary",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(targets, probabilities)),
        "confusion_matrix": confusion_matrix(targets, predictions).tolist(),
        "support": int(len(targets)),
    }


def save_json(payload: dict[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_confusion_matrix(matrix: list[list[int]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(matrix)
    labels = ["Non-PM", "PM"]

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    image = ax.imshow(values, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )
    threshold = values.max() / 2 if values.size else 0
    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                str(values[row, col]),
                ha="center",
                va="center",
                color="white" if values[row, col] > threshold else "black",
                fontsize=12,
                fontweight="bold",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
def save_training_history(history: list[dict[str, float]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], marker="o", label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, [row["train_accuracy"] for row in history], marker="o", label="Train")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], marker="o", label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
# End of module.
