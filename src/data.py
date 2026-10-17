from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


CLASS_NAMES = ["Non-PM", "Pathological Myopia"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


def label_from_training_filename(filename: str) -> int:
    """Map PALM training prefixes to the official binary PM target.

    P = pathological myopia (positive class); N and H belong to the
    non-pathological-myopia class for the challenge classification task.
    """

    prefix = Path(filename).stem[:1].upper()
    if prefix == "P":
        return 1
    if prefix in {"N", "H"}:
        return 0
    raise ValueError(f"Unsupported PALM filename prefix: {filename}")


def discover_training_samples(image_dir: str | Path) -> list[Sample]:
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Training image directory not found: {image_dir}")

    samples: list[Sample] = []
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            samples.append(Sample(path=path, label=label_from_training_filename(path.name)))

    if not samples:
        raise ValueError(f"No images found in: {image_dir}")
    return samples


def discover_validation_samples(
    image_dir: str | Path, labels_xlsx: str | Path
) -> list[Sample]:
    image_dir = Path(image_dir)
    labels_xlsx = Path(labels_xlsx)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Validation image directory not found: {image_dir}")
    if not labels_xlsx.is_file():
        raise FileNotFoundError(f"Validation label file not found: {labels_xlsx}")

    frame = pd.read_excel(labels_xlsx)
    required = {"imgName", "Label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Validation label file is missing columns: {sorted(missing)}")

    samples: list[Sample] = []
    for row in frame.itertuples(index=False):
        path = image_dir / str(row.imgName)
        if not path.is_file():
            raise FileNotFoundError(f"Validation image listed in labels is missing: {path}")
        label = int(row.Label)
        if label not in {0, 1}:
            raise ValueError(f"Expected binary label for {path.name}, got {label}")
        samples.append(Sample(path=path, label=label))
    return samples


class FundusDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        transform: Callable | None = None,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, sample.label, sample.path.name
