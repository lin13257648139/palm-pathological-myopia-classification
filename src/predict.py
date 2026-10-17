from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.data import CLASS_NAMES
from src.model import IMAGE_SIZE, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict pathological myopia for one image.")
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/checkpoints/best_model.pth")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    image_size = int(checkpoint.get("image_size", IMAGE_SIZE))
    class_names = checkpoint.get("class_names", CLASS_NAMES)

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with Image.open(args.image) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()

    predicted_index = int(probabilities.argmax().item())
    print(f"prediction: {class_names[predicted_index]}")
    print(f"confidence: {probabilities[predicted_index].item():.4f}")
    print(f"pm_probability: {probabilities[1].item():.4f}")
    print("Note: research/education demo only; not a medical diagnosis.")


if __name__ == "__main__":
    main()
