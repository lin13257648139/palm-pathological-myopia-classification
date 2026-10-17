from __future__ import annotations

import torch.nn as nn
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3


MODEL_NAME = "efficientnet_b3"
IMAGE_SIZE = 300


def build_model(num_classes: int = 2, pretrained: bool = True):
    weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
    model = efficientnet_b3(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model
