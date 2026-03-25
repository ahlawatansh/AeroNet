from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFile
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "artifacts" / "drone_bird_feature_lr.pkl"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_bundle(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    with Path(model_path).open("rb") as fh:
        return pickle.load(fh)


def build_feature_extractor(device: torch.device) -> torch.nn.Module:
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    backbone.fc = torch.nn.Identity()
    backbone.eval()
    backbone.to(device)
    return backbone


def preprocess(image: Image.Image, image_size: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transform(image.convert("RGB")).unsqueeze(0)


def predict_image(image: Image.Image, model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    bundle = load_bundle(model_path)
    classifier = bundle["classifier"]
    class_names = bundle["class_names"]
    image_size = bundle["image_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = build_feature_extractor(device)
    inputs = preprocess(image, image_size).to(device)

    with torch.no_grad():
        features = extractor(inputs).cpu().numpy()

    probs = classifier.predict_proba(features)[0]
    predicted_idx = int(np.argmax(probs))
    return {
        "label": class_names[predicted_idx],
        "confidence": float(probs[predicted_idx]),
        "probabilities": {
            class_name: float(probs[idx]) for idx, class_name in enumerate(class_names)
        },
    }


def predict_image_file(image_path: str | Path, model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    image = Image.open(image_path)
    return predict_image(image, model_path=model_path)
