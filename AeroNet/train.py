from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "drone_bird_feature_lr.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: list[Path], labels: np.ndarray, image_size: int) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image), int(self.labels[idx])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a bird-vs-drone classifier.")
    parser.add_argument("--image-size", type=int, default=160, help="Resize images to N x N before feature extraction.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--batch-size", type=int, default=64, help="Feature extraction batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def list_samples(dataset_dir: Path) -> tuple[list[Path], list[int], list[str]]:
    class_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
    class_names = [p.name for p in class_dirs]
    paths: list[Path] = []
    labels: list[int] = []

    for label_idx, class_dir in enumerate(class_dirs):
        for image_path in sorted(p for p in class_dir.iterdir() if p.is_file()):
            paths.append(image_path)
            labels.append(label_idx)

    return paths, labels, class_names


def stratified_split(
    paths: list[Path], labels: list[int], train_ratio: float, seed: int
) -> tuple[list[Path], list[Path], np.ndarray, np.ndarray]:
    grouped: dict[int, list[Path]] = {}
    for path, label in zip(paths, labels):
        grouped.setdefault(label, []).append(path)

    rng = random.Random(seed)
    train_paths: list[Path] = []
    val_paths: list[Path] = []
    train_labels: list[int] = []
    val_labels: list[int] = []

    for label, class_paths in grouped.items():
        current = class_paths[:]
        rng.shuffle(current)
        cutoff = int(len(current) * train_ratio)
        train_chunk = current[:cutoff]
        val_chunk = current[cutoff:]
        train_paths.extend(train_chunk)
        val_paths.extend(val_chunk)
        train_labels.extend([label] * len(train_chunk))
        val_labels.extend([label] * len(val_chunk))

    return train_paths, val_paths, np.array(train_labels), np.array(val_labels)


def build_feature_extractor(device: torch.device) -> torch.nn.Module:
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    backbone.fc = torch.nn.Identity()
    backbone.eval()
    backbone.to(device)
    return backbone


def extract_features(
    image_paths: list[Path],
    labels: np.ndarray,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> np.ndarray:
    dataset = ImagePathDataset(image_paths, labels, image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    extractor = build_feature_extractor(device)
    output: list[np.ndarray] = []

    with torch.no_grad():
        for images, _ in loader:
            features = extractor(images.to(device))
            output.append(features.cpu().numpy())

    return np.concatenate(output, axis=0)


def class_counts(labels: np.ndarray, class_names: list[str]) -> dict[str, int]:
    return {class_names[idx]: int((labels == idx).sum()) for idx in range(len(class_names))}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    paths, labels, class_names = list_samples(DATASET_DIR)
    train_paths, val_paths, y_train, y_val = stratified_split(paths, labels, args.train_ratio, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Extracting train features from {len(train_paths)} images on {device}...")
    x_train = extract_features(
        train_paths, y_train, args.image_size, args.batch_size, args.num_workers, device
    )
    print(f"Extracting validation features from {len(val_paths)} images...")
    x_val = extract_features(
        val_paths, y_val, args.image_size, args.batch_size, args.num_workers, device
    )

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=args.seed,
    )
    classifier.fit(x_train, y_train)

    val_probs = classifier.predict_proba(x_val)
    val_preds = np.argmax(val_probs, axis=1)
    val_accuracy = float(accuracy_score(y_val, val_preds))
    cm = confusion_matrix(y_val, val_preds).tolist()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as fh:
        pickle.dump(
            {
                "classifier": classifier,
                "class_names": class_names,
                "image_size": args.image_size,
            },
            fh,
        )

    metrics = {
        "train_counts": class_counts(y_train, class_names),
        "val_counts": class_counts(y_val, class_names),
        "class_names": class_names,
        "validation_accuracy": val_accuracy,
        "confusion_matrix": cm,
        "image_size": args.image_size,
        "feature_backbone": "resnet18_imagenet",
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
