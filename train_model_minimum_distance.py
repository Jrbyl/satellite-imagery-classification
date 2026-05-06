#!/usr/bin/env python3

import argparse
import random
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


CLASS_NAMES = [
    "Background",
    "Bareland",
    "Grass",
    "Pavement",
    "Road",
    "Tree",
    "Water",
    "Cropland",
    "Buildings",
]


def get_args():
    parser = argparse.ArgumentParser(description="Train a minimum-distance land-cover classifier")
    parser.add_argument("--seed", type=int, default=2026, help="random seed")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="training device")
    parser.add_argument("--train-fraction", type=float, default=1.0,
                        help="fraction of the training split to use")
    parser.add_argument("--val-fraction", type=float, default=1.0,
                        help="fraction of the validation split to use")
    parser.add_argument("--test-fraction", type=float, default=1.0,
                        help="fraction of the test split to use")
    parser.add_argument("--distance-chunk-size", type=int, default=262144,
                        help="number of pixels to score at once during inference")
    parser.add_argument("--checkpoint-path", type=str, default="minimum_distance_centroids.pt",
                        help="where to save the learned class centroids")
    return parser.parse_args()


def seed_everything(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def take_fraction(paths, fraction=1.0):
    fraction = min(max(fraction, 0.0), 1.0)
    n = max(1, int(len(paths) * fraction))
    return paths[:n]


def resolve_device(device_arg):
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this PyTorch install.")
        return torch.device("cuda")

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device("cpu")


class SatelliteSegmentationDataset(Dataset):
    def __init__(self, split_root, fraction=1.0):
        split_root = Path(split_root)
        image_dir = split_root / "images"
        label_dir = split_root / "labels"

        image_paths = sorted(image_dir.glob("*.tif"))
        if not image_paths:
            raise RuntimeError(f"No TIFF images found in {image_dir}")

        image_paths = take_fraction(image_paths, fraction)
        self.samples = []

        for image_path in image_paths:
            label_path = label_dir / image_path.name
            if label_path.exists():
                self.samples.append((image_path, label_path))

        if not self.samples:
            raise RuntimeError(f"No image/label pairs found under {split_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label_path = self.samples[idx]

        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)

        with rasterio.open(label_path) as src:
            mask = src.read(1).astype(np.int64)

        image = torch.from_numpy(image / 255.0)
        mask = torch.from_numpy(mask)
        return image, mask


def iterate_with_progress(loader, desc):
    if tqdm is not None:
        return tqdm(loader, desc=desc, unit="batch", leave=False)

    total_batches = len(loader)

    def generator():
        for batch_idx, batch in enumerate(loader, start=1):
            print(f"{desc}: batch {batch_idx}/{total_batches}", end="\r", flush=True)
            yield batch
        print(" " * 80, end="\r", flush=True)

    return generator()


def print_confusion_matrices(confusion_matrix):
    np.set_printoptions(suppress=True)

    cm = confusion_matrix[1:, 1:]
    class_names = CLASS_NAMES[1:]

    print("\nConfusion Matrix (raw pixel counts) | rows=true, cols=pred (classes 1-8 only):")
    header = "true\\pred".ljust(12) + " ".join([name[:10].rjust(10) for name in class_names])
    print(header)
    for i_row, name in enumerate(class_names):
        row = name[:10].ljust(12) + " ".join([str(int(cm[i_row, j])).rjust(10) for j in range(cm.shape[1])])
        print(row)

    row_sums = cm.sum(axis=1, keepdims=True).astype(np.float64)
    cm_row_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums != 0) * 100.0

    print("\nConfusion Matrix (% of TRUE class) | each row sums to 100%:")
    print(header)
    for i_row, name in enumerate(class_names):
        row = name[:10].ljust(12) + " ".join([f"{cm_row_pct[i_row, j]:9.2f}%"
                                             for j in range(cm.shape[1])])
        print(row)

    total = float(cm.sum())
    cm_global_pct = (cm / total * 100.0) if total > 0 else np.zeros_like(cm, dtype=np.float64)

    print("\nConfusion Matrix (% of ALL evaluated pixels) | sums to 100% over all cells:")
    print(header)
    for i_row, name in enumerate(class_names):
        row = name[:10].ljust(12) + " ".join([f"{cm_global_pct[i_row, j]:9.2f}%"
                                             for j in range(cm.shape[1])])
        print(row)


def summarize_metrics(confusion_matrix):
    intersection = np.diag(confusion_matrix).astype(np.float64)
    pred_area = confusion_matrix.sum(axis=0).astype(np.float64)
    true_area = confusion_matrix.sum(axis=1).astype(np.float64)
    union = pred_area + true_area - intersection

    ious = np.full(confusion_matrix.shape[0], np.nan, dtype=np.float64)
    valid = union > 0
    ious[valid] = intersection[valid] / union[valid]

    pixel_acc = intersection.sum() / max(confusion_matrix.sum(), 1)
    return ious, float(np.nanmean(ious[1:])), float(pixel_acc)


def print_split_metrics(split_name, ious, miou, pixel_acc):
    print(f"\n{split_name} mIoU (classes 1-8): {miou:.4f}")
    print(f"{split_name} overall pixel accuracy: {pixel_acc:.4f}")
    print(f"{split_name} per-class IoU:")
    for class_id in range(1, len(CLASS_NAMES)):
        print(f"  class {class_id} ({CLASS_NAMES[class_id]}): {ious[class_id]:.4f}")


def extract_features(images):
    # Match the CPU baseline features:
    # raw values + local 5x5 mean + local 5x5 std for each channel.
    images = images.to(torch.float32)
    padded = F.pad(images, (2, 2, 2, 2), mode="reflect")
    mean = F.avg_pool2d(padded, kernel_size=5, stride=1)
    mean_sq = F.avg_pool2d(padded * padded, kernel_size=5, stride=1)
    var = torch.clamp(mean_sq - mean * mean, min=0.0)
    std = torch.sqrt(var)
    return torch.cat([images, mean, std], dim=1)


def fit_minimum_distance_classifier(loader, device, num_classes):
    sums = None
    counts = torch.zeros(num_classes, dtype=torch.float64, device=device)

    progress = iterate_with_progress(loader, "Fit centroids")
    for images, masks in progress:
        images = images.to(device, non_blocking=True).to(torch.float32)
        masks = masks.to(device, non_blocking=True).to(torch.long)
        features = extract_features(images)

        bsz, channels, height, width = features.shape
        pixels = features.permute(0, 2, 3, 1).reshape(-1, channels).to(torch.float64)
        labels = masks.reshape(-1)

        if sums is None:
            sums = torch.zeros(num_classes, channels, dtype=torch.float64, device=device)

        sums.index_add_(0, labels, pixels)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float64))

        if tqdm is not None:
            progress.set_postfix(seen_pixels=int(counts.sum().item()))

    centroids = torch.zeros_like(sums)
    valid = counts > 0
    centroids[valid] = sums[valid] / counts[valid].unsqueeze(1)
    return centroids.to(torch.float32), counts


def predict_batch(images, centroids, chunk_size):
    features = extract_features(images)
    bsz, channels, height, width = features.shape
    pixels = features.permute(0, 2, 3, 1).reshape(-1, channels)
    preds = torch.empty(pixels.shape[0], dtype=torch.long, device=images.device)

    for start in range(0, pixels.shape[0], chunk_size):
        end = min(start + chunk_size, pixels.shape[0])
        chunk = pixels[start:end]
        dists = torch.cdist(chunk, centroids, p=2)
        preds[start:end] = dists.argmin(dim=1)

    return preds.reshape(bsz, height, width)


def evaluate_model(loader, centroids, device, num_classes, chunk_size, desc):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    progress = iterate_with_progress(loader, desc)
    with torch.no_grad():
        for images, masks in progress:
            images = images.to(device, non_blocking=True).to(torch.float32)
            masks = masks.to(device, non_blocking=True).to(torch.long)

            preds = predict_batch(images, centroids, chunk_size)

            preds_np = preds.detach().cpu().numpy().reshape(-1)
            targets_np = masks.detach().cpu().numpy().reshape(-1)

            valid = (targets_np >= 0) & (targets_np < num_classes)
            preds_np = preds_np[valid]
            targets_np = targets_np[valid]

            bincount = np.bincount(
                num_classes * targets_np + preds_np,
                minlength=num_classes * num_classes,
            )
            confusion_matrix += bincount.reshape(num_classes, num_classes)

            if tqdm is not None:
                batch_pixel_acc = (preds == masks).float().mean().item()
                progress.set_postfix(pix_acc=f"{batch_pixel_acc:.4f}")

    ious, miou, pixel_acc = summarize_metrics(confusion_matrix)
    return ious, miou, pixel_acc, confusion_matrix


def main():
    args = get_args()
    seed_everything(args.seed)

    root = Path("C:/satellite-imagery-classification/data/usa_europe_patches_512")
    train_dataset = SatelliteSegmentationDataset(root / "train", fraction=args.train_fraction)
    val_dataset = SatelliteSegmentationDataset(root / "val", fraction=args.val_fraction)
    test_dataset = SatelliteSegmentationDataset(root / "test", fraction=args.test_fraction)

    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")

    print("Train tiles:", len(train_dataset))
    print("Val tiles:  ", len(val_dataset))
    print("Test tiles: ", len(test_dataset))
    print("Feature channels per pixel: 9 (raw RGB + 5x5 mean RGB + 5x5 std RGB)")

    centroids, counts = fit_minimum_distance_classifier(
        loader=train_loader,
        device=device,
        num_classes=len(CLASS_NAMES),
    )

    print("\nTraining pixel counts by class:")
    for class_id, count in enumerate(counts.detach().cpu().tolist()):
        print(f"  class {class_id} ({CLASS_NAMES[class_id]}): {int(count)}")

    torch.save(
        {
            "centroids": centroids.detach().cpu(),
            "counts": counts.detach().cpu(),
            "class_names": CLASS_NAMES,
        },
        args.checkpoint_path,
    )
    print(f"\nSaved minimum-distance centroids to: {args.checkpoint_path}")

    train_ious, train_miou, train_acc, train_cm = evaluate_model(
        loader=train_loader,
        centroids=centroids,
        device=device,
        num_classes=len(CLASS_NAMES),
        chunk_size=args.distance_chunk_size,
        desc="Train eval",
    )
    print_split_metrics("Train", train_ious, train_miou, train_acc)

    val_ious, val_miou, val_acc, val_cm = evaluate_model(
        loader=val_loader,
        centroids=centroids,
        device=device,
        num_classes=len(CLASS_NAMES),
        chunk_size=args.distance_chunk_size,
        desc="Validation eval",
    )
    print_split_metrics("Validation", val_ious, val_miou, val_acc)
    print_confusion_matrices(val_cm)

    test_ious, test_miou, test_acc, test_cm = evaluate_model(
        loader=test_loader,
        centroids=centroids,
        device=device,
        num_classes=len(CLASS_NAMES),
        chunk_size=args.distance_chunk_size,
        desc="Test eval",
    )
    print_split_metrics("Test", test_ious, test_miou, test_acc)
    print_confusion_matrices(test_cm)


if __name__ == "__main__":
    main()
