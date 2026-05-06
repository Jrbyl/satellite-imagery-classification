#!/usr/bin/env python3

import argparse
import random
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import deeplabv3_resnet50

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
    parser = argparse.ArgumentParser(description="Train a TorchVision segmentation model")
    parser.add_argument("--seed", type=int, default=2026, help="random seed")
    parser.add_argument("--num-epochs", type=int, default=1, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda",
                        help="training device")
    parser.add_argument("--train-fraction", type=float, default=1,
                        help="fraction of the training split to use")
    parser.add_argument("--val-fraction", type=float, default=1,
                        help="fraction of the validation split to use")
    parser.add_argument("--test-fraction", type=float, default=1,
                        help="fraction of the test split to use")
    parser.add_argument("--checkpoint-path", type=str, default="best_torchvision_model.pt",
                        help="where to save the best model weights")
    return parser.parse_args()


def seed_everything(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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

        # Convert 0..255 imagery to CHW float tensors in the 0..1 range.
        image = torch.from_numpy(image / 255.0)
        mask = torch.from_numpy(mask)

        return image, mask


def build_model(num_classes):
    model = deeplabv3_resnet50(
        weights=None,
        weights_backbone=None,
        num_classes=num_classes,
    )
    return model


def update_metrics(logits, targets, num_classes, confusion_matrix):
    preds = logits.argmax(dim=1)
    preds = preds.detach().cpu().numpy().reshape(-1)
    targets = targets.detach().cpu().numpy().reshape(-1)

    valid = (targets >= 0) & (targets < num_classes)
    preds = preds[valid]
    targets = targets[valid]

    bincount = np.bincount(
        num_classes * targets + preds,
        minlength=num_classes * num_classes,
    )
    confusion_matrix += bincount.reshape(num_classes, num_classes)


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


def run_epoch(model, loader, device, optimizer, criterion, num_classes, training, desc):
    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    seen_samples = 0

    with torch.set_grad_enabled(training):
        progress = iterate_with_progress(loader, desc)
        for images, masks in progress:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            outputs = model(images)["out"]
            loss = criterion(outputs, masks)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            seen_samples += images.size(0)
            update_metrics(outputs, masks, num_classes, confusion_matrix)

            if tqdm is not None:
                preds = outputs.argmax(dim=1)
                batch_pixel_acc = (preds == masks).float().mean().item()
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    pix_acc=f"{batch_pixel_acc:.4f}",
                    seen=seen_samples,
                )

    epoch_loss = running_loss / max(len(loader.dataset), 1)
    ious, miou, pixel_acc = summarize_metrics(confusion_matrix)
    return epoch_loss, ious, miou, pixel_acc, confusion_matrix


def print_split_metrics(split_name, loss, ious, miou, pixel_acc):
    print(f"\n{split_name} loss: {loss:.4f}")
    print(f"{split_name} mIoU (classes 1-8): {miou:.4f}")
    print(f"{split_name} overall pixel accuracy: {pixel_acc:.4f}")
    print(f"{split_name} per-class IoU:")
    for class_id in range(1, len(CLASS_NAMES)):
        print(f"  class {class_id} ({CLASS_NAMES[class_id]}): {ious[class_id]:.4f}")


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

    model = build_model(num_classes=len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_miou = -1.0
    best_state = None

    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        train_loss, train_ious, train_miou, train_acc, train_cm = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
            num_classes=len(CLASS_NAMES),
            training=True,
            desc=f"Train {epoch}/{args.num_epochs}",
        )
        print_split_metrics("Train", train_loss, train_ious, train_miou, train_acc)

        val_loss, val_ious, val_miou, val_acc, val_cm = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
            num_classes=len(CLASS_NAMES),
            training=False,
            desc=f"Val {epoch}/{args.num_epochs}",
        )
        print_split_metrics("Validation", val_loss, val_ious, val_miou, val_acc)
        print_confusion_matrices(val_cm)

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            print(f"Saved new best model with validation mIoU {best_val_miou:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, args.checkpoint_path)
        print(f"\nBest checkpoint saved to: {args.checkpoint_path}")

    test_loss, test_ious, test_miou, test_acc, test_cm = run_epoch(
        model=model,
        loader=test_loader,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
        num_classes=len(CLASS_NAMES),
        training=False,
        desc="Test",
    )
    print_split_metrics("Test", test_loss, test_ious, test_miou, test_acc)
    print_confusion_matrices(test_cm)


if __name__ == "__main__":
    main()
