#!/usr/bin/env python3

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import rasterio


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
    parser = argparse.ArgumentParser(description="Train a GPU gradient-boosted-trees classifier")
    parser.add_argument("--seed", type=int, default=2026, help="random seed")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="training device")
    parser.add_argument("--train-fraction", type=float, default=1.0,
                        help="fraction of the training split to use")
    parser.add_argument("--val-fraction", type=float, default=1.0,
                        help="fraction of the validation split to use")
    parser.add_argument("--test-fraction", type=float, default=1.0,
                        help="fraction of the test split to use")
    parser.add_argument("--max-pixels-per-tile", type=int, default=1000,
                        help="maximum number of sampled training pixels per tile")
    parser.add_argument("--n-estimators", type=int, default=200,
                        help="number of boosting rounds")
    parser.add_argument("--learning-rate", type=float, default=0.1,
                        help="gradient boosting learning rate")
    parser.add_argument("--max-depth", type=int, default=6,
                        help="maximum depth of each tree")
    parser.add_argument("--subsample", type=float, default=0.8,
                        help="row subsampling rate")
    parser.add_argument("--colsample-bytree", type=float, default=0.8,
                        help="feature subsampling rate")
    parser.add_argument("--checkpoint-path", type=str, default="gradient_boosted_trees_xgboost.json",
                        help="where to save the trained XGBoost model")
    return parser.parse_args()


def seed_everything(seed=2026):
    random.seed(seed)
    np.random.seed(seed)


def take_fraction(paths, fraction=1.0):
    fraction = min(max(fraction, 0.0), 1.0)
    n = max(1, int(len(paths) * fraction))
    return paths[:n]


def print_progress(prefix, current, total):
    pct = 100.0 * current / max(total, 1)
    print(f"{prefix}: {current}/{total} ({pct:.1f}%)")


def resolve_device(device_arg):
    if device_arg == "cpu":
        return "cpu"

    if device_arg == "cuda":
        return "cuda"

    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def build_model(args, device):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError(
            "This script requires xgboost. Install it with "
            "`venv\\Scripts\\python.exe -m pip install xgboost`."
        ) from exc

    params = dict(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        objective="multi:softmax",
        num_class=len(CLASS_NAMES),
        tree_method="hist",
        random_state=args.seed,
        verbosity=1,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        eval_metric="mlogloss",
    )

    if device == "cuda":
        params["device"] = "cuda"
    else:
        params["device"] = "cpu"

    return XGBClassifier(**params)


def load_multiband_image(path_img):
    path_img = str(path_img)
    ext = Path(path_img).suffix.lower()

    if ext in [".tif", ".tiff"]:
        with rasterio.open(path_img) as src:
            arr = src.read()
        arr = np.transpose(arr, (1, 2, 0))
    else:
        img = cv2.imread(path_img, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path_img}")
        if img.ndim == 2:
            img = img[..., None]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        arr = img

    return arr.astype(np.float32)


def load_label_tif(path_lbl):
    with rasterio.open(path_lbl) as src:
        y = src.read(1)
    return y.astype(np.int64)


def extract_features(img):
    img = img.astype(np.float32)
    _, _, c = img.shape

    means = []
    stds = []
    for ch in range(c):
        band = img[:, :, ch]
        mean = cv2.blur(band, (5, 5))
        mean_sq = cv2.blur(band * band, (5, 5))
        var = np.maximum(mean_sq - mean * mean, 0.0)
        std = np.sqrt(var)
        means.append(mean[..., None])
        stds.append(std[..., None])

    means = np.concatenate(means, axis=2)
    stds = np.concatenate(stds, axis=2)
    return np.concatenate([img, means, stds], axis=2)


def sample_training_pixels(image_paths, label_paths, max_pixels_per_tile=1000):
    X_list = []
    y_list = []

    for i, (imgp, lp) in enumerate(zip(image_paths, label_paths)):
        img = load_multiband_image(imgp)
        y = load_label_tif(lp)

        h, w = y.shape
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        feat = extract_features(img)
        X_tile = feat.reshape(-1, feat.shape[2])
        y_tile = y.reshape(-1)

        valid_idx = np.arange(len(y_tile))
        n_take = min(max_pixels_per_tile, len(valid_idx))
        take = np.random.choice(valid_idx, size=n_take, replace=False)

        X_list.append(X_tile[take])
        y_list.append(y_tile[take])

        if (i + 1) % 25 == 0 or i == 0 or (i + 1) == len(label_paths):
            print_progress("Training sample collection", i + 1, len(label_paths))

    if not X_list:
        raise RuntimeError("No training samples were collected. Check image/label paths.")

    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def predict_full_image(model, img):
    feat = extract_features(img)
    h, w, f = feat.shape
    X = feat.reshape(-1, f)
    y_pred = model.predict(X)
    return y_pred.reshape(h, w)


def evaluate_model(model, image_paths, label_paths, classes_eval=(1, 2, 3, 4, 5, 6, 7, 8)):
    intersections = {c: 0 for c in classes_eval}
    unions = {c: 0 for c in classes_eval}
    n_eval = len(classes_eval)
    c2i = {c: k for k, c in enumerate(classes_eval)}
    cm = np.zeros((n_eval, n_eval), dtype=np.int64)
    total_pixels_all = 0
    correct_pixels_all = 0

    for i, (imgp, lp) in enumerate(zip(image_paths, label_paths)):
        img = load_multiband_image(imgp)
        y_true = load_label_tif(lp)

        h, w = y_true.shape
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        y_pred = predict_full_image(model, img)
        total_pixels_all += y_true.size
        correct_pixels_all += int((y_pred == y_true).sum())

        for c in classes_eval:
            intersections[c] += int(np.logical_and(y_true == c, y_pred == c).sum())
            unions[c] += int(np.logical_or(y_true == c, y_pred == c).sum())

        mask = np.isin(y_true, classes_eval)
        yt = y_true[mask].astype(np.int64)
        yp = y_pred[mask].astype(np.int64)

        mask2 = np.isin(yp, classes_eval)
        yt = yt[mask2]
        yp = yp[mask2]

        if yt.size > 0:
            yt_i = np.vectorize(c2i.get)(yt)
            yp_i = np.vectorize(c2i.get)(yp)
            cm += np.bincount(
                yt_i * n_eval + yp_i,
                minlength=n_eval * n_eval,
            ).reshape(n_eval, n_eval)

        if (i + 1) % 25 == 0 or i == 0 or (i + 1) == len(label_paths):
            print_progress("Evaluation progress", i + 1, len(label_paths))

    ious = {}
    for c in classes_eval:
        ious[c] = np.nan if unions[c] == 0 else intersections[c] / unions[c]

    miou = np.nanmean(list(ious.values()))
    acc = correct_pixels_all / (total_pixels_all + 1e-12)
    return ious, miou, acc, cm


def print_confusion_matrices(cm):
    np.set_printoptions(suppress=True)
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
        row = name[:10].ljust(12) + " ".join([f"{cm_row_pct[i_row, j]:9.2f}%" for j in range(cm.shape[1])])
        print(row)

    total = float(cm.sum())
    cm_global_pct = (cm / total * 100.0) if total > 0 else np.zeros_like(cm, dtype=np.float64)

    print("\nConfusion Matrix (% of ALL evaluated pixels) | sums to 100% over all cells:")
    print(header)
    for i_row, name in enumerate(class_names):
        row = name[:10].ljust(12) + " ".join([f"{cm_global_pct[i_row, j]:9.2f}%" for j in range(cm.shape[1])])
        print(row)


def print_split_metrics(split_name, ious, miou, acc):
    print(f"\n{split_name} per-class IoU:")
    for c in range(1, 9):
        print(f"  class {c}: {ious[c]:.4f}")
    print(f"{split_name} mIoU (classes 1-8): {miou:.4f}")
    print(f"{split_name} overall pixel accuracy: {acc:.4f}")


def main():
    args = get_args()
    seed_everything(args.seed)

    device = resolve_device(args.device)
    print(f"Requested device: {args.device}")
    print(f"XGBoost device: {device}")

    root = Path("C:/satellite-imagery-classification/data/usa_europe_patches_512")

    train_image_paths = take_fraction(sorted((root / "train" / "images").glob("*.tif")), args.train_fraction)
    train_label_paths = take_fraction(sorted((root / "train" / "labels").glob("*.tif")), args.train_fraction)
    val_image_paths = take_fraction(sorted((root / "val" / "images").glob("*.tif")), args.val_fraction)
    val_label_paths = take_fraction(sorted((root / "val" / "labels").glob("*.tif")), args.val_fraction)
    test_image_paths = take_fraction(sorted((root / "test" / "images").glob("*.tif")), args.test_fraction)
    test_label_paths = take_fraction(sorted((root / "test" / "labels").glob("*.tif")), args.test_fraction)

    print("Train tiles:", len(train_label_paths))
    print("Val tiles:  ", len(val_label_paths))
    print("Test tiles: ", len(test_label_paths))

    print("\nCollecting training samples...")
    X_train, y_train = sample_training_pixels(
        image_paths=train_image_paths,
        label_paths=train_label_paths,
        max_pixels_per_tile=args.max_pixels_per_tile,
    )
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    unique, counts = np.unique(y_train, return_counts=True)
    print("\nTraining sample class counts:")
    for u, c in zip(unique, counts):
        print(f"  class {u}: {c}")

    model = build_model(args, device)
    print("\nTraining gradient boosted trees...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train)],
        verbose=True,
    )

    model.save_model(args.checkpoint_path)
    print(f"\nSaved XGBoost model to: {args.checkpoint_path}")

    print("\nEvaluating on validation set...")
    val_ious, val_miou, val_acc, val_cm = evaluate_model(model, val_image_paths, val_label_paths)
    print_split_metrics("Validation", val_ious, val_miou, val_acc)
    print_confusion_matrices(val_cm)

    print("\nEvaluating on test set...")
    test_ious, test_miou, test_acc, test_cm = evaluate_model(model, test_image_paths, test_label_paths)
    print_split_metrics("Test", test_ious, test_miou, test_acc)
    print_confusion_matrices(test_cm)


if __name__ == "__main__":
    main()
