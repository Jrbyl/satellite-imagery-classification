#!/usr/bin/env python3

# Standard library imports
import os
import random
from pathlib import Path

# Third-party libraries
import cv2
import numpy as np
import rasterio

# Scikit-learn tools for building and training the SVM pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def seed_everything(seed=2026):
    # Set Python's random seed so random sampling is reproducible
    random.seed(seed)

    # Set NumPy's random seed so NumPy sampling is reproducible
    np.random.seed(seed)

def take_fraction(paths, fraction=0.25):
    n = max(1, int(len(paths) * fraction))
    return paths[:n]

def corresponding_image_path(lp, image_exts=(".tif", ".tiff", ".png", ".jpg", ".jpeg")):
    # Convert incoming label path to a Path object
    lp = Path(lp)

    # Break the full path into parts so we can replace "labels" with "images"
    parts = list(lp.parts)

    # If the path does not contain a "labels" folder, we cannot map it to an image
    if "labels" not in parts:
        return None

    # Find where "labels" appears in the path and replace it with "images"
    i = parts.index("labels")
    parts[i] = "images"
    img_path = Path(*parts)

    # First try using the same file extension as the label file
    img_same = img_path.with_suffix(lp.suffix)
    if img_same.exists():
        return str(img_same)

    # If that fails, try other possible image extensions
    img_base = img_path.with_suffix("")
    for ext in image_exts:
        cand = img_base.with_suffix(ext)
        if cand.exists():
            return str(cand)

    # Return None if no matching image file is found
    return None


def load_multiband_image(path_img):
    # Convert path to string in case a Path object was passed in
    path_img = str(path_img)

    # Read the file extension to decide how to load it
    ext = Path(path_img).suffix.lower()

    # Use rasterio for TIFF files because they may contain multiple bands
    if ext in [".tif", ".tiff"]:
        with rasterio.open(path_img) as src:
            arr = src.read()  # rasterio reads as (C, H, W)

        # Reorder to standard image shape (H, W, C)
        arr = np.transpose(arr, (1, 2, 0))

    else:
        # Use OpenCV for non-TIFF images
        img = cv2.imread(path_img, cv2.IMREAD_UNCHANGED)

        # Raise an error if the file could not be read
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path_img}")

        # If image is grayscale, add a channel dimension
        if img.ndim == 2:
            img = img[..., None]
        else:
            # OpenCV loads color images as BGR, so convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        arr = img

    # Convert to float32 for feature extraction and model input
    return arr.astype(np.float32)


def load_label_tif(path_lbl):
    # Load the label image using rasterio
    with rasterio.open(path_lbl) as src:
        # Read only the first band, since labels are stored as one class ID per pixel
        y = src.read(1)

    # Convert to int64 because labels are integer class IDs
    return y.astype(np.int64)


def extract_features(img):
    """
    Input:
        img: image array with shape (H, W, C)

    Output:
        feature array with shape (H, W, F)

    Features used for each pixel:
        1. Raw band values
        2. Local 5x5 mean for each band
        3. Local 5x5 standard deviation for each band
    """
    # Ensure image is float32 before doing math
    img = img.astype(np.float32)

    # Get image dimensions
    h, w, c = img.shape

    # Lists to store local mean and std feature maps
    means = []
    stds = []

    # Process each image channel independently
    for ch in range(c):
        # Extract one band/channel
        band = img[:, :, ch]

        # Compute local 5x5 mean using a box blur
        mean = cv2.blur(band, (5, 5))

        # Compute local mean of squared values
        mean_sq = cv2.blur(band * band, (5, 5))

        # Variance = E[x^2] - (E[x])^2
        var = np.maximum(mean_sq - mean * mean, 0.0)

        # Standard deviation is sqrt(variance)
        std = np.sqrt(var)

        # Add channel back so shapes match for concatenation later
        means.append(mean[..., None])
        stds.append(std[..., None])

    # Stack all mean maps into shape (H, W, C)
    means = np.concatenate(means, axis=2)

    # Stack all std maps into shape (H, W, C)
    stds = np.concatenate(stds, axis=2)

    # Final feature vector for each pixel:
    # [raw values, local means, local stds]
    feat = np.concatenate([img, means, stds], axis=2)

    return feat


def sample_training_pixels(label_paths, max_pixels_per_tile=1000, ignore_background=False):
    # Lists to collect sampled feature vectors and labels from all tiles
    X_list = []
    y_list = []

    # Loop over all training label files
    for i, lp in enumerate(label_paths):
        # Find the matching image file for this label file
        imgp = corresponding_image_path(lp)
        if imgp is None:
            print(f"Skipping missing image for label: {lp}")
            continue

        # Load image and corresponding label mask
        img = load_multiband_image(imgp)
        y = load_label_tif(lp)

        # Get label dimensions
        h, w = y.shape

        # If image and label dimensions do not match, resize the image to match labels
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        # Compute handcrafted features for every pixel in the image
        feat = extract_features(img)

        # Flatten image features to shape (num_pixels, num_features)
        X_tile = feat.reshape(-1, feat.shape[2])

        # Flatten labels to shape (num_pixels,)
        y_tile = y.reshape(-1)

        # Optionally remove background pixels (class 0) from training samples
        if ignore_background:
            valid_idx = np.where(y_tile > 0)[0]
        else:
            valid_idx = np.arange(len(y_tile))

        # If no valid pixels remain, skip this tile
        if len(valid_idx) == 0:
            continue

        # Sample up to max_pixels_per_tile random pixels from this tile
        n_take = min(max_pixels_per_tile, len(valid_idx))
        take = np.random.choice(valid_idx, size=n_take, replace=False)

        # Store sampled features and labels
        X_list.append(X_tile[take])
        y_list.append(y_tile[take])

        # Print occasional progress updates
        if (i + 1) % 100 == 0 or i == 0:
            print(f"Sampled training pixels from {i+1}/{len(label_paths)} tiles")

    # Safety check in case no samples were collected
    if len(X_list) == 0:
        raise RuntimeError("No training samples were collected. Check image/label paths.")

    # Combine sampled data from all tiles into one training matrix and label vector
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    return X, y


def predict_full_image(model, img):
    # Extract handcrafted features from the full image
    feat = extract_features(img)

    # Save the original image height, width, and number of features
    h, w, f = feat.shape

    # Flatten to shape (num_pixels, num_features) for scikit-learn prediction
    X = feat.reshape(-1, f)

    # Predict one class per pixel
    y_pred = model.predict(X)

    # Reshape predictions back to image shape
    return y_pred.reshape(h, w)


def evaluate_model(model, label_paths, classes_eval=(1, 2, 3, 4, 5, 6, 7, 8), max_tiles=None):
    # Store total intersections and unions for each class across the dataset
    intersections = {c: 0 for c in classes_eval}
    unions = {c: 0 for c in classes_eval}

    # Track overall pixel accuracy across all classes, including background
    total_pixels_all = 0
    correct_pixels_all = 0

    # Optionally evaluate only the first max_tiles files for faster testing
    use_paths = label_paths if max_tiles is None else label_paths[:max_tiles]

    # Loop through each evaluation tile
    for i, lp in enumerate(use_paths):
        # Find the corresponding image path
        imgp = corresponding_image_path(lp)
        if imgp is None:
            print(f"Skipping missing image for label: {lp}")
            continue

        # Load image and ground-truth label mask
        img = load_multiband_image(imgp)
        y_true = load_label_tif(lp)

        # Ensure image size matches label size
        h, w = y_true.shape
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        # Run the model to predict a class for every pixel
        y_pred = predict_full_image(model, img)

        # Update overall pixel accuracy counts
        total_pixels_all += y_true.size
        correct_pixels_all += int((y_pred == y_true).sum())

        # Update IoU counts for each evaluated class
        for c in classes_eval:
            intersections[c] += int(np.logical_and(y_true == c, y_pred == c).sum())
            unions[c] += int(np.logical_or(y_true == c, y_pred == c).sum())

        # Print occasional progress updates
        if (i + 1) % 25 == 0 or i == 0:
            print(f"Evaluated {i+1}/{len(use_paths)} tiles")

    # Convert intersection/union counts into final IoU values
    ious = {}
    for c in classes_eval:
        if unions[c] == 0:
            ious[c] = np.nan
        else:
            ious[c] = intersections[c] / unions[c]

    # Mean IoU across classes 1-8
    miou = np.nanmean(list(ious.values()))

    # Overall pixel accuracy, including background
    acc = correct_pixels_all / (total_pixels_all + 1e-12)

    return ious, miou, acc


def main():
    # Set random seeds so results are reproducible
    seed_everything(2026)

    # Define dataset locations
    train_root = Path("./data/usa_europe_patches_512/train")
    val_root   = Path("./data/usa_europe_patches_512/val")
    test_root  = Path("./data/usa_europe_patches_512/test")

    # Find all label files in each split
    train_label_paths = sorted(train_root.rglob("labels/*.tif"))
    val_label_paths   = sorted(val_root.rglob("labels/*.tif"))
    test_label_paths  = sorted(test_root.rglob("labels/*.tif"))

    # Keep only 25% of each split for quicker testing
    train_label_paths = take_fraction(train_label_paths, 0.25)
    val_label_paths = take_fraction(val_label_paths, 0.25)
    test_label_paths = take_fraction(test_label_paths, 0.25)

    # Print how many tiles were found in each split
    print("Train tiles:", len(train_label_paths))
    print("Val tiles:  ", len(val_label_paths))
    print("Test tiles: ", len(test_label_paths))

    # Sample training pixels from the training tiles
    print("\nCollecting training samples...")
    X_train, y_train = sample_training_pixels(
        train_label_paths,
        max_pixels_per_tile=1000,   # maximum number of pixels sampled from each tile
        ignore_background=False,    # set True if you want to exclude class 0 from training
    )

    # Print resulting training matrix shape
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    # Print sampled class distribution for debugging / imbalance inspection
    unique, counts = np.unique(y_train, return_counts=True)
    print("\nTraining sample class counts:")
    for u, c in zip(unique, counts):
        print(f"  class {u}: {c}")

    # Build the model pipeline:
    #   1. Standardize features
    #   2. Train a linear SVM classifier
    print("\nTraining SVM...")
    model = make_pipeline(
        StandardScaler(),
        LinearSVC(
            max_iter=3000,
            random_state=2026,
        )
    )

    # Fit the SVM on sampled training pixels
    model.fit(X_train, y_train)

    # Evaluate on validation split
    print("\nEvaluating on validation set...")
    val_ious, val_miou, val_acc = evaluate_model(
        model,
        val_label_paths,
        classes_eval=(1, 2, 3, 4, 5, 6, 7, 8),
        max_tiles=100,   # change to None to evaluate the full validation set
    )

    # Print validation IoU results
    print("\nValidation per-class IoU:")
    for c in range(1, 9):
        print(f"  class {c}: {val_ious[c]:.4f}")
    print(f"Validation mIoU (classes 1-8): {val_miou:.4f}")
    print(f"Validation overall pixel accuracy: {val_acc:.4f}")

    # Evaluate on test split
    print("\nEvaluating on test set...")
    test_ious, test_miou, test_acc = evaluate_model(
        model,
        test_label_paths,
        classes_eval=(1, 2, 3, 4, 5, 6, 7, 8),
        max_tiles=100,   # change to None to evaluate the full test set
    )

    # Print test IoU results
    print("\nTest per-class IoU:")
    for c in range(1, 9):
        print(f"  class {c}: {test_ious[c]:.4f}")
    print(f"Test mIoU (classes 1-8): {test_miou:.4f}")
    print(f"Test overall pixel accuracy: {test_acc:.4f}")


# Standard Python entry point
if __name__ == "__main__":
    main()