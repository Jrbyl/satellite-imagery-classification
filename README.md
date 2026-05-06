# Satellite Imagery Classification

## Overview

This project focuses on classifying land cover types from satellite imagery. The goal is to build a machine learning model that can identify and segment different categories of land without requiring manual, in-person mapping of a location.

The target land cover classes are:

- Bareland
- Grass
- Pavement
- Road
- Trees
- Water
- Cropland
- Buildings

This type of classification is useful for urban planning, agricultural monitoring, land-use analysis, and climate-related studies. The motivation behind the project is to automate land type detection over large geographic areas in a scalable and efficient way.

## Objective

Given satellite imagery, the project aims to classify and detect multiple land cover categories within an image. Rather than physically surveying land, this system uses image-based analysis to determine the type of terrain or built environment present in a region.

## Why This Matters

Land cover classification from satellite imagery has several practical applications, including:

- Urban planning
- Agricultural monitoring
- Land-use analysis
- Climate and environmental studies

Automating this process can save time, reduce cost, and improve the ability to analyze large regions consistently.

## Why Machine Learning

Machine learning is well-suited for this problem because land cover classification is a pattern recognition task. Instead of building complex rule-based logic for every terrain type, a model can be trained to recognize visual patterns directly from labeled examples.

## Dataset

The dataset consists of satellite images obtained from public databases.

### Data Source

- Public satellite imagery databases

### Data Collection

The images are collected from public databases and stored locally for use in the machine learning pipeline. If local storage becomes too large, image URLs may be used directly as inputs instead.

### Features

The primary input features are:

- Pixel RGB values from satellite images

### Labels

The land cover labels used in this project are:

- Bareland
- Grass
- Pavement
- Road
- Tree
- Water
- Cropland
- Buildings

### Dataset Split

- Total samples: 525
- Training set: 70%
- Validation set: 20%
- Test set: 10%

## Model

The original project proposal selected **logistic regression** as the machine learning algorithm.

The reasoning given was that the data must be classified into one of several distinct classes, making it a classification task rather than a regression problem. The model is intended to classify image regions into categories such as grass, trees, pavement, and other land cover types.

The repository now also includes a **PyTorch + TorchVision semantic segmentation** training script, which is a better fit for the image-mask patch dataset and can run directly on CUDA GPUs when PyTorch is installed with GPU support.

## Expected Outcomes

The project aims to produce an efficient model capable of detecting a variety of land types from satellite imagery.

Expected outcomes include:

- Mean Intersection over Union (IoU) of **0.5** or greater
- High pixel-wise accuracy
- Visual outputs showing predicted segmentation masks overlaid on satellite images

These results would demonstrate the model’s usefulness for real-world land classification tasks.

## CPU and GPU Execution

The training scripts now accept a `--device` flag:

```bash
python train_model_logistic_regression.py --device auto
python train_model_logistic_regression.py --device gpu
python train_model_svm.py --device gpu
python train_model_decision_tree.py --device gpu
```

Behavior:

- `--device auto` uses a GPU backend when RAPIDS cuML is installed, otherwise it falls back to CPU.
- `--device gpu` requires RAPIDS cuML for the logistic regression, SVM, and decision-tree-style GPU path.
- The decision tree GPU path uses `cuml.ensemble.RandomForestClassifier` with `n_estimators=1` as the closest supported GPU approximation to a single decision tree, so it is similar but not identical to scikit-learn's exact `DecisionTreeClassifier`.

Important for this machine:

- Native `scikit-learn` does not run these models on the GPU.
- RAPIDS cuML support on Windows is provided through **WSL2**, not a native Windows Python environment.
- If you want the GPU path, install RAPIDS inside a WSL2 Ubuntu environment and run the training scripts there with `--device gpu`.

## PyTorch Training

For GPU-native training with Torch/TorchVision, use:

```bash
python train_model_torchvision.py --device auto
python train_model_torchvision.py --device cuda --batch-size 4 --num-epochs 20
```

Notes:

- `train_model_torchvision.py` trains a `DeepLabV3-ResNet50` segmentation model on the paired TIFF image/mask patches.
- `--device auto` uses CUDA when available and otherwise falls back to CPU.
- The script reports train/validation/test loss, per-class IoU, mean IoU, and pixel accuracy.
- The best validation checkpoint is saved to `best_torchvision_model.pt` by default.

## Team Members

- Andrew Johnson
- Jon Beltzhoover
- Zach Lightly
- Dave Borucki
