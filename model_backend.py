#!/usr/bin/env python3

import argparse
from dataclasses import dataclass

import numpy as np


def parse_device_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Execution backend to use for model training and inference.",
    )
    return parser.parse_args()


def to_numpy_array(values):
    if isinstance(values, np.ndarray):
        return values

    if hasattr(values, "to_numpy"):
        return values.to_numpy()

    if hasattr(values, "get"):
        return values.get()

    return np.asarray(values)


@dataclass
class SimplePipeline:
    scaler: object
    estimator: object
    backend_name: str

    def fit(self, X, y):
        print("Scaling training features...")
        X_scaled = self.scaler.fit_transform(X)
        print("Finished scaling training features.")
        print("Fitting model...")
        self.estimator.fit(X_scaled, y)
        print("Finished fitting model.")
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        preds = self.estimator.predict(X_scaled)
        return to_numpy_array(preds)


def resolve_logistic_regression_backend(device):
    if device == "cpu":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        return SimplePipeline(
            scaler=StandardScaler(),
            estimator=LogisticRegression(
                max_iter=4252,
                random_state=2026,
                solver="saga",
                verbose=1,
            ),
            backend_name="scikit-learn (CPU)",
        )

    try:
        from cuml.linear_model import LogisticRegression
        from cuml.preprocessing import StandardScaler

        return SimplePipeline(
            scaler=StandardScaler(output_type="numpy"),
            estimator=LogisticRegression(
                max_iter=4252,
                solver="qn",
                output_type="numpy",
            ),
            backend_name="RAPIDS cuML (GPU)",
        )
    except ImportError:
        if device == "gpu":
            raise RuntimeError(
                "GPU execution was requested, but RAPIDS cuML is not installed. "
                "On Windows, RAPIDS is supported through WSL2 rather than a native "
                "Windows Python environment."
            )

        return resolve_logistic_regression_backend("cpu")


def resolve_linear_svm_backend(device):
    if device == "cpu":
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC

        return SimplePipeline(
            scaler=StandardScaler(),
            estimator=LinearSVC(
                max_iter=4252,
                random_state=2026,
                verbose=1,
            ),
            backend_name="scikit-learn (CPU)",
        )

    try:
        from cuml.preprocessing import StandardScaler
        from cuml.svm import LinearSVC

        return SimplePipeline(
            scaler=StandardScaler(output_type="numpy"),
            estimator=LinearSVC(
                max_iter=4252,
                output_type="numpy",
            ),
            backend_name="RAPIDS cuML (GPU)",
        )
    except ImportError:
        if device == "gpu":
            raise RuntimeError(
                "GPU execution was requested, but RAPIDS cuML is not installed. "
                "On Windows, RAPIDS is supported through WSL2 rather than a native "
                "Windows Python environment."
            )

        return resolve_linear_svm_backend("cpu")


def resolve_decision_tree_backend(device):
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    if device == "gpu":
        raise RuntimeError(
            "This repository's decision tree script still uses scikit-learn's "
            "DecisionTreeClassifier, which runs on CPU. RAPIDS exposes GPU "
            "alternatives for some classifiers, but not this exact training setup."
        )

    return SimplePipeline(
        scaler=StandardScaler(),
        estimator=DecisionTreeClassifier(
            max_depth=12,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=2026,
        ),
        backend_name="scikit-learn (CPU)",
    )
