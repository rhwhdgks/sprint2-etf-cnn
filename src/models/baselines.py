from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_linear_model(train_x: np.ndarray, train_y: np.ndarray, label_mode: str, max_iter: int):
    if label_mode == "classification":
        model = LogisticRegression(max_iter=max_iter, solver="lbfgs")
    elif label_mode == "regression":
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )
    else:
        raise ValueError(f"unsupported label_mode: {label_mode}")
    model.fit(train_x, train_y)
    return model


def predict_linear_model(model, features: np.ndarray, label_mode: str) -> tuple[np.ndarray, np.ndarray | None]:
    if label_mode == "classification":
        scores = model.predict_proba(features)[:, 1]
        confidence = np.abs(scores - 0.5) * 2.0
        return scores.astype(float), confidence.astype(float)

    predictions = model.predict(features).astype(float)
    return predictions, None
