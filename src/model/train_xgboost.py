import argparse
import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


def train_xgboost_regressor(
    features_path: Path,
    labels_path: Path,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, float]:
    x = load_npz(features_path)
    y = np.load(labels_path)

    if y is None or len(y) == 0:
        raise ValueError("Label array is empty. Ensure match_score exists in your dataset.")

    y = np.asarray(y, dtype=np.float32)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    preds = np.clip(preds, 0, 100)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "n_features": int(x.shape[1]),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "xgb_regressor.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model to: {output_dir / 'xgb_regressor.joblib'}")
    print(f"Saved metrics to: {output_dir / 'metrics.json'}")
    print(json.dumps(metrics, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost regressor on Phase 2 features.")
    parser.add_argument("--features-path", type=Path, default=Path("data/features/X_features.npz"))
    parser.add_argument("--labels-path", type=Path, default=Path("data/features/y.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_xgboost_regressor(
        features_path=args.features_path,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
