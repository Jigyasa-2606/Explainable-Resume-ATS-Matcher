from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

from resume_analyzer import TRAINED_MODEL_PATH, extract_scoring_features


def read_training_rows(csv_path: Path, limit: int | None = None) -> list[dict[str, str]]:
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"resume", "job_description", "match_score"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                break
            if row.get("resume", "").strip() and row.get("job_description", "").strip() and row.get("match_score", "").strip():
                rows.append(row)
    return rows


def normalize_targets(raw_targets: list[float]) -> tuple[list[float], str]:
    max_target = max(raw_targets) if raw_targets else 0
    if max_target <= 5:
        return [target * 20 for target in raw_targets], "1-5_to_0-100"
    if max_target <= 10:
        return [target * 10 for target in raw_targets], "1-10_to_0-100"
    return raw_targets, "already_0-100"


def build_feature_matrix(rows: list[dict[str, str]]) -> tuple[list[list[float]], list[float], list[str], str]:
    feature_dicts = [
        extract_scoring_features(row["resume"], row["job_description"], use_transformer=False)
        for row in rows
    ]
    feature_names = sorted(feature_dicts[0])
    raw_targets = [float(row["match_score"]) for row in rows]
    targets, target_scale = normalize_targets(raw_targets)
    matrix = [[features[name] for name in feature_names] for features in feature_dicts]
    return matrix, targets, feature_names, target_scale


def build_pairwise_rank_data(
    features: list[list[float]],
    targets: list[float],
    max_pairs: int,
    min_score_gap: float,
    random_state: int,
) -> tuple[list[list[float]], list[int]]:
    rng = random.Random(random_state)
    candidate_indices = list(range(len(targets)))
    pair_features: list[list[float]] = []
    pair_targets: list[int] = []
    attempts = 0
    max_attempts = max_pairs * 20

    while len(pair_features) < max_pairs and attempts < max_attempts:
        attempts += 1
        left, right = rng.sample(candidate_indices, 2)
        score_gap = targets[left] - targets[right]
        if abs(score_gap) < min_score_gap:
            continue

        higher, lower = (left, right) if score_gap > 0 else (right, left)
        diff = [features[higher][idx] - features[lower][idx] for idx in range(len(features[higher]))]
        pair_features.append(diff)
        pair_targets.append(1)

        # Add the reverse comparison too so the classifier learns both directions.
        pair_features.append([-value for value in diff])
        pair_targets.append(0)

    return pair_features[:max_pairs], pair_targets[:max_pairs]


def train_model(
    dataset: Path,
    output: Path,
    limit: int | None,
    test_size: float,
    random_state: int,
    max_rank_pairs: int,
    min_rank_gap: float,
) -> dict[str, object]:
    try:
        import joblib  # type: ignore[import-not-found]
        from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-not-found]
        from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore[import-not-found]
        from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score  # type: ignore[import-not-found]
        from sklearn.model_selection import train_test_split  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit("Install dependencies first with: pip install -r requirements.txt") from exc

    rows = read_training_rows(dataset, limit)
    if len(rows) < 20:
        raise ValueError("Need at least 20 valid rows to train and evaluate the model.")

    features, targets, feature_names, target_scale = build_feature_matrix(rows)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=test_size,
        random_state=random_state,
    )

    model = RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    ranker = None
    rank_metrics = {"available": False, "pair_count": 0}
    pair_features, pair_targets = build_pairwise_rank_data(
        features,
        targets,
        max_pairs=max_rank_pairs,
        min_score_gap=min_rank_gap,
        random_state=random_state,
    )
    if len(set(pair_targets)) == 2 and len(pair_targets) >= 50:
        px_train, px_test, py_train, py_test = train_test_split(
            pair_features,
            pair_targets,
            test_size=test_size,
            random_state=random_state,
            stratify=pair_targets,
        )
        ranker = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            random_state=random_state,
        )
        ranker.fit(px_train, py_train)
        rank_predictions = ranker.predict(px_test)
        rank_metrics = {
            "available": True,
            "pair_count": len(pair_targets),
            "accuracy": round(float(accuracy_score(py_test, rank_predictions)), 3),
            "test_pairs": len(py_test),
            "min_score_gap": min_rank_gap,
        }

    metrics = {
        "mae": round(float(mean_absolute_error(y_test, predictions)), 3),
        "rmse": round(math.sqrt(float(mean_squared_error(y_test, predictions))), 3),
        "r2": round(float(r2_score(y_test, predictions)), 3),
        "test_rows": len(y_test),
    }

    artifact = {
        "model": model,
        "ranker": ranker,
        "feature_names": feature_names,
        "metrics": metrics,
        "rank_metrics": rank_metrics,
        "trained_rows": len(rows),
        "target_scale": target_scale,
        "dataset": str(dataset),
        "model_type": "regression_plus_pairwise_ranker",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a resume-job match scoring model.")
    parser.add_argument("--dataset", type=Path, default=Path("resume_job_matching_dataset.csv"))
    parser.add_argument("--output", type=Path, default=TRAINED_MODEL_PATH)
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick experiments.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-rank-pairs", type=int, default=50000)
    parser.add_argument("--min-rank-gap", type=float, default=10.0)
    args = parser.parse_args()

    artifact = train_model(
        args.dataset,
        args.output,
        args.limit,
        args.test_size,
        args.random_state,
        args.max_rank_pairs,
        args.min_rank_gap,
    )
    metrics = artifact["metrics"]
    rank_metrics = artifact["rank_metrics"]
    print(f"Saved trained model to: {args.output}")
    print(f"Rows used: {artifact['trained_rows']}")
    print(f"Target scale: {artifact['target_scale']}")
    print(f"MAE: {metrics['mae']}")
    print(f"RMSE: {metrics['rmse']}")
    print(f"R2: {metrics['r2']}")
    if rank_metrics["available"]:
        print(f"Ranker pairs: {rank_metrics['pair_count']}")
        print(f"Ranker accuracy: {rank_metrics['accuracy']}")
    else:
        print("Ranker was not trained; not enough score-separated pairs.")


if __name__ == "__main__":
    main()
