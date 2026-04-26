from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable

from resume_analyzer import (
    analyze_sections,
    ats_score,
    compare_skills,
    extract_scoring_features,
    final_score,
    learning_to_rank_scores,
    semantic_similarity,
)
from train_match_model import normalize_targets


GROUP_COLUMNS = ["query_id", "resume_id", "group_id", "candidate_group"]


def read_rows(csv_path: Path, limit: int | None = None) -> list[dict[str, str]]:
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


def attach_relevance(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    raw_targets = [float(row["match_score"]) for row in rows]
    targets, _ = normalize_targets(raw_targets)
    return [
        {
            "resume": row["resume"],
            "job_description": row["job_description"],
            "relevance": target,
            **{column: row.get(column, "") for column in GROUP_COLUMNS},
        }
        for row, target in zip(rows, targets)
    ]


def build_groups(
    rows: list[dict[str, object]],
    group_size: int,
    random_state: int,
) -> tuple[list[list[dict[str, object]]], str]:
    for column in GROUP_COLUMNS:
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            value = str(row.get(column, "")).strip()
            if value:
                grouped[value].append(row)
        groups = [items for items in grouped.values() if len(items) >= 2]
        if groups:
            return groups, column

    shuffled = list(rows)
    random.Random(random_state).shuffle(shuffled)
    groups = [
        shuffled[index : index + group_size]
        for index in range(0, len(shuffled), group_size)
        if len(shuffled[index : index + group_size]) >= 2
    ]
    return groups, "synthetic_chunks"


def score_rule_based(row: dict[str, object]) -> float:
    resume = str(row["resume"])
    job = str(row["job_description"])
    semantic = semantic_similarity(resume, job, use_transformer=False)
    skills = compare_skills(resume, job)
    sections = analyze_sections(resume)
    ats, _ = ats_score(resume, job, skills)
    return float(final_score(semantic, skills.score, sections, ats))


def score_sbert_formula(row: dict[str, object]) -> float:
    resume = str(row["resume"])
    job = str(row["job_description"])
    semantic = semantic_similarity(resume, job, use_transformer=True)
    skills = compare_skills(resume, job)
    sections = analyze_sections(resume)
    ats, _ = ats_score(resume, job, skills)
    return float(final_score(semantic, skills.score, sections, ats))


def score_ltr_group(group: list[dict[str, object]]) -> list[float]:
    features = [
        extract_scoring_features(str(row["resume"]), str(row["job_description"]), use_transformer=False)
        for row in group
    ]
    ltr_scores = learning_to_rank_scores(features)
    if ltr_scores is not None:
        return [float(score) for score in ltr_scores]

    return [score_rule_based(row) for row in group]


def relevance_grade(relevance: float) -> float:
    # Dataset labels are normalized to 0-100; NDCG gains are more stable on a 0-5 grade.
    return relevance / 20.0


def dcg_at_k(relevances: list[float], k: int) -> float:
    return sum((2 ** relevance_grade(rel) - 1) / math.log2(index + 2) for index, rel in enumerate(relevances[:k]))


def ndcg_at_k(relevances: list[float], scores: list[float], k: int) -> float:
    ranked = [rel for _, rel in sorted(zip(scores, relevances), reverse=True)]
    ideal = sorted(relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(ranked, k) / ideal_dcg


def precision_at_k(relevances: list[float], scores: list[float], k: int, threshold: float) -> float:
    ranked = [rel for _, rel in sorted(zip(scores, relevances), reverse=True)[:k]]
    return sum(1 for rel in ranked if rel >= threshold) / max(len(ranked), 1)


def recall_at_k(relevances: list[float], scores: list[float], k: int, threshold: float) -> float:
    relevant_total = sum(1 for rel in relevances if rel >= threshold)
    if relevant_total == 0:
        return 0.0
    ranked = [rel for _, rel in sorted(zip(scores, relevances), reverse=True)[:k]]
    retrieved_relevant = sum(1 for rel in ranked if rel >= threshold)
    return retrieved_relevant / relevant_total


def mean_metrics(
    groups: list[list[dict[str, object]]],
    group_scores: Iterable[list[float]],
    k_values: list[int],
    relevant_threshold: float,
) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for group, scores in zip(groups, group_scores):
        relevances = [float(row["relevance"]) for row in group]
        for k in k_values:
            effective_k = min(k, len(group))
            buckets[f"ndcg@{k}"].append(ndcg_at_k(relevances, scores, effective_k))
            buckets[f"precision@{k}"].append(precision_at_k(relevances, scores, effective_k, relevant_threshold))
            buckets[f"recall@{k}"].append(recall_at_k(relevances, scores, effective_k, relevant_threshold))

    return {metric: round(mean(values), 4) if values else 0.0 for metric, values in buckets.items()}


def evaluate(
    dataset: Path,
    limit: int | None,
    group_size: int,
    k_values: list[int],
    relevant_threshold: float,
    random_state: int,
    include_sbert: bool,
) -> dict[str, object]:
    rows = attach_relevance(read_rows(dataset, limit))
    groups, group_mode = build_groups(rows, group_size, random_state)

    method_scores: dict[str, list[list[float]]] = {
        "rule_formula": [[score_rule_based(row) for row in group] for group in groups],
        "ltr_ranker": [score_ltr_group(group) for group in groups],
    }
    if include_sbert:
        method_scores["sbert_formula"] = [[score_sbert_formula(row) for row in group] for group in groups]

    method_metrics = {
        method: mean_metrics(groups, scores, k_values, relevant_threshold)
        for method, scores in method_scores.items()
    }
    baseline = method_metrics["rule_formula"]
    improvements = {
        method: {
            metric: round(value - baseline.get(metric, 0.0), 4)
            for metric, value in metrics.items()
        }
        for method, metrics in method_metrics.items()
        if method != "rule_formula"
    }

    return {
        "dataset": str(dataset),
        "rows": len(rows),
        "groups": len(groups),
        "group_mode": group_mode,
        "group_size": group_size,
        "relevant_threshold": relevant_threshold,
        "k_values": k_values,
        "transformer_requested": include_sbert,
        "transformer_env_enabled": os.getenv("USE_TRANSFORMER_EMBEDDINGS", "").lower() in {"1", "true", "yes"},
        "transformer_available": importlib.util.find_spec("sentence_transformers") is not None,
        "metrics": method_metrics,
        "improvements_vs_rule_formula": improvements,
    }


def print_report(result: dict[str, object]) -> None:
    print("Ranking Evaluation")
    print(f"Dataset: {result['dataset']}")
    print(f"Rows: {result['rows']}")
    print(f"Groups: {result['groups']} ({result['group_mode']})")
    if result["group_mode"] == "synthetic_chunks":
        print("Note: dataset has no query/resume groups, so fixed-size shuffled groups are used as a benchmark proxy.")
    print(f"Relevant threshold: {result['relevant_threshold']}")
    if result["transformer_requested"]:
        print(f"SBERT available: {result['transformer_available']}")
        if not result["transformer_available"]:
            print("Note: SBERT is not installed, so sbert_formula uses the semantic fallback.")
    print()

    metrics: dict[str, dict[str, float]] = result["metrics"]  # type: ignore[assignment]
    for method, values in metrics.items():
        print(method)
        for metric, value in values.items():
            print(f"  {metric}: {value}")
        print()

    improvements: dict[str, dict[str, float]] = result["improvements_vs_rule_formula"]  # type: ignore[assignment]
    if improvements:
        print("Improvement vs rule_formula")
        for method, values in improvements.items():
            print(method)
            for metric, value in values.items():
                prefix = "+" if value > 0 else ""
                print(f"  {metric}: {prefix}{value}")


def parse_k_values(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ranking quality with NDCG@K, Precision@K, and Recall@K.")
    parser.add_argument("--dataset", type=Path, default=Path("resume_job_matching_dataset.csv"))
    parser.add_argument("--limit", type=int, default=500, help="Optional row limit. Use 0 for the full dataset.")
    parser.add_argument("--group-size", type=int, default=20)
    parser.add_argument("--k", default="5,10")
    parser.add_argument("--relevant-threshold", type=float, default=80.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--include-sbert", action="store_true", help="Also evaluate transformer/SBERT semantic formula.")
    args = parser.parse_args()

    result = evaluate(
        dataset=args.dataset,
        limit=None if args.limit == 0 else args.limit,
        group_size=args.group_size,
        k_values=parse_k_values(args.k),
        relevant_threshold=args.relevant_threshold,
        random_state=args.random_state,
        include_sbert=args.include_sbert,
    )
    print_report(result)


if __name__ == "__main__":
    main()
