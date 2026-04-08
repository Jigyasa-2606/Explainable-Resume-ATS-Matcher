import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import shap


def _top_feature_contributions(
    shap_values: np.ndarray, feature_names: List[str], top_k: int = 10
) -> Dict[str, List[Dict[str, float]]]:
    values = shap_values.reshape(-1)
    pairs = list(zip(feature_names, values))
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    positives = [{"feature": f, "contribution": float(v)} for f, v in sorted_pairs if v > 0][:top_k]
    negatives = [{"feature": f, "contribution": float(v)} for f, v in sorted(pairs, key=lambda x: x[1]) if v < 0][:top_k]
    return {"positive": positives, "negative": negatives}


def explain_single_prediction(
    model_path: Path,
    feature_names_path: Path,
    x_row,
    top_k: int = 10,
) -> Dict[str, object]:
    model = joblib.load(model_path)
    feature_names = json.loads(feature_names_path.read_text(encoding="utf-8"))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_row)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    contrib = _top_feature_contributions(np.array(shap_values), feature_names, top_k=top_k)
    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = float(base_value.reshape(-1)[0])
    else:
        base_value = float(base_value)

    return {"base_value": base_value, "positive": contrib["positive"], "negative": contrib["negative"]}


def explain_single_prediction_loaded(
    model,
    feature_names: List[str],
    x_row,
    top_k: int = 10,
) -> Dict[str, object]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_row)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    contrib = _top_feature_contributions(np.array(shap_values), feature_names, top_k=top_k)
    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = float(base_value.reshape(-1)[0])
    else:
        base_value = float(base_value)
    return {"base_value": base_value, "positive": contrib["positive"], "negative": contrib["negative"]}


def load_feature_names(feature_names_path: Path) -> List[str]:
    return json.loads(feature_names_path.read_text(encoding="utf-8"))
