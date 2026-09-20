"""Nested session-pair evaluation with validation-selected ridge strength and sign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..llama_continuous_whole_model.analyze_continuous import load_traces, trace_windows


def accuracy(scores: np.ndarray, labels: np.ndarray, sign: int = 1) -> float:
    prediction = np.where(sign * scores >= 0, 1.0, -1.0)
    return float(np.mean([np.mean(prediction[labels == label] == label) for label in (-1, 1)]))


def ridge_scores(x_train, y_train, x_test, strength):
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    train = np.column_stack((np.ones(len(x_train)), (x_train - mean) / scale))
    test = np.column_stack((np.ones(len(x_test)), (x_test - mean) / scale))
    penalty = np.eye(train.shape[1]) * strength
    penalty[0, 0] = 0.0
    return test @ np.linalg.solve(train.T @ train + penalty, train.T @ y_train)


def evaluate_features(x, y, groups, strengths=(0.1, 1.0, 10.0, 100.0)) -> dict:
    """Hold complete pairs out of training, normalization and sign selection."""
    unique = sorted(np.unique(groups).tolist())
    if len(unique) < 4:
        raise ValueError("nested evaluation requires at least four session pairs")
    for group in unique:
        if set(np.unique(y[groups == group])) != {-1.0, 1.0}:
            raise ValueError("every pair must contain both labels")
    folds = []
    for held in unique:
        training_groups = [group for group in unique if group != held]
        choices = []
        for strength in strengths:
            inner_scores, inner_labels = [], []
            for validation_group in training_groups:
                fit = (groups != held) & (groups != validation_group)
                validation = groups == validation_group
                inner_scores.append(ridge_scores(x[fit], y[fit], x[validation], strength))
                inner_labels.append(y[validation])
            scores = np.concatenate(inner_scores)
            labels = np.concatenate(inner_labels)
            for sign in (1, -1):
                choices.append((accuracy(scores, labels, sign), strength, sign))
        # Ties keep earlier choices: smaller penalty, then the ordinary sign.
        selected = max(choices, key=lambda item: item[0])
        fit, test = groups != held, groups == held
        scores = ridge_scores(x[fit], y[fit], x[test], selected[1])
        folds.append({
            "held_pair": held,
            "training_pairs": training_groups,
            "validation_accuracy": selected[0],
            "selected_ridge_strength": selected[1],
            "selected_sign": selected[2],
            "balanced_accuracy": accuracy(scores, y[test], selected[2]),
            "unflipped_balanced_accuracy": accuracy(scores, y[test]),
            "test_windows": int(test.sum()),
        })
    result = float(np.mean([fold["balanced_accuracy"] for fold in folds]))
    return {
        "balanced_accuracy": result,
        "folds": folds,
        "selection": "ridge strength and sign chosen only on inner held-out session pairs",
        "outer_split": "one matched session pair held out",
        "pair_count": len(unique),
        "finite_sample_orientation_diagnostic": max(result, 1 - result),
        "diagnostic_is_not_an_independent_test": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizons-ms", default="5,10,20,50,100")
    args = parser.parse_args()
    traces = load_traces(args.root, "inference") + load_traces(args.root, "training")
    if not all(trace.health_ok and np.isfinite(trace.values).all() for trace in traces):
        raise ValueError("unhealthy trace in detector input")
    sessions = {
        role: sorted({trace.session_id for trace in traces if trace.process == role})
        for role in ("inference", "training")
    }
    # Match the capture runner's iN/tN naming explicitly, never by label statistics.
    pairs = {}
    for role, names in sessions.items():
        prefix = "i" if role == "inference" else "t"
        for name in names:
            if not name.startswith(prefix) or not name[1:].isdigit():
                raise ValueError(f"expected matched session names iN/tN, got {name!r}")
            pairs[(role, name)] = int(name[1:])
    results = {}
    for horizon in map(float, args.horizons_ms.split(",")):
        x, y, groups = [], [], []
        for trace in traces:
            features = trace_windows(trace, horizon)
            if not len(features):
                raise ValueError(f"trace too short for {horizon} ms")
            x.append(features)
            y.append(np.full(len(features), -1.0 if trace.process == "inference" else 1.0))
            groups.append(np.full(len(features), pairs[(trace.process, trace.session_id)]))
        results[str(horizon)] = evaluate_features(
            np.concatenate(x), np.concatenate(y), np.concatenate(groups)
        )
    payload = {"root": str(args.root), "attacker_input": "ADC features only", "horizons": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({h: r["balanced_accuracy"] for h, r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
