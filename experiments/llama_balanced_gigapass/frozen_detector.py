"""Freeze ADC-only attackers before an independent, predeclared physical test."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .audit_records import audit
from ..llama_continuous_whole_model.analyze_continuous import load_traces, trace_windows

HORIZONS = (5, 10, 20, 50, 100)


def source_hashes() -> dict:
    root = Path(__file__).resolve().parents[2]
    paths = [
        "experiments/llama_balanced_gigapass/" + name + ".py"
        for name in ("frozen_detector", "collect_pairs", "capture", "workload", "audit_records")
    ] + ["experiments/llama_continuous_whole_model/" + name + ".py"
         for name in ("cnn_detector", "analyze_continuous", "power_features")]
    return {name: digest(root / name) for name in paths}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def balanced_accuracy(prediction, labels) -> float:
    prediction, labels = np.asarray(prediction), np.asarray(labels)
    if prediction.shape != labels.shape or set(np.unique(labels)) != {0, 1}:
        raise ValueError("accuracy requires matching arrays containing both binary labels")
    return float(np.mean([np.mean(prediction[labels == label] == label) for label in (0, 1)]))


def predict(scores, decision: dict) -> np.ndarray:
    scores = np.asarray(scores)
    if not np.isfinite(scores).all() or decision["sign"] not in (-1, 1):
        raise ValueError("prediction requires finite scores and sign +1 or -1")
    if not np.isfinite(decision["threshold"]):
        raise ValueError("prediction threshold must be finite")
    return (decision["sign"] * scores >= decision["threshold"]).astype(np.int64)


def select_decision(validation_scores, validation_labels) -> dict:
    """No test labels or test scores are accepted here."""
    scores = np.asarray(validation_scores, dtype=float)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("validation scores must be nonempty and finite")
    choices = []
    for sign in (1, -1):
        oriented = sign * scores
        # Zero wins exact ties; include all-positive/all-negative controls.
        thresholds = [0.0, *np.quantile(oriented, np.linspace(0, 1, 33)).tolist(),
                      float(np.nextafter(oriented.min(), -np.inf)),
                      float(np.nextafter(oriented.max(), np.inf))]
        for threshold in thresholds:
            decision = {"sign": sign, "threshold": threshold}
            choices.append({**decision, "validation_accuracy": balanced_accuracy(
                predict(scores, decision), validation_labels)})
    return max(choices, key=lambda row: row["validation_accuracy"])


def auc(scores, labels) -> float:
    scores, labels = np.asarray(scores), np.asarray(labels)
    if len(scores) != len(labels) or not np.isfinite(scores).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError("AUC requires finite scores with both binary labels")
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    positive = int(np.sum(labels == 1))
    return float((ranks[labels == 1].sum() - positive * (positive + 1) / 2)
                 / (positive * np.sum(labels == 0)))


def partition(groups, seed: int) -> tuple[list[int], list[int]]:
    unique = np.unique(groups)
    if len(unique) < 8:
        raise ValueError("frozen evaluation requires at least eight development session pairs")
    shuffled = np.random.default_rng(seed).permutation(unique)
    validation = sorted(shuffled[:len(unique) // 3].tolist())
    training = sorted(set(unique.tolist()) - set(validation))
    return training, validation


def verify_bundle(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("status") != "frozen" or manifest.get("schema_version") != 1:
        raise ValueError("detector bundle is incomplete or has an unsupported schema")
    if manifest.get("source_sha256") != source_hashes():
        raise ValueError("detector, feature extraction or collection source changed since freezing")
    if set(manifest["horizons"]) != {str(h) for h in HORIZONS}:
        raise ValueError("frozen bundle must contain every predeclared horizon")
    for name, expected in manifest["file_sha256"].items():
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("detector bundle files must be direct children of its directory")
        if digest(root / name) != expected:
            raise ValueError(f"frozen detector file changed: {name}")
    referenced = {entry["ridge"]["file"] for entry in manifest["horizons"].values()}
    referenced.update(name for entry in manifest["horizons"].values() for name in entry["cnn"]["files"])
    if not referenced <= manifest["file_sha256"].keys():
        raise ValueError("a detector file is not covered by the frozen hash manifest")
    return manifest


def check_test_contract(manifest: dict, plan: dict) -> None:
    contract = manifest["test_plan"]
    for key in ("pairs", "captures_per_session", "seed", "attention_backend", "calibration"):
        if plan[key] != contract[key]:
            raise ValueError(f"test collection differs from frozen plan: {key}")
    test_seeds = {row["compute_seed"] for row in plan["plan"]}
    if test_seeds & set(manifest["development_compute_seeds"]):
        raise ValueError("test collection reuses development compute seeds")
    # Independently reconstruct the declared plan, not just its header.
    from .collect_pairs import collection_plan
    if plan["plan"] != collection_plan(contract["pairs"], contract["seed"]):
        raise ValueError("test session order or seeds differ from the declared plan")


def checked_traces(root: Path):
    result = audit(root)
    traces = load_traces(root, "inference") + load_traces(root, "training")
    return traces, result


def feature_data(traces, horizon):
    values, labels, groups = [], [], []
    for trace in traces:
        rows = trace_windows(trace, horizon)
        if not len(rows):
            raise ValueError("trace is shorter than the requested detector window")
        values.append(rows)
        labels.extend([int(trace.process == "training")] * len(rows))
        groups.extend([int(trace.session_id[1:])] * len(rows))
    return np.concatenate(values), np.asarray(labels), np.asarray(groups)


def ridge_fit(x, y, strength):
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-12] = 1.0
    design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    penalty = np.eye(design.shape[1]) * strength
    penalty[0, 0] = 0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ (2 * y - 1))
    return {"mean": mean, "scale": scale, "weights": weights}


def ridge_predict(state, x):
    return np.column_stack((np.ones(len(x)), (x - state["mean"]) / state["scale"])) @ state["weights"]


def cnn_scores(model, values, batch_size):
    import torch
    model.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            logits = model(torch.from_numpy(values[start:start + batch_size, None]).cuda())
            scores.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
    return np.concatenate(scores)


def train(args) -> None:
    import random
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from ..llama_continuous_whole_model.cnn_detector import PowerTraceCnn, make_windows

    if args.test_pairs < 4 or args.test_captures < 2 or args.epochs < 1 or args.ensemble_size < 1:
        raise ValueError("invalid collection or training counts")
    traces, checked = checked_traces(args.development_root)
    source_plan = json.loads((args.development_root / "collection_plan.json").read_text())
    groups = np.asarray([int(trace.session_id[1:]) for trace in traces])
    training_pairs, validation_pairs = partition(groups, args.seed)
    manifest = {
        "schema_version": 1, "status": "training", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_hashes(), "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "development_root": str(args.development_root.resolve()),
        "development_trace_sha256": checked["trace_sha256"],
        "development_compute_seeds": sorted({row["compute_seed"] for row in source_plan["plan"]}),
        "training_pairs": training_pairs, "validation_pairs": validation_pairs,
        "test_plan": {"pairs": args.test_pairs, "captures_per_session": args.test_captures,
                      "seed": args.test_seed, "attention_backend": source_plan["attention_backend"],
                      "calibration": source_plan["calibration"]},
        "protocol": {"seed": args.seed, "epochs": args.epochs, "cnn_ensemble_size": args.ensemble_size,
                     "cnn_channels": 32, "batch_size": 32, "learning_rate": 0.002,
                     "weight_decay": 0.001, "ridge_strengths": [0.1, 1, 10, 100],
                     "accuracy_target": 0.6, "family_wise_alpha": 0.05,
                     "primary_comparisons": len(HORIZONS) * 2, "bootstrap_replicates": 20000,
                     "bootstrap_unit": "independent matched session pair",
                     "bootstrap_note": "approximate percentile bounds; not a distribution-free guarantee",
                     "selection": "fixed training epochs; strength, direction and threshold use validation only"},
        "file_sha256": {}, "horizons": {},
    }
    from .collect_pairs import collection_plan
    check_test_contract(manifest, {**manifest["test_plan"],
                                  "plan": collection_plan(args.test_pairs, args.test_seed)})
    args.bundle.mkdir(parents=True, exist_ok=False)
    write_json(args.bundle / "manifest.json", manifest)
    for horizon in HORIZONS:
        x, y, groups = feature_data(traces, horizon)
        fit, val = np.isin(groups, training_pairs), np.isin(groups, validation_pairs)
        choices = []
        for strength in manifest["protocol"]["ridge_strengths"]:
            state = ridge_fit(x[fit], y[fit], strength)
            decision = select_decision(ridge_predict(state, x[val]), y[val])
            choices.append((decision["validation_accuracy"], strength, state, decision))
        _, strength, state, decision = max(choices, key=lambda item: item[0])
        filename = f"ridge_{horizon}ms.npz"
        np.savez(args.bundle / filename, **state)
        manifest["file_sha256"][filename] = digest(args.bundle / filename)
        ridge = {"file": filename, "strength": strength, "decision": decision}

        windows = make_windows(traces, horizon)
        groups = np.asarray([int(session[1:]) for session in windows.sessions])
        fit, val = np.isin(groups, training_pairs), np.isin(groups, validation_pairs)
        mean, std = (float(windows.values[fit].mean(dtype=np.float64)),
                     float(windows.values[fit].std(dtype=np.float64)))
        if not np.isfinite(std) or std < 1e-12:
            raise ValueError("development training traces have invalid ADC variance")
        values = (windows.values - mean) / std
        predictions, files, losses = [], [], []
        for member in range(args.ensemble_size):
            seed = args.seed + horizon * 100 + member
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            loader = DataLoader(TensorDataset(torch.from_numpy(values[fit, None]),
                                             torch.from_numpy(windows.labels[fit])),
                                batch_size=32, shuffle=True,
                                generator=torch.Generator().manual_seed(seed))
            model = PowerTraceCnn().cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.001)
            history = []
            for _ in range(args.epochs):
                model.train()
                total = 0.0
                for batch, labels in loader:
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(model(batch.cuda()), labels.cuda())
                    if not torch.isfinite(loss):
                        raise ValueError("non-finite detector training loss")
                    loss.backward()
                    optimizer.step()
                    total += float(loss.detach()) * len(labels)
                history.append(total / int(fit.sum()))
            predictions.append(cnn_scores(model, values[val], 32))
            filename = f"cnn_{horizon}ms_{member}.pt"
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, args.bundle / filename)
            files.append(filename)
            manifest["file_sha256"][filename] = digest(args.bundle / filename)
            losses.append(history)
            del model, optimizer
        decision = select_decision(np.mean(predictions, axis=0), windows.labels[val])
        manifest["horizons"][str(horizon)] = {"ridge": ridge,
            "cnn": {"files": files, "mean": mean, "std": std, "decision": decision,
                    "training_loss": losses}}
        write_json(args.bundle / "manifest.json", manifest)
        print(json.dumps({"horizon_ms": horizon, "ridge_validation": ridge["decision"],
                          "cnn_validation": decision}), flush=True)
    manifest["status"] = "frozen"
    manifest["frozen_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.bundle / "manifest.json", manifest)
    verify_bundle(args.bundle)
    print(f"FROZEN manifest_sha256={digest(args.bundle / 'manifest.json')}", flush=True)


def evaluate_scores(scores, labels, groups, decision, protocol) -> dict:
    prediction = predict(scores, decision)
    rows = [{"pair": int(pair), "balanced_accuracy": balanced_accuracy(prediction[groups == pair],
             labels[groups == pair]), "windows": int(np.sum(groups == pair))} for pair in np.unique(groups)]
    per_pair = np.asarray([row["balanced_accuracy"] for row in rows])
    rng = np.random.default_rng(protocol["seed"])
    sampled = rng.choice(per_pair, (protocol["bootstrap_replicates"], len(rows)), replace=True).mean(1)
    tail = protocol["family_wise_alpha"] / protocol["primary_comparisons"]
    upper = float(np.quantile(sampled, 1 - tail))
    accuracy = balanced_accuracy(prediction, labels)
    directed_auc = auc(decision["sign"] * scores, labels)
    return {"balanced_accuracy": accuracy, "pair_mean_balanced_accuracy": float(per_pair.mean()),
            "classifier_error_rate": 1 - accuracy, "oriented_roc_auc": directed_auc,
            "pair_bootstrap_upper_bound": upper,
            "bound_below_target": upper < protocol["accuracy_target"],
            "confusion_true_rows_predicted_columns": np.bincount(2 * labels + prediction, minlength=4)
                .reshape(2, 2).tolist(), "pairs": rows,
            "posthoc_orientation_diagnostic": max(accuracy, 1 - accuracy),
            "posthoc_auc_orientation_diagnostic": max(directed_auc, 1 - directed_auc),
            "diagnostics_are_not_independent_test_results": True}


def evaluate(args) -> None:
    import torch
    from ..llama_continuous_whole_model.cnn_detector import PowerTraceCnn, make_windows

    if args.output.exists():
        raise FileExistsError("refusing to overwrite an existing frozen evaluation")
    manifest = verify_bundle(args.bundle)
    plan = json.loads((args.test_root / "collection_plan.json").read_text())
    check_test_contract(manifest, plan)
    manifest_hash = digest(args.bundle / "manifest.json")
    if plan.get("frozen_detector_manifest_sha256") != manifest_hash:
        raise ValueError("test collection was not bound to this frozen detector manifest")
    traces, checked = checked_traces(args.test_root)
    if set(checked["trace_sha256"].values()) & set(manifest["development_trace_sha256"].values()):
        raise ValueError("test collection contains reused development traces")
    result = {"bundle_manifest_sha256": manifest_hash, "status": "evaluating",
              "protocol": manifest["protocol"], "test_plan": manifest["test_plan"],
              "physical_audit_passed": checked["passed"], "horizons": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for horizon in HORIZONS:
        frozen = manifest["horizons"][str(horizon)]
        x, y, groups = feature_data(traces, horizon)
        with np.load(args.bundle / frozen["ridge"]["file"], allow_pickle=False) as state:
            ridge_scores = ridge_predict(state, x)
        ridge = evaluate_scores(ridge_scores, y, groups, frozen["ridge"]["decision"], manifest["protocol"])
        windows = make_windows(traces, horizon)
        values = (windows.values - frozen["cnn"]["mean"]) / frozen["cnn"]["std"]
        predictions = []
        for filename in frozen["cnn"]["files"]:
            model = PowerTraceCnn().cuda()
            model.load_state_dict(torch.load(args.bundle / filename, map_location="cpu", weights_only=True))
            predictions.append(cnn_scores(model, values, 32))
            del model
        groups = np.asarray([int(session[1:]) for session in windows.sessions])
        cnn = evaluate_scores(np.mean(predictions, axis=0), windows.labels, groups,
                              frozen["cnn"]["decision"], manifest["protocol"])
        result["horizons"][str(horizon)] = {"ridge": ridge, "cnn": cnn}
        write_json(args.output, result)
        print(json.dumps({"horizon_ms": horizon, "ridge": ridge["balanced_accuracy"],
                          "cnn": cnn["balanced_accuracy"]}), flush=True)
    result["status"] = "complete"
    result["all_accuracy_bounds_below_target"] = all(
        r["bound_below_target"] for h in result["horizons"].values() for r in h.values())
    result["scope"] = "these frozen attackers and this mixed workload; not unmodified inference-only serving"
    write_json(args.output, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--development-root", type=Path, required=True)
    fit.add_argument("--bundle", type=Path, required=True)
    fit.add_argument("--seed", type=int, default=20260921)
    fit.add_argument("--epochs", type=int, default=15)
    fit.add_argument("--ensemble-size", type=int, default=3)
    fit.add_argument("--test-pairs", type=int, default=48)
    fit.add_argument("--test-captures", type=int, default=8)
    fit.add_argument("--test-seed", type=int, default=20271001)
    test = commands.add_parser("evaluate")
    test.add_argument("--bundle", type=Path, required=True)
    test.add_argument("--test-root", type=Path, required=True)
    test.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (train if args.command == "train" else evaluate)(args)


if __name__ == "__main__":
    main()
