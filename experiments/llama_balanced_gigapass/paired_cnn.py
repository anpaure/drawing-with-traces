"""Raw-ADC CNN with disjoint train, validation and test session pairs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from ..llama_continuous_whole_model.analyze_continuous import load_traces


def pair_partition(groups, test_pair):
    unique = sorted(np.unique(groups).tolist())
    if len(unique) < 4 or test_pair not in unique:
        raise ValueError("CNN audit requires at least four session pairs and an existing test pair")
    validation_pair = unique[(unique.index(test_pair) + 1) % len(unique)]
    return (groups != test_pair) & (groups != validation_pair), groups == validation_pair, groups == test_pair


def evaluate(root: Path, horizon: float, epochs: int, batch_size: int, seed: int) -> dict:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from ..llama_continuous_whole_model.cnn_detector import PowerTraceCnn, make_windows, balanced_accuracy

    traces = load_traces(root, "inference") + load_traces(root, "training")
    if not all(t.health_ok and np.isfinite(t.values).all() for t in traces):
        raise ValueError("CNN audit requires healthy finite traces")
    windows = make_windows(traces, horizon)
    groups = np.array([int(session[1:]) for session in windows.sessions])
    for group in np.unique(groups):
        if set(windows.labels[groups == group]) != {0, 1}:
            raise ValueError("each session pair must contain both labels")
    folds = []
    for pair in sorted(np.unique(groups).tolist()):
        random.seed(seed + pair)
        np.random.seed(seed + pair)
        torch.manual_seed(seed + pair)
        torch.cuda.manual_seed_all(seed + pair)
        train, val, test = pair_partition(groups, pair)
        mean = float(windows.values[train].mean(dtype=np.float64))
        std = float(windows.values[train].std(dtype=np.float64))
        if std < 1e-12:
            raise ValueError("training windows have zero ADC variance")
        x = (windows.values - mean) / std
        loader = DataLoader(TensorDataset(torch.from_numpy(x[train, None]),
                                         torch.from_numpy(windows.labels[train])),
                            batch_size=batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(seed + pair))
        model = PowerTraceCnn().cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.001)
        losses = []
        for _ in range(epochs):
            model.train()
            total, examples = 0.0, 0
            for values, labels in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(values.cuda()), labels.cuda())
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(labels)
                examples += len(labels)
            losses.append(total / examples)
        model.eval()

        def predictions(mask):
            selected = x[mask]
            result = []
            with torch.inference_mode():
                for start in range(0, len(selected), batch_size):
                    values = torch.from_numpy(selected[start:start + batch_size, None]).cuda()
                    result.append(model(values).argmax(-1).cpu().numpy())
            return np.concatenate(result)

        validation_prediction = predictions(val)
        validation_accuracy = balanced_accuracy(validation_prediction, windows.labels[val])
        flip = validation_accuracy < 0.5
        raw_prediction = predictions(test)
        prediction = 1 - raw_prediction if flip else raw_prediction
        folds.append({
            "test_pair": pair,
            "validation_pair": int(groups[val][0]),
            "training_pairs": sorted(np.unique(groups[train]).tolist()),
            "validation_accuracy": validation_accuracy,
            "flip_selected_on_validation": flip,
            "balanced_accuracy": balanced_accuracy(prediction, windows.labels[test]),
            "unflipped_balanced_accuracy": balanced_accuracy(raw_prediction, windows.labels[test]),
            "test_windows": int(test.sum()),
            "confusion_true_rows_predicted_columns": np.bincount(
                2 * windows.labels[test] + prediction, minlength=4
            ).reshape(2, 2).tolist(),
            "training_loss": losses,
        })
        print(json.dumps({"horizon_ms": horizon, **{k: folds[-1][k] for k in
                         ("test_pair", "balanced_accuracy", "flip_selected_on_validation")}}), flush=True)
        model = None
        del optimizer
    return {
        "horizon_ms": horizon, "epochs": epochs, "seed": seed, "batch_size": batch_size,
        "architecture": "raw-ADC 1D residual CNN", "folds": folds,
        "balanced_accuracy": float(np.mean([f["balanced_accuracy"] for f in folds])),
        "unflipped_balanced_accuracy": float(np.mean([f["unflipped_balanced_accuracy"] for f in folds])),
        "selection": "fixed epochs; prediction direction chosen on separate validation pair",
        "pair_count": len(folds),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--horizons-ms", default="5,10,20,50,100")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    result = {"root": str(args.root), "horizons": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for horizon in map(float, args.horizons_ms.split(",")):
        result["horizons"][str(horizon)] = evaluate(
            args.root, horizon, args.epochs, args.batch_size, args.seed
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
