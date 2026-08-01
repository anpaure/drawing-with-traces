#!/usr/bin/env python3
"""Train a raw-waveform CNN with complete inference/training sessions held out."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from .analyze_continuous import ContinuousTrace, load_traces
except ImportError:  # pragma: no cover - direct hardware invocation.
    from analyze_continuous import ContinuousTrace, load_traces


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, *, stride: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channels, channels, 9, stride=stride, padding=4, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 7, padding=3, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.skip = (
            nn.AvgPool1d(stride, stride=stride, ceil_mode=True)
            if stride > 1
            else nn.Identity()
        )
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(values) + self.skip(values))


class PowerTraceCnn(nn.Module):
    """Small multiscale temporal attacker; input is one raw ADC channel."""

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, channels, 31, stride=4, padding=15, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            ResidualBlock(channels, stride=2),
            ResidualBlock(channels, stride=2),
            ResidualBlock(channels, stride=2),
            nn.Conv1d(channels, channels * 2, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(channels * 2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels * 2, 2),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


@dataclass(frozen=True)
class WindowSet:
    values: np.ndarray
    labels: np.ndarray
    processes: np.ndarray
    sessions: np.ndarray


def make_windows(traces: list[ContinuousTrace], horizon_ms: float) -> WindowSet:
    rows = []
    labels = []
    processes = []
    sessions = []
    for trace in traces:
        window_samples = int(round(horizon_ms * 1e-3 * trace.sample_rate_hz))
        if window_samples < 16:
            raise ValueError("CNN windows must contain at least 16 samples")
        label = 0 if trace.process == "inference" else 1
        for start in range(0, len(trace.values) - window_samples + 1, window_samples):
            rows.append(trace.values[start : start + window_samples].astype(np.float32))
            labels.append(label)
            processes.append(trace.process)
            sessions.append(trace.session_id)
    if not rows:
        raise ValueError("No complete CNN windows could be extracted")
    return WindowSet(
        values=np.stack(rows),
        labels=np.asarray(labels, dtype=np.int64),
        processes=np.asarray(processes),
        sessions=np.asarray(sessions),
    )


def held_out_mask(windows: WindowSet, inference_session: str, training_session: str) -> np.ndarray:
    return ((windows.processes == "inference") & (windows.sessions == inference_session)) | (
        (windows.processes == "training") & (windows.sessions == training_session)
    )


def balanced_accuracy(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean([np.mean(prediction[truth == label] == label) for label in (0, 1)]))


def train_fold(
    windows: WindowSet,
    *,
    held_inference: str,
    held_training: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    test_mask = held_out_mask(windows, held_inference, held_training)
    if not test_mask.any() or test_mask.all():
        raise ValueError("fold must contain both training and held-out windows")
    train_values = windows.values[~test_mask]
    test_values = windows.values[test_mask]
    scalar_mean = float(train_values.mean(dtype=np.float64))
    scalar_std = float(train_values.std(dtype=np.float64))
    if scalar_std < 1e-12:
        raise ValueError("training ADC windows have zero variance")
    train_values = (train_values - scalar_mean) / scalar_std
    test_values = (test_values - scalar_mean) / scalar_std
    train_labels = windows.labels[~test_mask]
    test_labels = windows.labels[test_mask]

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_values[:, None, :]),
            torch.from_numpy(train_labels),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = PowerTraceCnn().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        examples = 0
        for values, labels in loader:
            values = values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(values)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach()) * len(labels)
            examples += len(labels)
        history.append(running_loss / examples)

    model.eval()
    prediction_batches = []
    with torch.inference_mode():
        for start in range(0, len(test_values), batch_size):
            values = torch.from_numpy(test_values[start : start + batch_size, None, :]).to(device)
            prediction_batches.append(model(values).argmax(dim=1).cpu().numpy())
    prediction = np.concatenate(prediction_batches)
    confusion = np.zeros((2, 2), dtype=np.int64)
    for truth, predicted in zip(test_labels, prediction):
        confusion[truth, predicted] += 1
    return {
        "held_out_inference_session": held_inference,
        "held_out_training_session": held_training,
        "balanced_accuracy": balanced_accuracy(prediction, test_labels),
        "test_windows": len(test_labels),
        "confusion_matrix_true_rows_predicted_columns": confusion.tolist(),
        "final_training_loss": history[-1],
        "training_loss_history": history,
        "normalization": {"training_adc_mean": scalar_mean, "training_adc_std": scalar_std},
    }


def evaluate(
    root: Path,
    *,
    horizon_ms: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    inference = load_traces(root, "inference")
    training = load_traces(root, "training")
    inference_sessions = sorted({trace.session_id for trace in inference})
    training_sessions = sorted({trace.session_id for trace in training})
    if len(inference_sessions) < 2 or len(training_sessions) < 2:
        raise ValueError("CNN evaluation requires at least two complete sessions per process")
    windows = make_windows([*inference, *training], horizon_ms)
    folds = []
    for inference_index, held_inference in enumerate(inference_sessions):
        for training_index, held_training in enumerate(training_sessions):
            folds.append(
                train_fold(
                    windows,
                    held_inference=held_inference,
                    held_training=held_training,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    seed=seed + 100 * inference_index + training_index,
                    device=device,
                )
            )
    accuracies = np.asarray([fold["balanced_accuracy"] for fold in folds])
    confusion = np.sum(
        [fold["confusion_matrix_true_rows_predicted_columns"] for fold in folds], axis=0
    )
    return {
        "root": str(root),
        "horizon_ms": horizon_ms,
        "attacker_input": "raw ADC windows only",
        "split": "leave one complete inference session and one complete training session out",
        "architecture": "1D residual CNN",
        "epochs_per_fold": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "device": str(device),
        "inference_sessions": inference_sessions,
        "training_sessions": training_sessions,
        "windows": len(windows.labels),
        "balanced_accuracy": float(accuracies.mean()),
        "fold_standard_deviation": float(accuracies.std(ddof=1)),
        "minimum_fold_accuracy": float(accuracies.min()),
        "maximum_fold_accuracy": float(accuracies.max()),
        "aggregate_confusion_matrix_true_rows_predicted_columns": confusion.tolist(),
        "folds": folds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon-ms", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.horizon_ms <= 0 or args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("horizon, epochs, batch size, and learning rate must be positive")
    result = evaluate(
        args.root,
        horizon_ms=args.horizon_ms,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=torch.device(args.device),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
