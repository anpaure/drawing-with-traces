"""Audit complete physical sessions and their associated workload progress."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def check_progress(snapshots: list[dict]) -> dict:
    if len(snapshots) < 2:
        raise ValueError("at least two workload snapshots are needed to verify advancing training")
    for snapshot in snapshots:
        for field in ("last_loss", "parameter_probe_delta_linf"):
            if not math.isfinite(snapshot[field]):
                raise ValueError(f"non-finite workload {field}")
        if snapshot["parameter_probe_delta_linf"] <= 0:
            raise ValueError("workload reports no changed parameter probe")
    updates = [s["mixed_training_updates"] for s in snapshots]
    tokens = [s["mixed_inference_tokens"] for s in snapshots]
    if any(b < a for a, b in zip(updates, updates[1:])) or updates[-1] <= updates[0]:
        raise ValueError("training updates did not advance monotonically through the session")
    if any(b < a for a, b in zip(tokens, tokens[1:])) or tokens[-1] <= tokens[0]:
        raise ValueError("inference output count did not advance monotonically through the session")
    return {"first_update": updates[0], "last_update": updates[-1],
            "first_inference_tokens": tokens[0], "last_inference_tokens": tokens[-1],
            "loss_range": [min(s["last_loss"] for s in snapshots), max(s["last_loss"] for s in snapshots)]}


def audit(root: Path) -> dict:
    plan = json.loads((root / "collection_plan.json").read_text())
    if plan["status"] != "complete":
        raise ValueError("collection plan is incomplete")
    sessions = []
    fingerprints: dict[int, set] = {}
    hashes = {}
    for planned in plan["plan"]:
        session = root / planned["role"] / planned["session_id"]
        paths = sorted((session / "records").rglob("*.json"))
        if len(paths) != plan["captures_per_session"]:
            raise ValueError(f"incomplete session: {session}")
        snapshots = []
        stds = []
        for path in paths:
            record = json.loads(path.read_text())
            labels = record["labels"]
            if labels["process"] != planned["role"] or labels["session_id"] != planned["session_id"]:
                raise ValueError(f"incorrect record labels: {path}")
            descriptor = record["channels"][record["primary_channel"]]
            trace = session / descriptor["path"]
            values = np.load(trace, allow_pickle=False)
            if not record["health"]["ok"] or not np.isfinite(values).all():
                raise ValueError(f"unhealthy physical trace: {trace}")
            if values.size != 150_000 or descriptor["sample_rate_hz"] != 1_500_000:
                raise ValueError(f"unexpected capture geometry: {trace}")
            worker = record["workload_metadata"]["worker"]["computation"]
            if worker["compute_seed"] != planned["compute_seed"]:
                raise ValueError(f"incorrect compute seed: {path}")
            if worker["attention_backend"] != plan["attention_backend"]:
                raise ValueError(f"incorrect attention backend: {path}")
            fingerprints.setdefault(planned["pair"], set()).add(labels["computation_fingerprint"])
            snapshots.append(record["result_summary"]["snapshot"])
            stds.append(float(values.std()))
            hashes[str(trace.relative_to(root))] = hashlib.sha256(trace.read_bytes()).hexdigest()
        sessions.append({**planned, "records": len(paths), "progress": check_progress(snapshots),
                         "adc_std_range": [min(stds), max(stds)]})
    if len(fingerprints) != plan["pairs"] or any(len(values) != 1 for values in fingerprints.values()):
        raise ValueError("paired host labels did not execute the same computation configuration")
    return {"passed": True, "pairs": plan["pairs"], "record_count": len(hashes),
            "sessions": sessions, "paired_fingerprints_match": True, "trace_sha256": hashes,
            "scope": "all trace samples; recorded workload snapshots; not every intermediate gradient"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("passed", "pairs", "record_count")}))


if __name__ == "__main__":
    main()
