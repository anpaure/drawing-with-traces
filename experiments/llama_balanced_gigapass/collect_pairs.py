"""Predeclare and collect independent matched session pairs on exclusive hardware."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path


def collection_plan(pairs: int, seed: int) -> list[dict]:
    if pairs < 4:
        raise ValueError("at least four matched session pairs are required")
    rng = random.Random(seed)
    plan = []
    for pair in range(pairs):
        roles = ["inference", "training"]
        rng.shuffle(roles)
        for role in roles:
            plan.append({
                "role": role, "session_id": f"{role[0]}{pair}", "pair": pair,
                "compute_seed": seed + 1000 + pair,
                "capture_seed": seed + 2000 + pair,
            })
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--attention-backend", choices=("sdpa", "eager"), default="eager")
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--captures", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    # Never overwrite or silently resume a partly collected evaluation.
    args.root.mkdir(parents=True, exist_ok=False)
    plan = collection_plan(args.pairs, args.seed)
    payload = {"plan": plan, "pairs": args.pairs, "captures_per_session": args.captures,
               "attention_backend": args.attention_backend, "seed": args.seed,
               "calibration": json.loads(args.calibration.read_text()),
               "status": "collecting"}
    plan_path = args.root / "collection_plan.json"
    plan_path.write_text(json.dumps(payload, indent=2) + "\n")
    for session in plan:
        command = [sys.executable, "-m", "experiments.llama_balanced_gigapass.capture",
                   "--output-dir", str(args.root), "--role", session["role"],
                   "--session-id", session["session_id"], "--captures", str(args.captures),
                   "--attention-backend", args.attention_backend,
                   "--calibration-input", str(args.calibration),
                   "--compute-seed", str(session["compute_seed"]),
                   "--capture-seed", str(session["capture_seed"])]
        print(f"START {session['session_id']}", flush=True)
        with (args.root / f"{session['session_id']}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"DONE {session['session_id']}", flush=True)
    payload["status"] = "complete"
    plan_path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
