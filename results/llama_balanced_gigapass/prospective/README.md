# Frozen prospective test — results pending

The [protocol](../../../experiments/llama_balanced_gigapass/PROSPECTIVE_EVALUATION.md)
was fixed before final test collection. The bundle contains five ridge models,
fifteen CNN checkpoints, training-only normalization, validation-selected thresholds
and directions, source hashes, and the exact new capture contract.

- Frozen UTC: `2026-09-20T18:12:50.774240+00:00`.
- Manifest SHA-256: `566369c1423923fa901801c26a84ce45922a44be96ab6d04cb28ad22b2516fd4`.
- Training pairs from the old dataset: `0, 2, 3, 5, 6, 7, 8, 11`.
- Validation pairs from the old dataset: `1, 4, 9, 10`.
- Prospective test: 48 new pairs, 768 physical captures, collection seed `20271001`.
- Test compute seeds: `20272001` through `20272048`; no development-seed overlap.

The bundle was copied from the H100 and its source/model hashes reverified locally.
Checkpoint files are tensor-only state dictionaries, loaded with `weights_only=True`.
No final-test score, passing gate, or new throughput improvement is asserted here.

```bash
python -m experiments.llama_balanced_gigapass.frozen_detector evaluate \
  --bundle results/llama_balanced_gigapass/prospective/frozen_attackers_v1 \
  --test-root artifacts/balanced_eager_prospective48_v1 \
  --output runs/prospective_evaluation.json
```

Evaluation requires the complete audited test set, its matching frozen-manifest
binding, and the source version recorded by the bundle. Source changes are rejected
rather than silently using different preprocessing or inference logic.

## Hardware observation during collection

The [18:39:38 UTC driver snapshot](diagnostics/thermal_observation_20260920_183938_utc.txt)
reports software thermal slowdown **active**, GPU temperature 91°C, and SM clock
1,080 MHz versus a reported maximum of 1,755 MHz. This is a point observation, not
a reconstruction of the earlier timing runs or a claim of proportional performance
loss. The cumulative throttle counters span earlier activity and are not per-run
measurements. No clocks, power limits, cooling settings or detector decisions were
changed in response, and no traces are filtered using this observation.
