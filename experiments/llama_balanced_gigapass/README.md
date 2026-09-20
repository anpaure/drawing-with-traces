# Implementation guide

The [branch README](../../README.md) describes the measured claim and limitations.

| File | Purpose |
|---|---|
| `workload.py` | Model, data ring, two CUDA streams, persistent SideCapture worker |
| `scheduler.py` | Calibration and deterministic service accounting |
| `benchmark.py` | Rotated dedicated/mixed timing and explicit acceptance thresholds |
| `validate.py` | Full-model comparison with ordinary sequential SGD; operation profile |
| `capture.py` | Continuous ChipWhisperer capture with random trigger delay |
| `audit_detector.py` | Nested ridge regularization/sign selection; held-out session pairs |
| `paired_cnn.py` | Raw-ADC CNN with separate training, validation and test session pairs |
| `collect_pairs.py` | Predeclared capture order; matched seeds within independently seeded pairs |
| `audit_records.py` | ADC health and hashes; finite recorded losses and advancing workload counters |
| `screen_attention.py` | Same-state backend comparison and ABBA throughput screen |
| `render_results.py` | Raw-signal plot, saved metrics, trace SHA-256 manifest |
| `render_paired_report.py` | Seed-selected unprocessed windows and complete paired-detector report |

## Scheduling

Let `tT` be the native time of one training batch and `tI(b)` the time of inference
batch `b`. Calibration finds two batches with `rLow=tI(low)/tT <= 1 <= rHigh`.
The upper batch's target fraction is `(1-rLow)/(rHigh-rLow)`. Each cycle chooses
the batch minimizing the absolute accumulated difference between inference and
training service units. For fixed measured costs, this bounds the accumulated
error and makes the average ratio approach one.

This is an accounting guarantee relative to a calibration, not live measurement
of exclusive GPU time under concurrent kernels. Calibration is performed at startup:
paired native measurements followed by damped correction using sustained blocks.
There is no online DVFS estimator. Longer runs need renewed validation if conditions change.

## Measurement

Dedicated inference and dedicated training use the same engine, batch sizes, and
operator backends as mixed service. Each benchmark round rotates the measurement
order. The reported slowdown is mixed wall time divided by the corresponding
native wall time for the same amount of completed work. Startup, calibration and
scope transfer are excluded. A checksum and parameter probe are diagnostics, not
exports of model outputs or a trained model.

The saved passing measurement and repeat measurement differ at the percent level.
`stale_calibration_measurement.json` retains the failed transfer of a calibration
between sessions. It must not be silently dropped to claim universal balance.

## Detector interpretation

Both role labels execute exactly the same mixed workload. A low score measures
whether a classifier can recover an arbitrary host label from physical session
variation. It does not test whether an unmodified inference service can be told
apart from the mixed service. Collecting both controls is needed for that stronger claim.

The audit requires `iN` and `tN` sessions to form explicit matched pairs. Entire
pairs are excluded from normalization, fitting, regularization selection, and
prediction-direction selection. Outer-test labels are used only for scoring.
Never flip predictions based on their test accuracy and report that as an independent
held-out result. The orientation diagnostic is labeled as a diagnostic.

## Evidence

`results/llama_balanced_gigapass/` stores measured timing, full numerical checks,
scope health, raw-waveform CNN results, feature-detector results, and a trace hash
manifest. Raw ADC files are retained under
`artifacts/balanced_gigapass_capture_grouped_v1/` locally and under the same relative
path in `/home/amodo/strict_inference_shaped_v2` on the H100. Raw captures are excluded
from Git. Re-render with:

```bash
python -m experiments.llama_balanced_gigapass.render_results \
  --root artifacts/balanced_gigapass_capture_grouped_v1 \
  --results results/llama_balanced_gigapass
python -m pytest -q
```

For a completed paired collection, run the record audit before rendering:

```bash
python -m experiments.llama_balanced_gigapass.audit_records \
  --root artifacts/balanced_eager_pairs12_v1 \
  --output artifacts/balanced_eager_pairs12_v1/progress_audit.json
python -m experiments.llama_balanced_gigapass.render_paired_report \
  --root artifacts/balanced_eager_pairs12_v1 \
  --benchmark results/llama_balanced_gigapass/eager_certification.json \
  --output-dir results/llama_balanced_gigapass
```

`--raw-only` renders before detector evaluation finishes. Trace selection uses only
capture identifiers and a fixed seed. Windows retain all samples, their original ADC
scale and original time coordinates; no alignment, filtering or normalization is applied.
The record audit checks captured samples and associated workload snapshots, not every
intermediate gradient. The separate full-model numerical validation covers that claim
for its explicitly checked iterations.
