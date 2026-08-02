# Committed strict-experiment evidence

These are compact derivatives of the real H100 and ChipWhisperer runs. Raw 80-trace capture stores remain
on the acquisition host; the committed files preserve the complete final metrics, every detector fold,
the representative plot, profiler summaries, benchmarks, and numerical validations.

| Path | Contents |
|---|---|
| `summary.json` | Compact selected configuration, result, throughput, arithmetic, and validation claims |
| `provenance.json` | Compute host, probe chain, acquisition settings, software, and analog caveat |
| `best_no_cover/continuous_power_metrics.json` | All stationary metrics and 125 held-out fold records (25 at each of five horizons) |
| `best_no_cover/continuous_power_overview.png` | Raw trace, RMS envelope, PSD, and detector overview |
| `benchmarks/ordinary_training.json` | Warmed ordinary CUDA-graph throughput baseline |
| `benchmarks/streaming_grouped_training.json` | Warmed selected no-cover throughput and corrected dW audit |
| `profiles/bf16_inference.json` | One complete cached-decode CUDA stream profile |
| `profiles/streaming_grouped_training.json` | One complete selected training-update CUDA stream profile |
| `validation/fused_optimizer_bitwise_equivalence.json` | Scalar-vs-fused shaped update comparison over 1.235B values |
| `validation/selected_backend_vs_ordinary.json` | Grouped-M1 shaped step versus ordinary BF16 training |
| `ablations/summary.json` | Final and smoke metrics for independent alternatives |
| `actuator_controls/summary.json` | Physical width/cadence controls and the failed <60% detector gate |

## Interpretation guardrails

- `mean_similarity = 0.9750` is **not** an indistinguishability score. The held-out detector remains
  90.25–95.75% accurate.
- The inference reference in this strict experiment is BF16. Older plots that called it “quantized” were
  mislabeled and have been regenerated.
- Single-session smoke rows in the ablation table do not have enough sessions for held-out evaluation.
- The selected dW audit includes 8.044 TFLOP of reported zero-padding arithmetic; it is not useful model
  training work.
- Fusion is bitwise equivalent within the selected shaped algorithm. The selected BF16 grouped-M1
  algorithm is not bitwise identical to ordinary dense PyTorch.
- The later actuator controls are negative evidence: none beats the retrained classifier more than 40%
  of the time. A fixed pre-existing classifier is not accepted as confirmation.
