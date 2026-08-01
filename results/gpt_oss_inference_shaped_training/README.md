# Curated GPT-OSS inference-shaped-training evidence

This directory contains compact artifacts from real NVIDIA H100 + ChipWhisperer Husky Plus runs. Raw
150,000-sample channels remain on the capture host; this Git branch stores the independently generated
metrics, health status, CUDA timing summaries, accepted loss trajectories, and rendered comparisons.

## Authoritative result

`full_model_cover8_matched_27/` is the strongest matched control:

* 27 inference and 27 training records per variant;
* ordinary and cover-8 variants begin from identical weights;
* both accept the same 27 causal updates and loss trajectory;
* all 108 records pass SideCapture health checks with zero capture rejection;
* every one of the nine cover-cycle offsets appears three times;
* aggregate power-only accuracy changes from 95.83% to 57.59%;
* training-first accuracy remains 80.83%, recorded as an explicit limitation.

Each variant contains:

* `power_similarity.json` — capture health, signal metrics, CUDA timings, every held-out classifier
  fold, offset breakdown, and per-trace summary statistics;
* `continuous_power_comparison.png` — raw snippet, RMS envelope, spectrum, and metric panel.

`capture_summary.json` records model, parameter names/shapes, capture settings, and accepted losses.

## Other evidence

* `full_model_cover4_final/` — lower-overhead cover-4 tradeoff.
* `balanced_projection/result.json` — exact-gradient and kernel-process proof for a balanced layer-2
  projection schedule.
* `layer23_attention_rows128_v1/` — retained negative result showing that dense M=1 bursts become more
  classifiable.
* `kernel_audit/` — compact whole-model phase profile and decode-comparison metrics.

The methodology and caveats are documented in
[`experiments/gpt_oss_inference_shaped_training/README.md`](../../experiments/gpt_oss_inference_shaped_training/README.md).
