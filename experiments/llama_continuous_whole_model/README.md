# Continuous whole-model Llama camouflage: method and evidence

This directory implements the experiment summarized in the [branch README](../../README.md). The
question is narrow: **can real full-parameter Llama training be scheduled so an external current probe
has more difficulty separating it from real cached inference?**

![Experiment summary](../../results/llama_continuous_whole_model/readme_explainer.png)

## Scope at a glance

| Property | Validated setting |
|---|---|
| GPU | NVIDIA H100 PCIe 80 GB |
| Model | `meta-llama/Llama-3.1-8B`, 32 layers |
| Inference | NF4 weights, BF16 compute, changing prefill + KV-cached decode |
| Training | BF16, all 8.03B parameters, causal loss + full backward + AdamW |
| Sensor | PicoScope TA189 clamp → 6 dB attenuator → Husky Plus |
| Capture | burst, 1.5 MSPS, 100 ms, 150,000 samples, 12-bit, 10 dB gain |
| Observer input | normalized ADC samples only |
| Split | leave one complete inference session and one complete training session out |

The processes start and warm before acquisition. Each capture uses an independent 0–500 ms delay after
arming. The observer gets no workload boundaries, CUDA events, loss, prompt, token, cache, NVML, or host
timing. Four independently initialized sessions per class contribute eight healthy traces each.

The TA189 is specified flat only through 100 kHz, while analysis extends above that. Values are therefore
comparative normalized ADC activity, not calibrated watts. Exact hardware and software versions are in
[`provenance.json`](../../results/llama_continuous_whole_model/provenance.json).

## Validated result

| Schedule | Similarity | 5 ms | 20 ms | 100 ms | Useful targets/s |
|---|---:|---:|---:|---:|---:|
| Ordinary 8-bit AdamW | 0.678 | 99.98% | 100.00% | 100.00% | 356 |
| Sequence 8 × accumulation 16, no cover | 0.742 | 96.95% | 98.13% | 100.00% | 67 |
| Sequence 16 × accumulation 8 + decode each | 0.768 | 96.05% | 97.66% | 100.00% | 135 |
| Fused AdamW + 8 decodes/update | 0.889 | 69.98% | 74.22% | 93.36% | **381** |
| **Fused AdamW + 12 decodes/update** | **0.896** | **62.77%** | **63.13%** | **77.73%** | 307 |
| Fused AdamW + 12 ± 4 decodes/update | **0.931** | 64.55% | 73.05% | 89.06% | 305 |
| Fused AdamW + 16 decodes/update | 0.872 | 72.81% | 83.13% | 82.42% | 254 |
| 11 decodes interleaved through backward | **0.919** | 67.25% | 88.36% | 96.88% | 317 |

Accuracy is session-held-out balanced accuracy; chance is 50%. “Similarity” averages Jensen–Shannon
similarity for raw amplitude, per-trace normalized amplitude, and Welch spectra. Similarity is useful for
screening, but the held-out detector is the actual objective.

For cover-12, a raw-waveform residual 1D CNN reaches **71.70%** at 5 ms in the strongest attempted
configuration; ridge reaches 62.77%. The detector families agree that the signal is much weaker than the
ordinary baseline, but neither supports an indistinguishability claim. Fold variation is substantial and
is retained in [`final_summary.json`](../../results/llama_continuous_whole_model/final_summary.json).

![Measured cover-12 traces, envelope, spectrum, and detector](../../results/llama_continuous_whole_model/final_eval/fused-cover12/continuous_power_overview.png)

## Mechanisms tested

| Mechanism | Purpose | Outcome |
|---|---|---|
| Short-sequence accumulation | Repeat smaller forward/backward motifs | Some physical improvement; poor throughput and still highly detectable |
| Exact token-row linear autograd | Force dense calls toward one-token `M=1` geometry | Numerically exact, but 30× slower and less kernel-similar |
| Hybrid exact weight-gradient rows | Make part of `dW` use decode-shaped GEMMs | Exact controlled test; serialization produced a new signature |
| Fixed real decode cover | Insert genuine NF4 decode after each update | Best detector result at 12 tokens/update |
| Decode inside backward | Break backward into inference-separated segments | High marginal similarity; regular cadence exposed at long horizons |
| Randomized cover length | Remove one fixed period | Higher marginal similarity; detector still learned the session mixture |

### Why `M=1` was not enough

| Kernel profile | Process similarity to decode | ATen GEMMs with `M=1` |
|---|---:|---:|
| Ordinary 8-bit AdamW | 0.411 | 0.0% |
| Fused AdamW | 0.442 | 0.0% |
| Fused AdamW + 12 decodes | **0.843** | 2.6% |
| Exact token-row decomposition | 0.342 | **96.0%** |
| Exact hybrid `dW` decomposition | 0.409 | 87.5% |

The original 8-bit optimizer alone consumes about 290 ms of profiled CUDA time, while all dense training
GEMMs consume about 31 ms. Optimizing one GEMM geometry therefore targets a minority of the temporal
process. Row decomposition also creates thousands of serialized BF16 launches and memory operations.
The full stream—not one local proxy—is what the sensor and detector observe.

## Throughput accounting

“Useful targets” counts causal-loss targets only. It excludes cover tokens and redundant operations.

- original 8-bit-Adam baseline: **356 targets/s**;
- fused AdamW without cover: **940 targets/s**;
- fused AdamW + 12 decode tokens: **307 targets/s**.

Thus cover-12 retains 86% of the original baseline but 33% of the corresponding faster fused baseline.
Fused AdamW and 8-bit AdamW use the same AdamW rule and hyperparameters; their implementations and state
formats differ. The cover-12 peak allocation is about 65.3 GiB and includes the BF16 training model plus
the NF4 serving model.

## Interpretation

**The result says:** an adaptive workload can substantially degrade a power-only classifier while all
model training remains real and all parameters continue to update.

**The result does not say:** backward became inference, cover has no cost, training is invisible, or the
effect transfers unchanged to another monitor. A longer window already recovers information in this
setup. A serious defender should test cadence, use longer context, and combine independent telemetry.

The related [`jacobcd52/silhouette`](https://github.com/jacobcd52/silhouette) project solves a different
objective: target-specific A100/NVML waveform tracking using channel identification, deconvolution, and
iterative feedback. It is a better controller for drawing a chosen low-rate curve; it does not evaluate
training against an independent inference class. The [branch README](../../README.md#comparison-with-jacobcd52silhouette)
contains the direct comparison and source links.

## Reproduce

Benchmark one schedule without opening the scope:

```bash
python benchmark_schedule.py \
  --sequence-length 128 \
  --gradient-accumulation-steps 1 \
  --optimizer adamw_fused \
  --cover-decode-tokens-per-microbatch 12
```

Capture and analyze a continuous candidate:

```bash
python capture_continuous.py \
  --mode inference \
  --session-id inference-00 \
  --output-dir runs/fused-cover12 \
  --captures 8

python capture_continuous.py \
  --mode training \
  --session-id training-00 \
  --output-dir runs/fused-cover12 \
  --captures 8 \
  --optimizer adamw_fused \
  --cover-decode-tokens-per-microbatch 12

python analyze_continuous.py --root runs/fused-cover12
```

Audit the CUDA process and regenerate committed figures:

```bash
python profile_kernel_schedule.py \
  --mode training --optimizer adamw_fused \
  --cover-decode-tokens-per-microbatch 12 \
  --output runs/training-fused-cover12.json
python render_readme_explainer.py
```

## Files

| File | Purpose |
|---|---|
| `continuous_workloads.py` | Persistent full-model inference and training workers |
| `training_shapes.py` | Exact row/hybrid autograd mechanisms |
| `capture_continuous.py` | SideCapture acquisition and randomized arming |
| `analyze_continuous.py` | Similarity and session-held-out ridge detector |
| `cnn_detector.py` | Raw-waveform residual CNN detector |
| `profile_kernel_schedule.py` | Complete CUDA stream profiling |
| `kernel_similarity.py` | Boundary-free kernel-process comparison |
| `render_final_results.py` | Final trade-off figure |
| `render_readme_explainer.py` | Concise README figure from committed metrics |
