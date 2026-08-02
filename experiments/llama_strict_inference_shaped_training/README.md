# Strict no-cover Llama experiment

This directory implements the branch's zero-cover experiment. Its training configuration has no field
that can request inference tokens or load a second model. The selected result is a negative one: shaping
closes marginal waveform statistics but does not defeat the held-out detector.

## Hard invariants

`StrictWorkloadConfig.strict_invariants()` is saved into every benchmark and capture:

```text
inference_cover_tokens = 0
secondary_model_instances = 0
filler_kernels = 0
all extra FLOPs are forward/gradient work or reported redundant training arithmetic
optimizer updates use current real gradients
```

The inference and training classes run in separate worker processes for data collection. Each process
loads exactly one model. “No second model” refers to the training process; the independently captured
inference class necessarily has its own inference worker.

## Selected dataflow

```text
BF16 token batch
  -> grouped-M1 forward through every dense module
  -> real transformer loss and backward
       -> grouped-M1 dX
       -> save X and dY when each dW becomes dependency-ready
       -> exact padded dW tiles
       -> grouped physical dW submission
       -> fused/interleaved SGD update
  -> CUDA-graph replay of the complete update
```

Each worker allocates one synthetic token batch at setup and reuses it. That removes data-loader and
batch-content drift from the scheduling comparison; it does not establish long-run language-model
training quality.

### Forward and dX

`StrictM1Linear` retains each original `Parameter` object and replaces only the linear execution. A
logical matrix product is split into groups of shared-RHS `1 × K @ K × N` products. The selected tile
group has 128 rows. Bias and all surrounding transformer operations remain unchanged.

`shaping_backend=grouped-m1` controls forward and `dX`. Deferred `dW` follows the scheduler's row-tile
path below; the backend name does not claim that deferred dW is an `M=1` operation.

### Exact dW

For `Y = X W.T`, each row tile of `dW.T` is computed from `X.T @ dY`. The token reduction is padded to
the module input width. Padding changes no result but creates explicitly redundant arithmetic with
inference-like reduction geometry.

The selected scheduler begins dW work as operands become available during reverse-mode traversal.
Shape-compatible tasks from stable layer cohorts are submitted with private `torch._grouped_mm`, then
their outputs are accumulated into the exact gradient buffer. The H100 validation records:

| dW audit | Per update |
|---|---:|
| Logical exact tiles | 2,576 |
| Physical GEMM launches | 176 |
| Grouped launches | 160 |
| Logical tiles covered by grouped launches | 2,560 |
| Useful dW arithmetic | 0.316 TFLOP |
| Padding arithmetic | 8.044 TFLOP |
| Total dW arithmetic | 8.360 TFLOP |

The private grouped API is feature-checked and fails with a direct error on unsupported CUDA/PyTorch
builds. `streaming-round-robin` remains the portable fallback.

### Optimizer

`InterleavedSGD` registers post-accumulate hooks for ordinary parameters and accepts manually completed
dense gradients from the scheduler. Updates are flushed in eight-tensor `foreach` buckets. Shared/tied
parameters are deferred until all contributions exist. The full selected model has 146 trainable tensor
objects and 1,235,814,400 values.

## Evaluation protocol

- GPU: NVIDIA H100 PCIe;
- sensor: external current clamp through the Husky Plus chain;
- capture: 1.5 MSPS, 100 ms, 150,000 samples, 12-bit, 10 dB gain;
- random delay after arm: 0–500 ms;
- five inference and five training sessions;
- eight accepted traces per session and process;
- all 80 traces pass health checks;
- observer input: ADC samples only.

`analyze_continuous.py` extracts non-overlapping windows at 5, 10, 20, 50, and 100 ms. The ridge model
standardizes using training folds only, and every fold holds out one complete session from each process.
The committed result includes all 25 session-pair folds at every horizon.

Stationary similarity averages three Jensen–Shannon similarities: raw amplitude, per-trace normalized
amplitude, and Welch PSD. It is a screening metric, not the security objective.

## Validation boundaries

### Algebra and fusion

Local float64 tests compare shaped forward, dX, dW, tied-parameter accumulation, and SGD with ordinary
linear algebra. H100 validation then compares scalar and fused versions of the *same selected shaped
algorithm*: loss, all 1.235B gradient values, and all 1.235B updated values are bitwise equal.

### BF16 ordinary comparison

Grouped-M1 and ordinary dense BF16 kernels reduce products in different orders. A selected-backend H100
run therefore reports, rather than hides, the numerical delta:

- loss absolute difference: `1.9169e-4`;
- gradient relative L2: `4.7590e-3`;
- updated-parameter relative L2: `8.2738e-7`;
- updated-parameter element equality: `0.99998936`.

“Exact dW” means the scheduled blocks implement the exact algebra for the shaped step. It does not mean
bitwise equality to a different BF16 kernel reduction order.

## Independent ablations

The implementation includes alternatives because they tested distinct hypotheses:

- `streaming-round-robin`: exact dW tiles, no physical grouped GEMM;
- `streaming-inference-cycle`: choose dependency-ready projection families in
  `q,k,v,o,gate,up,down` order;
- `streaming-grouped`: selected stable-cohort grouped dW scheduler;
- `--no-cuda-graph`: expose natural host launch cadence;
- `--kernel-launch-period-us`: experimental host pacing, no timing/filler kernel;
- `trace_drawing.py`: redundant exact current-step dW actuator for isolated waveform-control tests.

The actuator is not the selected continuous result. Appending its profile after a full update produced a
separable two-regime process and is retained only as a failed control.

### Physical actuator follow-up

The follow-up did not rescue this PyTorch-level design:

1. A 1 ms calibration varied both tile width and the number of exact current-gradient `dW` blocks. It
   achieved 3.689 us maximum deadline overrun across 68 bins, so the scope and replay machinery were
   capable of delivering the requested schedule.
2. Back-to-back fixed-width profiles remained detectable. The best complete control was four profile
   repetitions: held-out balanced accuracy was 85.1%, 85.4%, 86.0%, 86.3%, and 86.3% at 5, 10, 20,
   50, and 100 ms.
3. Fixed-duration bins made the periodic signature stronger. A 2 ms-bin/four-repetition run scored
   82.8–93.8%; a superficially promising 1.1 ms-bin/thirteen-repetition screen scored 93.8–96.1% after
   the attacker was retrained on two fresh sessions.
4. Aperiodic width programs removed the explicit 1 ms cadence and matched gross RMS, but introduced a
   different spectral fingerprint. Those one-session runs are diagnostics only, not held-out evidence.

An offline convex-hull diagnostic combined the standardized feature means of 22 physically measured
strict-training regimes. Even the optimal nonnegative mixture retained RMS feature residuals of 0.910,
0.934, 0.920, 0.968, and 1.103 standard deviations across the same horizons. This is not a detector
score, but it shows why another duty-ratio sweep is not a credible route to the 60% gate: the measured
PyTorch regimes do not surround the inference feature distribution.

The compact physical results are in
[`../../results/llama_strict_inference_shaped_training/actuator_controls/summary.json`](../../results/llama_strict_inference_shaped_training/actuator_controls/summary.json).
The next experiment therefore needs a shared lower-level carrier for forward, `dX`, and `dW`, rather
than appending more gradient work after an already distinctive training update.

## Files

| File | Purpose |
|---|---|
| `strict_shapes.py` | Shaped linear autograd, exact dW scheduling, grouping, pacing audit |
| `strict_optimizer.py` | Fused/interleaved SGD with tied-parameter handling |
| `strict_workloads.py` | Single-model persistent inference/training workers |
| `capture_strict.py` | SideCapture/Husky acquisition with randomized armed delay |
| `benchmark_strict.py` | Useful-target throughput excluding model load |
| `profile_strict.py` | Complete CUDA-kernel stream profile |
| `validate_strict.py` | Selected shaped backend versus ordinary BF16 training |
| `validate_optimizer_fusion.py` | Scalar versus fused shaped-update bitwise proof |
| `trace_drawing.py` | Accounted real-gradient actuator, not selected camouflage |
| `refine_trace_drawing.py` | Isolated closed-loop actuator refinement |

The shared detector is
[`../llama_continuous_whole_model/analyze_continuous.py`](../llama_continuous_whole_model/analyze_continuous.py).

## Commands

Benchmark the ordinary and selected schedules:

```bash
python -m experiments.llama_strict_inference_shaped_training.benchmark_strict \
  --mode ordinary-training --training-batch-size 128 --sequence-length 1 \
  --observe-seconds 8 --output runs/ordinary.json

python -m experiments.llama_strict_inference_shaped_training.benchmark_strict \
  --mode shaped-training --training-batch-size 128 --sequence-length 1 \
  --tile-rows 128 --shaping-backend grouped-m1 \
  --weight-gradient-schedule streaming-grouped \
  --streaming-dw-tasks-per-record 16 \
  --grouped-dw-min-batch 4 --grouped-dw-max-batch 16 \
  --optimizer-bucket-size 8 --observe-seconds 8 \
  --output runs/selected.json
```

The ordinary benchmark uses the same full-parameter SGD update rule and CUDA graph; only the shaped
linear replacements and deferred scheduler are absent.

Capture the inference class with the same model and acquisition settings:

```bash
python -m experiments.llama_strict_inference_shaped_training.capture_strict \
  --mode inference --session-id inference-s0 --seed 4101 \
  --output-dir runs/final --captures 8 \
  --duration 100ms --sample-rate 1.5MHz \
  --inference-batch-size 1024 --inference-decode-tokens 64
```

See the [branch README](../../README.md#reproduce) for training capture and validation commands.
