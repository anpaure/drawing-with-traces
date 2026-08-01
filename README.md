# Can GPT-OSS training look like cached inference in a power trace?

This branch contains the **GPT-OSS inference-shaped-training experiment**. It is a controlled precursor
to the stronger continuous whole-model Llama study on
[`experiment/llama-continuous-whole-model`](https://github.com/anpaure/drawing-with-traces/tree/experiment/llama-continuous-whole-model).
The PNG-drawing project remains on [`main`](https://github.com/anpaure/drawing-with-traces/tree/main).

![Experiment, result, and limits](results/gpt_oss_inference_shaped_training/readme_explainer.png)

## Answer

**Inference cover can make the aggregate trace look close to inference, but it does not make training
itself invisible.** The best run performs one genuine GPT-OSS attention-training update followed by
eight genuine cached-decode covers.

| External monitor result | Ordinary training | Cover 4 | Cover 8 |
|---|---:|---:|---:|
| Mean signal similarity to inference | 89.42% | 98.01% | **99.28%** |
| 5 ms power-only detector | **95.83%** | 64.44% | **57.59%** |
| Mean CUDA time / cycle | 91.34 ms | 254.17 ms | 408.86 ms |
| Slowdown | 1.00× | 2.83× | **4.48×** |

Chance is 50%. The important qualification is phase conditioning: cover-8 windows that begin with
training remain **80.83%** detectable, while most windows beginning inside the eight decode covers are
near chance. The 57.59% aggregate is therefore primarily a result about **cover-dominated windows**.

## What actually ran

### Inference reference

- `openai/gpt-oss-20b` complete cached decode;
- one generated token at a time with a real KV cache.

### Training target

- a complete GPT-OSS forward and backward pass;
- causal loss and real optimizer updates;
- all 96 q/k/v/o attention-projection matrices across all 24 layers trainable;
- **637,009,920 updated parameters**.

This is not full-parameter GPT-OSS training. Non-attention weights participate in the full forward
computation but are not optimizer targets. Ordinary and covered variants follow the same 27 accepted
updates, with loss falling from `6.722904` to `0.012314`.

### Best process

- one real attention-training update;
- eight real cached-decode covers;
- the starting cycle offset is rotated across captures.

Acquisition starts at a known workload call. It is not an indefinitely running, randomly sampled
service. The detector receives only features derived from current-probe ADC samples—no CUDA markers,
phase labels, loss, model metadata, timing counters, or NVML.

![Measured cover-8 comparison](results/gpt_oss_inference_shaped_training/full_model_cover8_matched_27/shaped/continuous_power_comparison.png)

## The local matmul experiment

For one layer-2 projection, the exact weight gradient can be transposed and evaluated one input-feature
row at a time:

```text
y  = x Wᵀ
dX = dY W
dWᵀ[j, :] = x[:, j]ᵀ dY
```

Zero-padding the reduction gives the same `M=1, K=input, N=output` geometry as one-token inference
without changing the mathematical gradient. In the balanced 4,096-row test:

- input gradients are bitwise identical;
- weight-gradient relative L2 error is `1.253 × 10⁻⁴` from BF16 reduction order;
- causal loss falls from `8.129142` to `8.117710` after the selected real update;
- local kernel-process similarity rises from **0.3612** to **0.9399**.

But the shaped local projection takes **1,705 ms instead of 35 ms**. At physical full-model scale,
applying 128 shaped rows to the final layer makes the detector worse: **87.78% → 95.56%**. Matching one
GEMM geometry created a highly regular launch burst rather than hiding the full temporal process.

## What this proves—and does not prove

**Supported by this experiment**

- Real inference cover can dominate short aggregate windows and substantially reduce a power-only
  classifier.
- A real gradient can be decomposed into inference-shaped GEMMs with high local kernel similarity.
- Local kernel matching and physical process matching are different objectives.

**Not supported**

- Backward or optimization became inference.
- Full-parameter GPT-OSS training was hidden; only 637M attention parameters were updated.
- The 57.59% result holds for training-centered windows: those remain 80.83% detectable.
- The process is a realistic unsynchronized service, universally indistinguishable, or low-overhead.

The implication is narrower than “power monitoring is broken”: a monitor that averages over unknown
phase can be manipulated with cover work, while a phase-aware, long-horizon, or independent monitor can
still accumulate evidence from recurring backward regions.

## Comparison with `jacobcd52/silhouette`

[`jacobcd52/silhouette`](https://github.com/jacobcd52/silhouette) is a related waveform-control
implementation, not a replication of this experiment. Its documented 2-second A100 run uses an
identified NVML channel, deconvolution, and iterative learning control to match a chosen contour.

| | This branch | `silhouette` at commit `7042c4e` |
|---|---|---|
| Goal | Make an independent inference class and training process collide | Draw a chosen target curve |
| Sensor | External clamp + ChipWhisperer, 1.5 MSPS | In-process NVML, 49.8 Hz |
| Workload | Full GPT-OSS forward/backward; 637M attention parameters updated | ResNet-18 training for 67% of the trace; filler matmuls elsewhere |
| Control | Fixed cover schedule with rotated offsets | System identification, deconvolution, target-specific feedback |
| Primary evidence | Inference-vs-training detector accuracy | Replay MAE/correlation to the chosen contour |

`silhouette` is better at **closed-loop reproduction of a known low-rate waveform**. This branch asks a
harder classification question, but has weaker process realism than the later Llama branch because its
capture is call-aligned and its optimizer scope is partial. See the external
[controller](https://github.com/jacobcd52/silhouette/blob/7042c4eac5a7456fa38e0d8821189da8db5b0378/power_trace_2s.py),
[engine](https://github.com/jacobcd52/silhouette/blob/7042c4eac5a7456fa38e0d8821189da8db5b0378/s2s.py),
and [analysis](https://github.com/jacobcd52/silhouette/blob/7042c4eac5a7456fa38e0d8821189da8db5b0378/ANALYSIS_2S.md).

## Reproduce and inspect

```bash
cd experiments/gpt_oss_inference_shaped_training

python run_balanced_projection.py \
  --output-dir runs/balanced4096 --layer 2 \
  --total-rows 4096 --microbatch-rows 256

python capture_full_model_power.py \
  --output-dir runs/cover8 --all-attention-projections \
  --traces-per-process 27 --decode-repeats 3 --training-repeats 1 \
  --duration 100ms --sample-rate 1.5MHz --gain-db 10 \
  --shaped-inference-rows 0 --shaped-forward-rows 2 \
  --shaped-input-gradient-rows 2 --shaped-cover-decodes 8
```

- [Method and file guide](experiments/gpt_oss_inference_shaped_training/README.md)
- [Balanced local projection evidence](results/gpt_oss_inference_shaped_training/balanced_projection/result.json)
- [Cover-8 capture manifest](results/gpt_oss_inference_shaped_training/full_model_cover8_matched_27/capture_summary.json)
- [Cover-8 detector and similarity metrics](results/gpt_oss_inference_shaped_training/full_model_cover8_matched_27/shaped/power_similarity.json)
- [Whole-model kernel audit](results/gpt_oss_inference_shaped_training/kernel_audit/kernel_comparison.json)
