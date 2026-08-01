# Drawing with traces

Train a real model while shaping its measured GPU power activity into an image silhouette.

This is a standalone experiment built on [SideCapture](https://github.com/anpaure/sidecapture).
It does not synthesize, replace, or geometrically warp measured values.

![Measured 10 ms silhouette](results/fast-10ms/measured_silhouette.png)

## Best real-hardware result

| Property | Result |
|---|---:|
| Requested profile duration | **10 ms** |
| Controlled bins | 20 (0.5 ms each) |
| ChipWhisperer rate | **10 MSPS burst** |
| Captured samples | 150,000 |
| Model parameters | 16,777,216 |
| Pearson correlation | **0.9968** |
| Normalized MAE | **2.40%** |
| Normalized RMSE | **2.94%** |
| R² | **0.9933** |
| Shape accuracy (`100 × (1 − NRMSE)`) | **97.06%** |
| Training loss | **0.99772 → 0.83642** |

Hardware: NVIDIA H100 PCIe and ChipWhisperer Husky Plus. The committed result includes the raw
SideCapture dataset, explicit bin annotations, health report, exact commands, calibration, and derived
arrays under [`results/fast-10ms`](results/fast-10ms). SideCapture accepted the best trace on its first
attempt with no health issues, no ADC clipping, and all 150,000 requested samples present.

## Where SideCapture is used

This experiment does **not** bypass SideCapture. The fast capture path constructs
`sidecapture.ChipWhispererSampler` and runs the workload through `sidecapture.Experiment` with a
`DirectoryStore`, retry policy, warmup, CUDA synchronization, validators, and transactional workload
commit. The resulting manifest uses `sidecapture.dataset/v1` and records the sampler, resolved capture
plan, hardware/software provenance, annotations, validation metrics, and retry attempt.

The split of responsibility is:

- **SideCapture:** plans, arms, and reads the Husky; maps annotations; validates traces; retries failed
  acquisitions; and crash-safely commits the raw trace and metadata.
- **This repository:** lowers the image into a one-dimensional envelope, schedules genuine tiled
  gradient operations, calibrates tile width to measured activity, and scores/plots the committed trace.

## How the millisecond version works

A filled 2D shape first becomes a valid one-dimensional target:

1. Extract the foreground mask.
2. Measure the foreground height in every image column.
3. Move every column down to a common baseline.
4. Smooth and normalize the resulting height envelope.

The fast workload is then a hand-written tiled gradient for

```text
loss = 0.5 / batch × ||XW − Y||²
gradient = Xᵀ(XW − Y) / batch
```

Every controlled tile computes both `X @ W[:, start:end]` and its real gradient block
`X.T @ residual[:, start:end]`. Tile width changes the GEMM shape and therefore the measured GPU
activity. Repeated visits are averaged by output column before one deferred SGD update, so changing the
power pattern does not change the mathematical gradient.

On the H100, the tiled result matched the full untiled manual gradient exactly in the validation run
(`relative L2 = 0`, `max absolute difference = 0`). Every accepted iteration reduced the model loss.

## Calibration and iterative learning control

The first SideCapture run sweeps tile widths from idle through 4096 columns. At 10 ms, each width is
held for four consecutive 0.5 ms bins; calibration uses the repeated steady-state bins to match the
timing and AC-coupled frontend behavior of the drawing itself. It automatically chooses the most
monotonic ChipWhisperer feature and builds a width-to-activity map.

![Tile-width calibration](results/fast-10ms/tile_width_calibration.png)

The image envelope is inverted through that calibration. Subsequent real captures use iterative learning
control (ILC): measured per-bin error can adjust the next tile-width sequence. For the 10 ms result, the
calibrated feed-forward command (iteration 0) was already best; later high-gain corrections amplified
run-to-run variation, so the reported score is not cherry-picked after shifting or warping the trace.

## Reproduce the 10 ms experiment

```bash
python -m pip install -e .

drawing-with-traces fast \
  --image assets/target.png \
  --output runs/fast-10ms \
  --duration-ms 10 \
  --points 20 \
  --iterations 10 \
  --ilc-gain 1.0
```

SideCapture provides:

- `ChipWhispererSampler` acquisition and hardware planning;
- trigger-to-host annotation mapping;
- ADC clipping, finite-value, expected-length, variance, flatline, and bounds validation;
- automatic retries and sampler recovery;
- crash-safe records, raw channels, labels, artifacts, and provenance.

The 10 ms feature is normalized AC-coupled ChipWhisperer activity, **not calibrated watts**.

## Independent 100 ms result

The earlier 100 ms SideCapture/Husky run used 60 controlled bins at 1.5 MSPS and reached 94.46% shape
accuracy (`r = 0.9861`, R² = 0.9687). Its complete raw dataset and calibration remain under
[`results/fast-100ms`](results/fast-100ms).

![Measured 100 ms silhouette](results/fast-100ms/measured_silhouette.png)

## Seconds-scale watts mode

For slow absolute-power drawings, the project also supports SideCapture's timestamped NVML sampler. The
installed ChipWhisperer path is high-pass/AC-coupled, so NVML is the honest source for curves expressed in
watts over seconds.

The measured 10-second adaptive run reached 92.04% shape accuracy and `r = 0.9583`:

![Measured 10 second silhouette](results/h100-silhouette-short-v1/measured_silhouette.png)

```bash
drawing-with-traces run \
  --image assets/target.png \
  --output runs/h100-silhouette \
  --duration-s 10 \
  --adaptive
```

## Target preview

```bash
drawing-with-traces preview \
  --image assets/target.png \
  --output results/target-envelope.png
```

![Lowered target](results/target-envelope.png)

## Measurement integrity

- Raw traces are committed before optimizer state changes.
- Rejected captures clear gradients and retry without applying an update twice.
- Exact target values, tile commands, operation counts, model configuration, and timing are stored.
- Accuracy uses the unshifted measured curve; no lag shifting is applied to the plotted data.
- The plotted fast curve uses the calibration-selected feature and a documented 0.8-bin Gaussian display
  smoothing. Raw per-bin values remain available beside it.
- Hardware retries remain visible through each record's `attempt` and rejection log.
