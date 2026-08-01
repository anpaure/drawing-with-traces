# Experiment and implementation guide

## Scope

`drawing-with-traces` turns one image boundary into a timed sequence of real gradient GEMMs, records the
H100 input-current side channel through SideCapture and a ChipWhisperer Husky Plus, and refines the
command with iterative learning control (ILC).

The strongest verified path is:

```text
PNG alpha mask
  → upper-boundary envelope (120 values)
  → dense tile-width calibration
  → persistent tiled training workload
  → SideCapture/Husky burst acquisition
  → annotated per-bin RMS activity
  → target-independent normalization
  → multiscale scoring and ILC
```

## Image lowering

`extract_envelope` supports three target definitions:

- `height`: foreground height in each image column;
- `upper-boundary`: inverted top contour, used by the logo result;
- `lower-boundary`: inverted bottom contour.

RGBA alpha is treated as foreground evidence, which matters for black logos on transparent backgrounds.
The foreground is cropped, sampled to the requested number of positions, smoothed, and normalized to
`[0, 1]`. The exact source and envelope SHA-256 hashes are stored.

## Training workload

`TiledLinearTrainingWorkload` allocates a persistent teacher–student regression problem on CUDA:

```text
X: batch × hidden
W: hidden × output
Y: X × teacher_weight
```

For output columns `[j:k]`, one controlled operation calculates:

```text
prediction = X @ W[:, j:k]
residual   = prediction - Y[:, j:k]
gradient   = X.T @ residual / batch
```

The profile can revisit columns. Gradient sums are divided by their visit counts before SGD, and any
unvisited columns are completed after acquisition. A full untiled gradient is also calculated for
numerical verification and no-drawing timing. BF16 matmuls plus FP32 accumulation/master weights explain
the small nonzero equivalence error.

## Timing and active baseline

The verified 100 ms trace contains 120 equal-duration bins. Unlike the early idle-floor implementation,
every target bin executes a nonzero gradient tile. Target zero maps to 128 columns. The same tile runs in
the 2 ms lead region so the H100 enters the profile from the correct low-work state rather than an idle
345 MHz clock state. The 2 ms tail remains idle to expose the end boundary.

Active-lead operations are redundant for the optimizer but are genuine model arithmetic. Their operation
count, tile-width sum, FLOPs, and wall time are explicitly included in performance metadata.

## Calibration

Timed mode predeclares RMS as its measurement feature. It measures an interleaved random ordering of
dense tile widths, repeated in different transition contexts. The current width grid is dense in the
control-sensitive range and adds high-width anchors. Isotonic regression produces a monotonic activity
curve and truncates widths beyond the measured peak.

Calibration is performed before drawing, without looking at the target error. The resulting feature,
sign, widths, measured values, within-width standard deviations, and monotonic curve are saved.

## Feedback

The feed-forward command is the inverse calibration evaluated at the image envelope. ILC then retains
the best delivered command and adds a calibration-scaled correction:

```text
error        = target - measured
delta_width  = inverse(target + gain × error) - inverse(target)
next_command = best_command + delta_width
```

Corrections are quantized to 32-column Tensor Core-compatible widths and clipped to the calibrated range.
Optional correction smoothing acts only on the learned residual, never on the target baseline. It is off
by default because the hardware ablation reduced fidelity.

With multiple replicates, each replicate is a separate accepted trace and optimizer step. Pointwise median
curves drive feedback, but only a single physical trace may be promoted as the result.

## Capture transaction

SideCapture calls workload setup once and then runs repeated captures against the same model. During one
attempt:

1. Gradient buffers and visit counters are cleared.
2. Pre-update train and held-out losses are evaluated.
3. The sampler arms and the workload emits explicit lead/profile/bin/tail annotations.
4. SideCapture reads and validates the Husky trace.
5. On acceptance, missing gradient columns are completed, equivalence is checked, SGD is applied, and
   post-update losses are measured.
6. On rejection, accumulated gradients are cleared and no optimizer step occurs.

This prevents a scope failure or retry from applying an update twice.

## Saved evidence

The published directory contains the complete `sidecapture.dataset/v1` store:

- `captures/channels`: raw float16 ADC samples;
- `captures/records`: capture plan, labels, health, provenance, and workload metadata;
- `captures/annotations`: sample-aligned regions and all 120 bin boundaries;
- `captures/artifacts`: commands, per-bin operation counts, and gradient-column visits;
- per-replicate derived arrays and plots;
- calibration and training-comparison plots;
- full and compact experiment summaries.

## Claims and non-claims

Supported by the committed evidence:

- the plotted curve came from one real Husky capture of H100 training;
- all 120 positions execute real forward/gradient GEMMs;
- the persistent model's train and held-out losses decrease;
- the tiled gradient closely matches the untiled manual gradient;
- raw health checks pass and sample/event counts are complete.

Not claimed:

- calibrated watts (the Husky path is AC-coupled);
- bit-for-bit FP32 equivalence;
- identical throughput to ordinary training;
- a literal 14-layer residual-MLP Fable reproduction.
