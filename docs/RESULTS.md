# Resolution and controller results

The table uses the stricter multiscale score whenever raw data was available. It combines native-bin and
sigma-2 RMSE, so a visually smooth but locally oscillating trace cannot win only through filtering.

| Configuration | Distinct positions | Best multiscale fidelity | Main observation |
|---|---:|---:|---|
| Timed RMS, original idle floor | 120 | 92.96% | Strong outlier, but hard to replay because low bins idle the GPU |
| Timed RMS, original idle floor | 240 | 90.93% | Twice the positions, but 0.417 ms bins produce native ripple |
| Regularized ILC | 240 | 90.33% | Smoother commands did not recover the lost temporal response |
| Active floor, idle lead | 120 | 92.77% | Nonzero target bins substantially improve native fidelity |
| Active floor + active lead | 120 | 94.09% | Removing the pre-profile idle transition is a clear improvement |
| Active floor + active lead, three-replicate ILC | **120** | **94.26%** | Best verified single physical trace |
| Active floor + active lead | 100 | 91.06% | Longer bins alone do not compensate for a poorer calibration state |
| 14-layer residual MLP, RMS control | 120 | 93.08% | Exact autograd parity and 13.77× rather than 184× slowdown |
| 112-layer residual MLP, `diff_rms`, latest-feedback ILC | **120** | **93.66%** | 1.879B parameters, exact gradients, and only 1.74× slowdown |

## Why 120 won

At 100 ms, 120 positions provide 0.833 ms per command and roughly 1,250 ADC samples per annotated bin at
1.5 MSPS. Moving to 240 positions leaves 0.417 ms and about 625 samples. ADC count is still ample, but the
H100 workload and analog chain do not settle enough for every abrupt level transition. The result is more
discrete command resolution but worse native waveform fidelity.

Therefore 120 is not a memory or software cap. It is the current measured Pareto point between contour
resolution and delivered power fidelity.

## Why repetitions are not extra resolution

`tile_repeats` repeats the same operation and increases signal energy; it does not create new positions.
The 2× repeat ablation was dominated and remains off. The verified profile has 120 unique time positions.

Calibration context repeats and ILC replicates serve different purposes:

- calibration repeats measure the same width under different preceding transitions;
- ILC replicates are separate complete training captures used for robust feedback;
- neither changes the number of blocks in the promoted physical trace.

## Current limiting factors

1. The H100 power response has temporal memory and DVFS sensitivity.
2. The installed clamp/Husky path is AC-coupled and reports activity, not DC watts.
3. The simple pointwise ILC does not explicitly invert the temporal impulse response.
4. Both teacher–student tasks approach convergence after many captures, changing operating state.
5. The pointwise controller cannot distinguish stable response error from stochastic native-bin noise.

The new `latest` feedback mode partially tracks slow response drift. It improved the best physical trace
from 92.60% to 93.66%, but identical commands still degraded over later captures, so thermal/DVFS state
remains an unresolved plant variable.

## Residual-MLP efficiency result

The residual engine has 14 square 4096-wide layers (234,881,024 parameters). Its handwritten reverse
pass matched PyTorch autograd exactly, and its delivered tiled gradient matched the untiled gradient
exactly for every verified optimizer step. Layer-wide warmup eliminated first-touch timing failures; five
consecutive synthetic-context 100 ms steps and the complete scope run stayed within the bin-overrun gate.

The production run reached 93.08% multiscale fidelity. Its ordinary step is materially heavier than the
linear baseline, so the fixed 100 ms waveform is less dominant: median throughput was 9.10 drawing versus
125.31 ordinary steps/s, a 13.77× slowdown. This does not beat the linear trace's 94.26% shape score, but
it is the stronger demonstration that a substantial model can genuinely train while drawing.

A post-hoc feature ablation found one 94.84% `mean_abs` trace, but a fresh prospectively selected
`mean_abs` run produced an unstable calibration range and only 70.47%. That run is not promoted. RMS
remains the reproducible control feature.

## 112-layer efficiency/fidelity result

The strongest practical configuration uses 112 square 4096-wide residual layers, batch 2048, residual
scale `0.04419417382415922`, and 1,879,048,192 trainable parameters. It was selected from a geometry
sweep rather than by merely scaling the 14-layer workload:

- 56 layers / batch 4096 made even a 32-column GEMM too electrically active;
- restricting repeated profile work to 1, 4, or 14 carrier layers did not improve fidelity;
- 112 layers / batch 2048 restored tile-width separability while keeping ordinary training heavy;
- `diff_rms` supplied a full 32–3072-column control range and was substantially more stable than RMS or
  percentile span in this geometry.

The first unrefined trace reached 89.62%. Median-of-three ILC with latest-trace feedback promoted one
unaveraged trace at 93.66% multiscale fidelity (92.97% native, 94.43% smoothed, Pearson 0.9791). Every
accepted step matched both PyTorch autograd and the ordinary untiled gradient exactly in the tested BF16
path. Median throughput was 4.93 drawing versus 8.61 ordinary optimizer steps/s: 57.3% retained, or
1.74× slowdown.
