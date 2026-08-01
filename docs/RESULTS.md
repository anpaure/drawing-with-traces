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
4. The teacher–student linear task approaches convergence after many captures, changing operating state.
5. A residual-MLP tiled implementation and model-based temporal controller remain future work.
