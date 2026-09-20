# Eager batch-size screen

Real H100 measurements: 12 rounds, four mixed iterations per round. Each candidate
recalibrates native service in-process and compares against dedicated workloads at
its own batch geometry. No filler computation, clock changes or power-limit changes.

| Training rows | Inference rows | Native I:T | Inference slowdown | Training slowdown | Estimated TFLOP/s |
|---|---|---:|---:|---:|---:|
| 1,024 baseline | 4,352 / 4,608 | 1.0171:1 | 1.899× | 1.931× | 166.5 |
| 2,048 | 7,168 / 7,680 | 0.9776:1 | 2.013× | 1.967× | 165.8 |
| 4,096 | 13,312 / 14,336 | 0.9997:1 | 1.914× | 1.913× | 169.9 |

Combined throughput differs little in these separate runs. The 4,096-row result
is about 2.1% higher, not evidence of a reliable improvement without repeated
comparisons. The 2,048-row run narrowly misses the configured 2% service-balance
limit (2.24% measured imbalance); its original failed flag is retained. This is
not an architectural failure or a reason to chase sub-percent timing excursions.

Training token throughput rises from about 9,095/s to 10,017/s and 10,992/s;
inference throughput falls from 40,076/s to 37,041/s and 35,780/s. These are different
work mixes at approximately equal native service, not free gains. Larger batches
change SGD update frequency and do not establish equal learning progress per token.

The 4,096-row candidate passed two full mixed iterations against sequential eager
PyTorch: all 1,235,814,400 parameter elements, all gradients, inference logits and
last loss were bit-equal and checked parameters/gradients finite. 69,091 parameter
elements changed. **It has not received a new physical detector evaluation.**
The 1,024-row eager configuration remains the primary capture-tested result.

Reproduce the largest candidate:

```bash
python -m experiments.llama_balanced_gigapass.benchmark \
  --attention-backend eager --training-batch-size 4096 \
  --inference-candidates 8192,9216,10240,11264,12288,13312,14336,15360,16384,17408,18432,19456,20480,21504,22528,23552,24576 \
  --benchmark-rounds 12 --cycles-per-block 4 \
  --calibration-output runs/batch4096_calibration.json \
  --output runs/batch4096_benchmark.json --allow-failure
python -m experiments.llama_balanced_gigapass.validate \
  --attention-backend eager --calibration runs/batch4096_calibration.json \
  --output runs/batch4096_validation.json
```

The 2,048-row screen uses inference candidates 4,096 through 12,288 in increments
of 512. All other benchmark flags match. Raw per-round timings are preserved in
each new benchmark JSON. Original stdout and exit codes remain in
`artifacts/balanced_eager_batch_screen_v1/` on the local checkout and H100.
