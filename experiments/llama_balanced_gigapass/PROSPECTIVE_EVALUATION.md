# Prospective detector test

The earlier cross-validation results had large session-to-session variation and
some below-chance scores. This test separates attacker development from collection
of the final test set. **The test is pending; a frozen manifest is not a passing result.**

## Fixed before test capture

- Workload: the already capture-tested eager configuration, training batch 1,024;
  the existing 4,352/4,608 inference calibration is reused unchanged.
- Development data: the 12 previously collected matched pairs. A seeded split
  assigns eight pairs to training and four to validation. Both roles in a pair
  stay in the same split.
- Ridge: ADC features, four regularization strengths, training-only normalization.
- CNN: ensemble of three independent 32-channel residual networks, 15 epochs each.
  Training-only ADC normalization; ensemble averages logit differences.
- Ridge strength, score orientation and decision thresholds are selected using
  validation data only. No detector is refit after its threshold is selected.
- Test: 48 new matched pairs, eight 100 ms / 1.5 MSPS captures per session.
  Collection seed 20271001 produces compute seeds disjoint from development data.
- Windows: 5, 10, 20, 50 and 100 ms. No metadata enters either classifier.

The complete detector files and relevant source files are SHA-256 hashed. The
collection records the frozen manifest hash **before** its first capture. Evaluation
rejects modified files, changed collection settings, reused compute seeds, repeated
raw trace hashes, incomplete sessions and unhealthy/non-finite traces. Both trace
labels continue executing the same mixed inference-and-SGD computation.

## Report, without changing direction on the test data

Every attacker/window combination reports fixed-direction balanced accuracy,
error rate, ROC AUC, confusion matrix and per-pair scores. Bootstrap resampling uses
whole matched pairs, not the many correlated windows inside each capture.

The predeclared one-sided percentile bound uses 20,000 bootstrap replicates and
Bonferroni adjustment over ten primary comparisons at family-wise alpha 0.05.
The accuracy target is below 60%, corresponding to the user's clarified target
of more than 40% classifier error. These are approximate bootstrap bounds, not
distribution-free guarantees; they assume pairs are sufficiently independent.
Persistent environmental drift can weaken that assumption.

Post-hoc orientation diagnostics are retained separately. They are not independent
held-out results and cannot be used to claim a stronger success. A low score from
these finite detectors does not establish zero detectability or camouflage against
ordinary inference-only serving.

## Reproduce

```bash
python -m experiments.llama_balanced_gigapass.frozen_detector train \
  --development-root artifacts/balanced_eager_pairs12_v1 \
  --bundle artifacts/balanced_frozen_attackers_v1

python -m experiments.llama_balanced_gigapass.collect_pairs \
  --root artifacts/balanced_eager_prospective48_v1 \
  --calibration artifacts/balanced_gigapass/eager_calibration.json \
  --attention-backend eager --pairs 48 --captures 8 --seed 20271001 \
  --detector-bundle artifacts/balanced_frozen_attackers_v1

python -m experiments.llama_balanced_gigapass.frozen_detector evaluate \
  --bundle artifacts/balanced_frozen_attackers_v1 \
  --test-root artifacts/balanced_eager_prospective48_v1 \
  --output artifacts/balanced_eager_prospective48_v1/frozen_evaluation.json
```

All destinations must be new. The runner refuses to silently overwrite or resume
a partial experiment. On failure, existing data and the original plan are retained;
no partial collection is silently scored as the full predeclared test.
