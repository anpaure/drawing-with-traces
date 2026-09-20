# Contextual AMP candidate — hardware verification pending

The existing single-token/BF16-storage result remains unchanged. This separate
candidate removes two limitations before further throughput optimization:

- **Context length 128:** attention spans multiple causal positions. The validator
  requires nonzero query/key gradients and changed query/key weights in every layer.
- **FP32 parameter storage and SGD updates:** BF16 is selected for eligible forward
  operations by autocast. There is no separate low-precision optimizer master copy.

Forward/loss run under autocast, then backward and the optimizer run outside it,
following [PyTorch's AMP guidance](https://docs.pytorch.org/docs/2.10/amp.html).
The AMP cache is disabled in both native and mixed runs. TF32 remains disabled.

## Unchanged scheduling principle

One uncached causal inference forward and one full forward/loss/backward/SGD update
share the same model. Their CUDA streams may overlap. The optimizer waits for
inference; the next iteration waits for the optimizer. The stream-scheduling
implementation is inherited unchanged from the validated baseline.

The default training batch is eight sequences (1,024 tokens). Inference batch size
is calibrated in-process from 16 through 72 sequences to target equal native GPU
service. Nothing is padded with a filler workload. Both products are computed.

This is **prefill-style inference**, not a KV-cache generation server. Data remain
deterministic synthetic token sequences. It tests real gradients and updates, not
language-model quality. It does not claim to resemble inference-only serving.

## Required hardware checks

1. Full-model sequential AMP reference versus mixed execution, starting from the
   same state and consuming identical inputs.
2. Bitwise equality of every parameter, gradient, captured inference-head output,
   and final loss after two iterations; all checked tensors finite.
3. FP32 stored parameters/gradients, BF16 inference-head outputs, and nonzero
   gradients **and actual updates** for all attention query/key projection weights.
4. Twelve rotated native/mixed timing rounds, four iterations per block, with raw
   timings retained. Report the 2× target separately from the 1% measurement margin.

CPU tests exercise AMP enqueue logic on a small causal-attention network and check
that backward runs outside autocast. **They do not establish H100 correctness or
throughput.** The H100 screen is queued behind the frozen capture/evaluation job and
will refuse to start if another compute process owns the GPU.

```bash
python -m experiments.llama_balanced_gigapass.contextual_amp \
  --attention-backend sdpa --output runs/contextual_amp/sdpa.json
python -m experiments.llama_balanced_gigapass.contextual_amp \
  --attention-backend eager --output runs/contextual_amp/eager.json
```

Each output path must be new. Numerical failures and configuration are retained in
the result JSON; non-finite values are explicitly serialized as diagnostic strings.
No speedup is claimed yet. Any useful candidate still needs its own SideCapture
measurements before replacing the capture-tested baseline.
