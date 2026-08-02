# Identical-microkernel carrier

This experiment replaces every linear GEMM used by the custom Llama inference
and exact-training paths with one runtime-shape/stride Triton kernel.

## What is proved

- Inference and training load the same Triton kernel binary (same Triton hash
  and cubin SHA-256).
- The kernel uses fixed `64 x 64 x 32` tiles, four warps, and three stages.
- `M`, `N`, `K`, and every matrix stride remain runtime arguments, so Triton
  does not specialize separate binaries for the observed GEMM geometries.
- A full 1.235B-parameter update matches the ordinary implementation exactly:
  loss, all gradients, and all updated parameters have zero measured error.
- Every carrier GEMM in this baseline contributes useful training arithmetic;
  it uses no inference cover traffic or filler kernels.

## What is *not* identical

The two modes do **not** execute the same complete temporal program. They share
the GEMM *binary*, but inference and training still have different GEMM
dimensions, launch counts, order, nonlinear kernels, optimizer kernels, memory
traffic, and dependency-driven gaps. In other words, this experiment matches
the instruction alphabet, not the sentence.

## Physical result

Five independently captured sessions per class produced very similar stationary
power distributions, but a fresh session-held-out ridge attacker still separated
the modes. Balanced accuracy was 82.9%, 89.3%, 92.2%, 94.8%, and 97.0% at
5, 10, 20, 50, and 100 ms respectively. The required gate is below 60% at every
horizon, so this branch is a negative baseline.

The next experiment must align the ordered, cyclic computation-block streams,
not tune this carrier's stationary distribution further. Its intended construction
is a minimum-cost cyclic shortest common supersequence: each real workload is a
subsequence of one common schedule, while missing blocks execute with realistic
scratch operands and discarded outputs.

See `results/llama_identical_microkernel_carrier/` for the compact evidence.
