#!/usr/bin/env python3
from __future__ import annotations

import collections
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "openai/gpt-oss-20b"
ROOT = Path(os.environ.get("EXPERIMENT_ROOT", "runs/gpt_oss_kernel_profile"))
ROOT.mkdir(parents=True, exist_ok=True)


def tensor_nbytes(tensor: torch.Tensor) -> int:
    try:
        return tensor.numel() * tensor.element_size()
    except Exception:
        return 0


def parameter_inventory(model) -> dict:
    by_dtype = collections.defaultdict(lambda: {"tensors": 0, "numel": 0, "bytes": 0, "trainable_numel": 0})
    largest = []
    total_numel = 0
    trainable_numel = 0
    total_bytes = 0
    for name, p in model.named_parameters():
        key = str(p.dtype)
        n = int(p.numel())
        b = int(tensor_nbytes(p))
        total_numel += n
        total_bytes += b
        if p.requires_grad:
            trainable_numel += n
        row = by_dtype[key]
        row["tensors"] += 1
        row["numel"] += n
        row["bytes"] += b
        row["trainable_numel"] += n if p.requires_grad else 0
        largest.append({"name": name, "shape": list(p.shape), "dtype": key, "numel": n, "bytes": b, "requires_grad": bool(p.requires_grad), "class": type(p).__qualname__})
    largest.sort(key=lambda x: x["bytes"], reverse=True)
    return {
        "total_numel": total_numel,
        "trainable_numel": trainable_numel,
        "total_bytes": total_bytes,
        "by_dtype": dict(by_dtype),
        "largest": largest[:120],
    }


def event_summary(prof, name: str, wall_ms: float) -> dict:
    raw = list(prof.events())
    cuda = [e for e in raw if str(e.device_type).endswith("CUDA")]
    kernels = []
    by_kernel = collections.defaultdict(lambda: {"count": 0, "total_us": 0.0, "max_us": 0.0})
    for e in cuda:
        dur = float(getattr(e, "device_time_total", 0.0) or getattr(e, "cuda_time_total", 0.0) or 0.0)
        key = str(e.key)
        row = by_kernel[key]
        row["count"] += 1
        row["total_us"] += dur
        row["max_us"] = max(row["max_us"], dur)
        tr = getattr(e, "time_range", None)
        kernels.append({
            "name": key,
            "duration_us": dur,
            "start_us": None if tr is None else float(tr.start),
            "correlation_id": int(getattr(e, "correlation_id", -1)),
        })
    kernels.sort(key=lambda x: (math.inf if x["start_us"] is None else x["start_us"]))
    top = sorted(
        ({"name": k, **v} for k, v in by_kernel.items()),
        key=lambda x: x["total_us"],
        reverse=True,
    )
    ops = []
    for e in raw:
        key = str(e.key)
        if not key.startswith("aten::"):
            continue
        shapes = getattr(e, "input_shapes", None)
        if key in {
            "aten::mm", "aten::matmul", "aten::bmm", "aten::addmm", "aten::linear",
            "aten::_scaled_mm", "aten::scaled_dot_product_attention",
            "aten::_scaled_dot_product_flash_attention_for_cuda",
        } or shapes:
            ops.append({
                "name": key,
                "input_shapes": shapes,
                "cpu_time_us": float(getattr(e, "cpu_time_total", 0.0) or 0.0),
                "device_time_us": float(getattr(e, "device_time_total", 0.0) or 0.0),
            })
    return {
        "phase": name,
        "wall_ms": wall_ms,
        "cuda_kernel_count": len(cuda),
        "cuda_kernel_total_us": sum(x["duration_us"] for x in kernels),
        "top_cuda_kernels": top[:120],
        "cuda_kernel_sequence": kernels,
        "aten_ops": ops,
    }


def run_profile(name: str, fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=False) as prof:
        with record_function(name):
            value = fn()
        torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1e3
    prof.export_chrome_trace(str(ROOT / f"{name}.chrome.json"))
    result = event_summary(prof, name, wall_ms)
    result["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    result["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    (ROOT / f"{name}.summary.json").write_text(json.dumps(result, indent=2))
    print(f"PROFILE {name}: wall={wall_ms:.3f}ms kernels={result['cuda_kernel_count']} cuda={result['cuda_kernel_total_us']/1000:.3f}ms", flush=True)
    return value, result


print("loading", MODEL, flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    local_files_only=True,
    dtype=torch.bfloat16,
    device_map=torch.cuda.current_device(),
)
model.config.use_cache = True
inventory = parameter_inventory(model)
(ROOT / "parameter_inventory.json").write_text(json.dumps(inventory, indent=2))
print(json.dumps({k: inventory[k] for k in ["total_numel", "trainable_numel", "total_bytes", "by_dtype"]}, indent=2), flush=True)

batch = tokenizer("Training can imitate inference", return_tensors="pt", add_special_tokens=False)
ids = batch["input_ids"][:, :2].to("cuda")
if ids.shape[1] < 2:
    raise RuntimeError("tokenizer produced fewer than two tokens")
mask = torch.ones_like(ids)
labels = ids.clone()

# Decode target: a real cached one-token GPT-OSS decode.
model.eval()
with torch.inference_mode():
    prefill = model(input_ids=ids[:, :1], attention_mask=mask[:, :1], use_cache=True, logits_to_keep=1)
    past = prefill.past_key_values
    decode_id = ids[:, 1:2]
    decode_mask = mask[:, :2]
    for _ in range(2):
        _ = model(input_ids=decode_id, attention_mask=decode_mask, past_key_values=past, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()

def decode_once():
    with torch.inference_mode():
        return model(input_ids=decode_id, attention_mask=decode_mask, past_key_values=past, use_cache=True, logits_to_keep=1)

decode_out, decode_summary = run_profile("decode", decode_once)
decode_checksum = float(decode_out.logits.float().sum().cpu())

# Real causal-LM training. Optimizer is deliberately SGD to avoid a large state tensor;
# it still updates every exposed trainable GPT-OSS parameter.
model.train()
optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=1e-9)
optimizer.zero_grad(set_to_none=True)
out = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)
out.loss.backward()
optimizer.zero_grad(set_to_none=True)
torch.cuda.synchronize()

def training_forward():
    return model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)

train_out, forward_summary = run_profile("training_forward", training_forward)
loss = train_out.loss
_, backward_summary = run_profile("training_backward", loss.backward)
nonzero_grad_params = sum(1 for p in model.parameters() if p.grad is not None and bool(torch.count_nonzero(p.grad).item()))
grad_params = sum(1 for p in model.parameters() if p.grad is not None)
grad_norm_sq = 0.0
for p in model.parameters():
    if p.grad is not None:
        grad_norm_sq += float(p.grad.float().square().sum().cpu())
grad_norm = math.sqrt(grad_norm_sq)

# Save one representative tensor before/after to prove the update changed GPT-OSS weights.
selected_name, selected = next((n, p) for n, p in model.named_parameters() if p.requires_grad and p.grad is not None and bool(torch.count_nonzero(p.grad).item()))
before = selected.detach().float().clone()
_, step_summary = run_profile("optimizer_step", optimizer.step)
max_update = float((selected.detach().float() - before).abs().max().cpu())
optimizer.zero_grad(set_to_none=True)

meta = {
    "model": MODEL,
    "tokens": ids.detach().cpu().tolist(),
    "decode_checksum": decode_checksum,
    "training_loss": float(loss.detach().float().cpu()),
    "parameter_inventory": {k: inventory[k] for k in ["total_numel", "trainable_numel", "total_bytes", "by_dtype"]},
    "grad_parameter_tensors": grad_params,
    "nonzero_grad_parameter_tensors": nonzero_grad_params,
    "gradient_l2_norm": grad_norm,
    "updated_parameter": selected_name,
    "updated_parameter_max_absolute_delta": max_update,
    "profiles": {s["phase"]: {k: s[k] for k in ["wall_ms", "cuda_kernel_count", "cuda_kernel_total_us", "peak_allocated_bytes", "peak_reserved_bytes"]} for s in [decode_summary, forward_summary, backward_summary, step_summary]},
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
}
(ROOT / "profile_summary.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2), flush=True)
