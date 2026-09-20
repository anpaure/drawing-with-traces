from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from experiments.llama_balanced_gigapass.contextual_amp import ContextualAMPConfig, ContextualAMPEngine, save_result
from experiments.llama_balanced_gigapass.workload import BalancedGigaPassEngine


@pytest.mark.parametrize("kwargs,match", [
    ({"sequence_length": 1}, "sequence_length"),
    ({"parameter_dtype": "bfloat16"}, "FP32 masters"),
    ({"autocast_dtype": "float16"}, "BF16 autocast"),
])
def test_contextual_config_rejects_degenerate_or_wrong_precision(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ContextualAMPConfig(**kwargs)


def test_contextual_candidate_reuses_the_validated_stream_schedule():
    assert ContextualAMPEngine.run_mixed_batches is BalancedGigaPassEngine.run_mixed_batches
    config = ContextualAMPConfig()
    assert config.training_batch_size * config.sequence_length == 1024
    assert not config.autocast_cache_enabled


def tiny_cpu_engine():
    torch = pytest.importorskip("torch")
    torch.manual_seed(44)

    class CausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(32, 8)
            self.q_proj = torch.nn.Linear(8, 8, bias=False)
            self.k_proj = torch.nn.Linear(8, 8, bias=False)
            self.v_proj = torch.nn.Linear(8, 8, bias=False)
            self.lm_head = torch.nn.Linear(8, 32, bias=False)

        def forward(self, input_ids, use_cache=False):
            hidden = self.embedding(input_ids)
            q, k, v = self.q_proj(hidden), self.k_proj(hidden), self.v_proj(hidden)
            scores = q @ k.transpose(-1, -2) / (8 ** 0.5)
            mask = torch.ones(scores.shape[-2:], dtype=torch.bool).tril()
            attended = scores.masked_fill(~mask, float("-inf")).softmax(-1) @ v
            return SimpleNamespace(logits=self.lm_head(hidden + attended))

    engine = ContextualAMPEngine(ContextualAMPConfig(
        training_batch_size=2, sequence_length=4, inference_batch_candidates=(2, 3), data_ring_size=2,
    ))
    # Unit test the actual AMP enqueue routines on CPU; never call CUDA setup.
    engine.device_type = "cpu"
    engine.model = CausalLM()
    engine.optimizer = torch.optim.SGD(engine.model.parameters(), lr=engine.config.learning_rate, foreach=True)
    engine.training_ids = torch.arange(16).reshape(2, 2, 4)
    engine.training_targets = engine.training_ids + 1
    engine.inference_ids = torch.arange(24).reshape(2, 3, 4)
    return engine


def test_amp_backward_matches_independent_reference_and_updates_fp32_masters():
    torch = pytest.importorskip("torch")
    engine = tiny_cpu_engine()
    reference = copy.deepcopy(engine.model)
    initial = copy.deepcopy(engine.model.state_dict())
    optimizer = torch.optim.SGD(reference.parameters(), lr=engine.config.learning_rate, foreach=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
        logits = reference(engine.training_ids[0]).logits
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 32).float(),
                                                engine.training_targets[0].reshape(-1))
    loss.backward()
    optimizer.step()

    backward_autocast_states = []
    handle = engine.model.q_proj.weight.register_hook(
        lambda gradient: backward_autocast_states.append(torch.is_autocast_enabled("cpu")))
    engine._enqueue_training_backward()
    engine._enqueue_optimizer_step()
    handle.remove()
    assert backward_autocast_states == [False]
    assert torch.equal(loss, engine._last_loss)
    for (name, parameter), (reference_name, expected) in zip(engine.model.named_parameters(), reference.named_parameters()):
        assert name == reference_name
        assert parameter.dtype == parameter.grad.dtype == torch.float32
        assert torch.equal(parameter, expected)
        assert torch.equal(parameter.grad, expected.grad)
    for name in ("q_proj.weight", "k_proj.weight"):
        parameter = dict(engine.model.named_parameters())[name]
        assert torch.count_nonzero(parameter.grad) > 0
        assert torch.count_nonzero(parameter != initial[name]) > 0


def test_amp_inference_processes_context_without_parameter_mutation():
    torch = pytest.importorskip("torch")
    engine = tiny_cpu_engine()
    initial = copy.deepcopy(engine.model.state_dict())
    engine._enqueue_inference(2)
    assert engine.total_inference_outputs == 8
    assert engine._last_inference_checksum.dtype == torch.int64
    assert all(p.grad is None for p in engine.model.parameters())
    assert all(torch.equal(value, initial[name]) for name, value in engine.model.state_dict().items())


def test_failed_numerical_run_is_saved_without_invalid_json(tmp_path):
    import json
    destination = tmp_path / "result.json"
    save_result(destination, {"status": "failed", "loss": float("nan"), "norms": [float("inf")]})
    assert json.loads(destination.read_text()) == {"status": "failed", "loss": "nan", "norms": ["inf"]}
    assert not destination.with_suffix(".json.tmp").exists()
