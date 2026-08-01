from __future__ import annotations

import pytest

from experiments.llama_continuous_whole_model.capture_continuous import (
    RandomPretriggerDelaySampler,
)
from experiments.llama_continuous_whole_model.analyze_continuous import (
    ContinuousTrace,
    grouped_ridge_accuracy,
    stationary_similarities,
    trace_windows,
)
from experiments.llama_continuous_whole_model.analyze_kernel_profiles import (
    family_fractions,
    gemm_geometry,
)
from experiments.llama_continuous_whole_model.continuous_workloads import (
    WorkerConfig,
    cover_layer_indices,
    cyclic_slice,
    sample_cover_token_count,
)
from experiments.llama_continuous_whole_model.cnn_detector import (
    PowerTraceCnn,
    held_out_mask,
    make_windows,
)
from experiments.llama_continuous_whole_model.training_shapes import (
    LinearShapeConfig,
    ShapedLinear,
    hybrid_weight_gradient,
    shape_linear_modules,
)

import torch
from torch import nn


def test_cyclic_slice_wraps_without_padding() -> None:
    assert cyclic_slice([10, 20, 30], 2, 5) == [30, 10, 20, 30, 10]


@pytest.mark.parametrize(
    ("values", "start", "length", "message"),
    [([], 0, 1, "at least one"), ([1], 0, 0, "positive")],
)
def test_cyclic_slice_rejects_invalid_input(values, start, length, message) -> None:
    with pytest.raises(ValueError, match=message):
        cyclic_slice(values, start, length)


def test_cover_layer_indices_select_regular_backward_boundaries() -> None:
    assert cover_layer_indices(32, 3) == list(range(0, 32, 3))
    with pytest.raises(ValueError, match="interval"):
        cover_layer_indices(32, 0)


def test_cover_token_jitter_is_bounded_and_reproducible() -> None:
    import random

    left = random.Random(9)
    right = random.Random(9)
    actual = [sample_cover_token_count(12, 4, left) for _ in range(20)]
    expected = [sample_cover_token_count(12, 4, right) for _ in range(20)]
    assert actual == expected
    assert min(actual) >= 8 and max(actual) <= 16


def test_worker_metadata_hashes_corpus_instead_of_storing_it() -> None:
    config = WorkerConfig(mode="training", session_id="train-00", corpus="private training text")
    metadata = config.metadata()

    assert "corpus" not in metadata
    assert metadata["corpus_characters"] == len("private training text")
    assert len(metadata["corpus_sha256"]) == 64
    assert metadata["optimizer"] == "adamw8bit"
    assert metadata["training_sequence_length"] == 128


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mode": "invalid"}, "mode"),
        ({"session_id": ""}, "session_id"),
        ({"training_sequence_length": 0}, "training_sequence_length"),
        ({"inference_quantization": "int2"}, "inference_quantization"),
        ({"optimizer": "magic"}, "optimizer"),
        ({"gradient_accumulation_steps": 0}, "gradient_accumulation_steps"),
        ({"linear_shaping": "magic"}, "linear_shaping"),
        ({"cover_decode_tokens_per_microbatch": -1}, "cover_decode_tokens"),
        (
            {"cover_decode_tokens_per_microbatch": 4, "cover_decode_token_jitter": 5},
            "cover_decode_token_jitter",
        ),
        ({"cover_backward_layer_interval": -1}, "cover_backward_layer_interval"),
    ],
)
def test_worker_config_rejects_invalid_values(kwargs, message) -> None:
    arguments = {"mode": "training", "session_id": "session", **kwargs}
    with pytest.raises(ValueError, match=message):
        WorkerConfig(**arguments)


class FakeSampler:
    def __init__(self) -> None:
        self.request = object()
        self.events: list[object] = []

    def open(self):
        self.events.append("open")

    def plan(self):
        self.events.append("plan")
        return "resolved"

    def arm(self):
        self.events.append("arm")

    def trigger(self):
        self.events.append("trigger")
        return "anchor"

    def observe_trigger(self, anchor):
        self.events.append(("observe", anchor))

    def finish(self):
        self.events.append("finish")
        return "batch"

    def abort(self):
        self.events.append("abort")

    def recover(self, error):
        self.events.append(("recover", error))
        return True

    def close(self):
        self.events.append("close")

    def metadata(self):
        return {"base": True}


def test_random_delay_sampler_delegates_lifecycle() -> None:
    base = FakeSampler()
    sampler = RandomPretriggerDelaySampler(
        base,
        minimum_delay_s=0,
        maximum_delay_s=0,
        seed=7,
    )

    sampler.open()
    assert sampler.plan() == "resolved"
    sampler.arm()
    assert sampler.trigger() == "anchor"
    sampler.observe_trigger("external")
    assert sampler.finish() == "batch"
    sampler.abort()
    error = RuntimeError("retry")
    assert sampler.recover(error)
    sampler.close()

    assert base.events == [
        "open",
        "plan",
        "arm",
        "trigger",
        ("observe", "external"),
        "finish",
        "abort",
        ("recover", error),
        "close",
    ]
    metadata = sampler.metadata()
    assert metadata["base"] is True
    assert metadata["random_pretrigger_delay"]["last_delay_s"] == 0


def test_hybrid_weight_gradient_is_exact() -> None:
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(3, 7, generator=generator, dtype=torch.float64)
    grad_output = torch.randn(3, 5, generator=generator, dtype=torch.float64)

    actual = hybrid_weight_gradient(
        x,
        grad_output,
        inference_rows=2,
        bias_epilogue=True,
    )
    expected = grad_output.T @ x
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_shaped_linear_matches_ordinary_forward_and_backward() -> None:
    generator = torch.Generator().manual_seed(19)
    ordinary = nn.Linear(7, 5, bias=True, dtype=torch.float64)
    source = nn.Linear(7, 5, bias=True, dtype=torch.float64)
    source.load_state_dict(ordinary.state_dict())
    shaped = ShapedLinear(
        source,
        LinearShapeConfig(
            forward_rows=1,
            input_gradient_rows=1,
            weight_gradient_inference_rows=2,
        ),
    )
    x = torch.randn(2, 3, 7, generator=generator, dtype=torch.float64, requires_grad=True)
    shaped_x = x.detach().clone().requires_grad_(True)
    upstream = torch.randn(2, 3, 5, generator=generator, dtype=torch.float64)
    ordinary(x).backward(upstream)
    shaped(shaped_x).backward(upstream)

    torch.testing.assert_close(shaped_x.grad, x.grad, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(shaped.weight.grad, ordinary.weight.grad, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(shaped.bias.grad, ordinary.bias.grad, rtol=1e-12, atol=1e-12)


def test_shape_linear_modules_replaces_every_linear_without_changing_parameters() -> None:
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Sequential(nn.Linear(3, 2)))
    parameter_ids = {id(parameter) for parameter in model.parameters()}

    names = shape_linear_modules(model, "token-row")

    assert names == ["0", "2.0"]
    assert isinstance(model[0], ShapedLinear)
    assert isinstance(model[2][0], ShapedLinear)
    assert {id(parameter) for parameter in model.parameters()} == parameter_ids


def test_kernel_profile_compaction_counts_m_equals_one_geometry() -> None:
    profile = {
        "top_kernel_families": [
            {"family": "dense_gemm", "total_us": 30},
            {"family": "elementwise", "total_us": 10},
        ],
        "aten_gemm_shapes": [
            {"name": "aten::mm", "input_shapes": [[1, 8], [8, 4]], "count": 7},
            {"name": "aten::mm", "input_shapes": [[3, 8], [8, 4]], "count": 2},
        ],
    }

    assert family_fractions(profile) == {"dense_gemm": 0.75, "elementwise": 0.25}
    geometry = gemm_geometry(profile)
    assert geometry["aten_gemm_calls"] == 9
    assert geometry["m_equals_1_calls"] == 7
    assert geometry["m_equals_1_fraction"] == pytest.approx(7 / 9)


def _synthetic_trace(process: str, session: int, index: int) -> ContinuousTrace:
    sample_rate = 100_000.0
    rng = torch.Generator().manual_seed(1000 + 10 * session + index)
    values = torch.randn(2000, generator=rng, dtype=torch.float64).numpy() * 0.02
    time_axis = torch.arange(2000, dtype=torch.float64).numpy() / sample_rate
    frequency = 4_000 if process == "inference" else 12_000
    values += 0.2 * torch.sin(torch.tensor(2 * torch.pi * frequency * time_axis)).numpy()
    values += session * 0.003  # Session-specific nuisance shift must be held out.
    return ContinuousTrace(
        process=process,
        session_id=f"{process}-{session}",
        index=index,
        sample_rate_hz=sample_rate,
        values=values,
        health_ok=True,
    )


def test_grouped_detector_holds_out_complete_sessions_and_uses_adc_windows() -> None:
    inference = [_synthetic_trace("inference", session, index) for session in range(2) for index in range(2)]
    training = [_synthetic_trace("training", session, index) for session in range(2) for index in range(2)]

    result = grouped_ridge_accuracy(inference, training, horizon_ms=5)

    assert result is not None
    assert result["attacker_input"] == "power samples only"
    assert result["split"].startswith("leave one complete inference session")
    assert len(result["folds"]) == 4
    assert result["balanced_accuracy"] > 0.95


def test_trace_analysis_shapes_and_similarity_ranges() -> None:
    inference = [_synthetic_trace("inference", 0, 0)]
    training = [_synthetic_trace("training", 0, 0)]
    windows = trace_windows(inference[0], horizon_ms=5)
    similarities = stationary_similarities(inference, training)

    assert windows.shape[0] == 4
    assert windows.shape[1] > 20
    assert all(0 <= value <= 1 for value in similarities.values())


def test_cnn_windows_and_session_mask_do_not_mix_held_out_sessions() -> None:
    traces = [
        _synthetic_trace(process, session, 0)
        for process in ("inference", "training")
        for session in range(2)
    ]
    windows = make_windows(traces, horizon_ms=5)
    mask = held_out_mask(windows, "inference-1", "training-0")

    assert len(windows.values) == 16
    assert mask.sum() == 8
    assert set(windows.sessions[mask]) == {"inference-1", "training-0"}


def test_power_trace_cnn_accepts_raw_windows() -> None:
    model = PowerTraceCnn(channels=8)
    output = model(torch.randn(3, 1, 7500))
    assert output.shape == (3, 2)
