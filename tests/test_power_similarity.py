from __future__ import annotations

import numpy as np

from experiments.gpt_oss_inference_shaped_training.analyze_power_capture import (
    TraceRecord,
    describe_experiment,
    histogram_similarity,
    js_similarity,
)


def test_js_similarity_identity_and_separation() -> None:
    distribution = np.asarray([1.0, 2.0, 3.0])
    assert js_similarity(distribution, distribution) == 1.0
    assert js_similarity(np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0])) == 0.0


def test_histogram_similarity_ignores_repeat_count() -> None:
    left = np.asarray([-1.0, 0.0, 1.0])
    right = np.tile(left, 10)
    assert histogram_similarity(left, right, bins=3) == 1.0


def test_experiment_title_comes_from_capture_metadata() -> None:
    scope = [
        f"model.layers.{layer}.self_attn.{projection}.weight"
        for layer in range(24)
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
    ]
    common = {
        "labels": {
            "model": "openai/gpt-oss-20b",
            "parameter_scope": scope,
            "cover_decodes_per_training_step": 8,
            "variant": "inference_shaped",
        }
    }
    inference = TraceRecord(
        process="inference",
        index=0,
        phase="inference",
        phase_offset=None,
        sample_rate_hz=1_500_000,
        values=np.zeros(10),
        metadata={**common, "labels": {**common["labels"], "task": "cached_decode"}},
    )
    training = TraceRecord(
        process="training",
        index=1,
        phase="training",
        phase_offset=0,
        sample_rate_hz=1_500_000,
        values=np.zeros(10),
        metadata=common,
    )

    assert describe_experiment([inference], [training]) == (
        "gpt-oss-20b: cached decode vs inference-shaped training of all 96 "
        "attention-projection weights "
        "+ 8 cached-decode covers/step"
    )
