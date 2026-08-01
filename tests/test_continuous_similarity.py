from __future__ import annotations

from experiments.gpt_oss_inference_shaped_training.continuous_similarity import (
    continuous_kernel_similarity,
)


def event(name: str, family: str, start: float, duration: float) -> dict[str, object]:
    return {"name": name, "family": family, "start_us": start, "duration_us": duration}


def repeat_stream(stream: list[dict[str, object]], repetitions: int) -> list[dict[str, object]]:
    span = max(float(row["start_us"]) + float(row["duration_us"]) for row in stream)
    return [
        {**row, "start_us": float(row["start_us"]) + repetition * span}
        for repetition in range(repetitions)
        for row in stream
    ]


def test_identical_stream_is_one() -> None:
    stream = [event("a", "gemm", 0, 4), event("b", "elementwise", 6, 2)]
    scores = continuous_kernel_similarity(stream, stream)
    assert all(score == 1.0 for score in scores.values())


def test_repeat_count_does_not_materially_change_continuous_score() -> None:
    stream = [event("a", "gemm", 0, 4), event("b", "elementwise", 6, 2)]
    scores = continuous_kernel_similarity(repeat_stream(stream, 3), repeat_stream(stream, 9))
    assert scores["family_duration_js_similarity"] == 1.0
    assert scores["exact_kernel_duration_js_similarity"] == 1.0
    assert scores["family_bigram_js_similarity"] == 1.0
    assert scores["family_trigram_js_similarity"] == 1.0
    assert scores["continuous_kernel_similarity"] > 0.99


def test_different_kernel_process_scores_lower() -> None:
    inference = [event("gemm", "gemm", 0, 5), event("add", "elementwise", 6, 2)]
    training = [event("reduce", "reduction", 0, 5), event("reduce", "reduction", 7, 5)]
    scores = continuous_kernel_similarity(inference, training)
    assert scores["family_duration_js_similarity"] < 0.2
    assert scores["family_bigram_js_similarity"] < 0.2
    assert scores["continuous_kernel_similarity"] < 0.6
