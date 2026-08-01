import numpy as np
import pytest

from drawing_with_traces.workload import TrainingDutyWorkload


def build(**overrides):
    options = dict(
        target_values=np.array([0, 0.5, 1], dtype=np.float32),
        duty_values=np.array([0, 0.25, 1], dtype=np.float32),
        profile_duration_s=3,
        quantum_s=0.1,
        hidden_size=4096,
        feedforward_size=8192,
        batch_size=256,
    )
    options.update(overrides)
    return TrainingDutyWorkload(**options)


def test_model_metadata_does_not_require_cuda():
    workload = build()
    assert workload.parameter_count == 67_108_864
    assert workload.segment_duration_s == 1
    assert workload.total_duration_s == 9
    assert workload.metadata()["transactional_update"] is True


def test_target_and_duty_must_match():
    with pytest.raises(ValueError, match="same shape"):
        build(duty_values=np.array([0, 1]))


def test_quantum_must_fit_inside_segment():
    with pytest.raises(ValueError, match="shorter"):
        build(quantum_s=1)


@pytest.mark.parametrize("duty", [-0.1, 1.1, float("nan")])
def test_invalid_duty_is_rejected(duty):
    with pytest.raises(ValueError):
        build(duty_values=np.array([0, duty, 1]))
