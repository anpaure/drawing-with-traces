import numpy as np
import pytest

from drawing_with_traces.fast import FastCalibration, TiledLinearTrainingWorkload


def calibration():
    return FastCalibration(
        widths=np.array([0, 32, 128, 512, 4096], dtype=float),
        feature_name="rms",
        feature_sign=-1,
        measured_feature=np.array([5, 4, 3, 2, 1], dtype=float),
        monotonic_activity=np.array([-5, -4, -3, -2, -1], dtype=float),
        within_width_std=np.full(5, 0.01),
    )


def test_fast_calibration_maps_target_to_supported_widths():
    commands = calibration().commands_for(np.array([0, 0.25, 0.5, 0.75, 1]))
    assert commands[0] == 0
    assert commands[-1] == 4096
    assert set(commands) <= {0, 32, 128, 512, 4096}


def test_fast_calibration_round_trip():
    original = calibration()
    restored = FastCalibration.from_dict(original.to_dict())
    assert restored.feature_name == "rms"
    assert restored.feature_sign == -1
    assert np.array_equal(restored.widths, original.widths)


def test_tiled_workload_metadata_is_transactional_without_cuda():
    workload = TiledLinearTrainingWorkload(
        np.array([0, 32, 128, 4096]),
        profile_duration_s=0.01,
        hidden_size=4096,
        batch_size=512,
    )
    assert workload.bin_duration_s == pytest.approx(0.0025)
    assert workload.parameter_count == 16_777_216
    assert workload.metadata()["transactional_update"] is True
    assert "X.T @" in workload.metadata()["gradient"]


@pytest.mark.parametrize("commands", [[16, 32], [-1, 32], [32, 8192]])
def test_tiled_workload_rejects_invalid_widths(commands):
    with pytest.raises(ValueError):
        TiledLinearTrainingWorkload(
            np.asarray(commands),
            profile_duration_s=0.01,
            hidden_size=4096,
        )
