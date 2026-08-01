import numpy as np

from drawing_with_traces.analysis import CalibrationCurve, _isotonic_increasing
from drawing_with_traces.cli import parse_levels


def test_isotonic_regression_removes_decrease():
    fitted = _isotonic_increasing(np.array([50, 100, 90, 200], dtype=float))
    assert np.all(np.diff(fitted) >= 0)
    assert fitted.tolist() == [50, 95, 95, 200]


def test_calibration_inverts_measured_power_curve():
    curve = CalibrationCurve(
        duty=np.array([0, 0.5, 1.0]),
        measured_power_w=np.array([60, 140, 300]),
        monotonic_power_w=np.array([60, 140, 300]),
    )
    duties = curve.duties_for(np.array([0, 0.5, 1]))
    assert duties[0] == 0
    assert duties[-1] == 1
    assert 0.5 < duties[1] < 1


def test_parse_levels_sorts_deduplicates_and_requires_endpoints():
    assert parse_levels("1,.5,0,.5").tolist() == [0, 0.5, 1]


def test_calibration_serialization_round_trip():
    original = CalibrationCurve(
        np.array([0, 1]),
        np.array([55, 300]),
        np.array([55, 300]),
    )
    restored = CalibrationCurve.from_dict(original.to_dict())
    assert np.array_equal(restored.duty, original.duty)
    assert np.array_equal(restored.monotonic_power_w, original.monotonic_power_w)

