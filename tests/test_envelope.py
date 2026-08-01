import numpy as np
import pytest
from PIL import Image, ImageDraw

from sidecapture.errors import ConfigurationError

from drawing_with_traces.envelope import extract_envelope, load_envelope, save_envelope


def make_shape(path):
    image = Image.new("RGB", (160, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.polygon([(10, 60), (10, 40), (40, 15), (80, 50), (120, 25), (150, 55), (150, 60)], fill="black")
    image.save(path)


def test_extract_envelope_lowers_columns_and_normalizes(tmp_path):
    source = tmp_path / "shape.png"
    make_shape(source)
    target = extract_envelope(source, points=64, smoothing_sigma_points=0)
    assert target.values.shape == (64,)
    assert target.values.dtype == np.float32
    assert target.values.min() == pytest.approx(0)
    assert target.values.max() == pytest.approx(1)
    assert target.values[15] > target.values[35]
    assert target.values[50] > target.values[35]


def test_smoothing_reduces_step_roughness(tmp_path):
    source = tmp_path / "shape.png"
    make_shape(source)
    rough = extract_envelope(source, points=80, smoothing_sigma_points=0).values
    smooth = extract_envelope(source, points=80, smoothing_sigma_points=2).values
    assert np.abs(np.diff(smooth, n=2)).sum() < np.abs(np.diff(rough, n=2)).sum()


def test_envelope_round_trip_and_idempotent_save(tmp_path):
    source = tmp_path / "shape.png"
    make_shape(source)
    target = extract_envelope(source, points=32)
    save_envelope(tmp_path / "run", target)
    save_envelope(tmp_path / "run", target)
    restored = load_envelope(tmp_path / "run")
    assert np.array_equal(restored.values, target.values)
    assert restored.metadata() == target.metadata()


def test_save_refuses_different_target(tmp_path):
    source = tmp_path / "shape.png"
    make_shape(source)
    save_envelope(tmp_path / "run", extract_envelope(source, points=32))
    with pytest.raises(ConfigurationError, match="does not match"):
        save_envelope(tmp_path / "run", extract_envelope(source, points=48))


def test_blank_image_is_rejected(tmp_path):
    source = tmp_path / "blank.png"
    Image.new("RGB", (20, 20), "white").save(source)
    with pytest.raises(ConfigurationError, match="no foreground"):
        extract_envelope(source)
