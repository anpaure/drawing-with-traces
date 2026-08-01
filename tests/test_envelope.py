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
    assert restored.extraction_mode == "height"


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


def test_alpha_only_black_logo_supports_upper_boundary(tmp_path):
    source = tmp_path / "transparent-logo.png"
    pixels = np.zeros((20, 32, 4), dtype=np.uint8)
    for column in range(2, 30):
        top = 14 - column // 3
        pixels[top:18, column, 3] = 255
    Image.fromarray(pixels, mode="RGBA").save(source)

    target = extract_envelope(
        source,
        points=16,
        smoothing_sigma_points=0,
        extraction_mode="upper-boundary",
    )

    assert target.extraction_mode == "upper-boundary"
    assert target.values[0] < target.values[-1]
    assert target.metadata()["extraction_mode"] == "upper-boundary"
    assert target.crop_bbox == (2, 5, 30, 18)


def test_upper_boundary_and_column_height_are_distinct(tmp_path):
    source = tmp_path / "floating-shape.png"
    pixels = np.zeros((24, 32, 4), dtype=np.uint8)
    for column in range(32):
        top = 3 + column // 5
        bottom = min(23, top + 4 + column // 4)
        pixels[top : bottom + 1, column, 3] = 255
    Image.fromarray(pixels, mode="RGBA").save(source)

    upper = extract_envelope(
        source, points=16, smoothing_sigma_points=0, extraction_mode="upper-boundary"
    )
    height = extract_envelope(source, points=16, smoothing_sigma_points=0, extraction_mode="height")

    assert upper.values[0] > upper.values[-1]
    assert height.values[0] < height.values[-1]
    assert not np.allclose(upper.values, height.values)


def test_invalid_extraction_mode_is_rejected(tmp_path):
    source = tmp_path / "shape.png"
    make_shape(source)
    with pytest.raises(ValueError, match="extraction_mode"):
        extract_envelope(source, extraction_mode="diagonal")
