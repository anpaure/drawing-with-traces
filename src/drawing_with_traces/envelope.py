"""Collapse a filled silhouette onto a baseline to obtain a power target."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sidecapture.errors import ConfigurationError
from sidecapture.store import atomic_json, atomic_numpy


@dataclass(frozen=True)
class PowerEnvelope:
    """A normalized, one-dimensional silhouette height profile."""

    values: np.ndarray
    source_sha256: str
    envelope_sha256: str
    source_size: tuple[int, int]
    crop_bbox: tuple[int, int, int, int]
    background_rgb: tuple[float, float, float]
    smoothing_sigma_points: float

    def metadata(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("values")
        value["source_size"] = list(self.source_size)
        value["crop_bbox"] = list(self.crop_bbox)
        value["background_rgb"] = list(self.background_rgb)
        value["points"] = int(self.values.size)
        value["minimum"] = float(self.values.min())
        value["maximum"] = float(self.values.max())
        return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def envelope_hash(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype="<f4")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return values.copy()
    radius = max(1, int(np.ceil(4 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(x / sigma))
    kernel /= kernel.sum()
    padded = np.pad(values, radius, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def extract_envelope(
    path: str | Path,
    *,
    points: int = 120,
    foreground_distance: float = 32.0,
    smoothing_sigma_points: float = 1.5,
) -> PowerEnvelope:
    """Flatten each foreground column to a baseline and retain its height."""

    if points < 8:
        raise ValueError("points must be at least 8")
    if foreground_distance <= 0:
        raise ValueError("foreground_distance must be positive")
    if smoothing_sigma_points < 0:
        raise ValueError("smoothing_sigma_points cannot be negative")
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ConfigurationError(f"silhouette image does not exist or is not a file: {path}")

    image = Image.open(path).convert("RGBA")
    pixels = np.asarray(image, dtype=np.float32)
    height, width = pixels.shape[:2]
    patch = max(1, min(height, width) // 50)
    corners = np.concatenate(
        [
            pixels[:patch, :patch].reshape(-1, 4),
            pixels[:patch, -patch:].reshape(-1, 4),
            pixels[-patch:, :patch].reshape(-1, 4),
            pixels[-patch:, -patch:].reshape(-1, 4),
        ]
    )
    background = np.median(corners[:, :3], axis=0)
    if float(np.median(corners[:, 3])) < 16:
        foreground = pixels[:, :, 3] >= 128
    else:
        distance = np.linalg.norm(pixels[:, :, :3] - background[None, None, :], axis=2)
        foreground = (distance >= foreground_distance) & (pixels[:, :, 3] >= 128)
    y, x = np.where(foreground)
    if not x.size:
        raise ConfigurationError(
            "no foreground was found; lower --foreground-distance or use a contrasting background"
        )
    left, top, right, bottom = int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1
    crop = foreground[top:bottom, left:right]
    raw_height = np.zeros(crop.shape[1], dtype=np.float64)
    for column in range(crop.shape[1]):
        rows = np.flatnonzero(crop[:, column])
        if rows.size:
            raw_height[column] = rows[-1] - rows[0] + 1
    nonzero = np.flatnonzero(raw_height > 0)
    if not nonzero.size:
        raise ConfigurationError("foreground extraction produced no non-empty columns")
    raw_height = np.interp(
        np.arange(raw_height.size, dtype=np.float64),
        nonzero.astype(np.float64),
        raw_height[nonzero],
    )
    resampled = np.interp(
        np.linspace(0, raw_height.size - 1, points),
        np.arange(raw_height.size),
        raw_height,
    )
    smoothed = _gaussian_smooth(resampled, smoothing_sigma_points)
    span = float(smoothed.max() - smoothed.min())
    if span <= 0:
        raise ConfigurationError("silhouette has constant column height and cannot define a power curve")
    values = np.ascontiguousarray((smoothed - smoothed.min()) / span, dtype=np.float32)
    return PowerEnvelope(
        values=values,
        source_sha256=sha256_file(path),
        envelope_sha256=envelope_hash(values),
        source_size=(width, height),
        crop_bbox=(left, top, right, bottom),
        background_rgb=tuple(float(value) for value in background),
        smoothing_sigma_points=float(smoothing_sigma_points),
    )


def save_envelope(root: str | Path, envelope: PowerEnvelope) -> None:
    root = Path(root)
    artwork = root / "artwork"
    values_path = artwork / "target_envelope.npy"
    metadata_path = artwork / "target.json"
    if values_path.exists() or metadata_path.exists():
        if not values_path.exists() or not metadata_path.exists():
            raise ConfigurationError(f"{artwork} contains a partial target; restore or remove it")
        existing_values = np.load(values_path, allow_pickle=False)
        existing_metadata = json.loads(metadata_path.read_text())
        if envelope_hash(existing_values) != envelope.envelope_sha256:
            raise ConfigurationError("stored target envelope does not match the requested silhouette")
        if existing_metadata != envelope.metadata():
            raise ConfigurationError("stored target metadata does not match the requested silhouette")
        return
    atomic_numpy(envelope.values, values_path)
    atomic_json(envelope.metadata(), metadata_path)


def load_envelope(root: str | Path) -> PowerEnvelope:
    root = Path(root)
    values_path = root / "artwork" / "target_envelope.npy"
    metadata_path = root / "artwork" / "target.json"
    if not values_path.exists() or not metadata_path.exists():
        raise ConfigurationError(f"{root} has no saved power envelope")
    values = np.load(values_path, allow_pickle=False).astype(np.float32)
    metadata = json.loads(metadata_path.read_text())
    if envelope_hash(values) != metadata["envelope_sha256"]:
        raise ConfigurationError("target_envelope.npy does not match target.json")
    return PowerEnvelope(
        values=values,
        source_sha256=str(metadata["source_sha256"]),
        envelope_sha256=str(metadata["envelope_sha256"]),
        source_size=tuple(metadata["source_size"]),
        crop_bbox=tuple(metadata["crop_bbox"]),
        background_rgb=tuple(metadata["background_rgb"]),
        smoothing_sigma_points=float(metadata["smoothing_sigma_points"]),
    )
