"""Draw silhouettes with measured side-channel traces from real model training."""

from .analysis import CalibrationCurve, analyze_drawing, render_drawing
from .envelope import PowerEnvelope, extract_envelope, load_envelope, save_envelope
from .fast import FastCalibration, FastRefinementController, TiledLinearTrainingWorkload
from .workload import AdaptiveTrainingPowerWorkload, TrainingDutyWorkload

__all__ = [
    "CalibrationCurve",
    "AdaptiveTrainingPowerWorkload",
    "PowerEnvelope",
    "FastCalibration",
    "FastRefinementController",
    "TiledLinearTrainingWorkload",
    "TrainingDutyWorkload",
    "analyze_drawing",
    "extract_envelope",
    "load_envelope",
    "render_drawing",
    "save_envelope",
]
