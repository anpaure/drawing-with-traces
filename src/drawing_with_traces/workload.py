"""Real CUDA training with a time-varying duty cycle."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from sidecapture.errors import ConfigurationError
from sidecapture.workloads import Workload


def sleep_until(deadline_ns: int) -> None:
    while True:
        remaining_ns = deadline_ns - time.monotonic_ns()
        if remaining_ns <= 0:
            return
        if remaining_ns > 300_000:
            time.sleep((remaining_ns - 150_000) / 1e9)


class TrainingDutyWorkload(Workload):
    """Modulate genuine BF16 forward/backward work using a duty schedule.

    Backward passes happen throughout the measured interval. Parameters mutate
    only in ``on_accept`` after SideCapture has durably committed the telemetry,
    making rejected captures safe to retry without double-applying updates.
    """

    replay_safe = True

    def __init__(
        self,
        target_values: np.ndarray,
        duty_values: np.ndarray,
        *,
        profile_duration_s: float,
        quantum_s: float = 0.02,
        lead_s: float = 3.0,
        tail_s: float = 3.0,
        hidden_size: int = 4096,
        feedforward_size: int = 8192,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        gradient_scale: float = 1e-5,
        seed: int = 1729,
    ):
        target_values = np.asarray(target_values, dtype=np.float32)
        duty_values = np.asarray(duty_values, dtype=np.float32)
        if target_values.ndim != 1 or target_values.size < 2:
            raise ValueError("target_values must be a one-dimensional array with at least two points")
        if target_values.shape != duty_values.shape:
            raise ValueError("target_values and duty_values must have the same shape")
        if not np.all(np.isfinite(target_values)) or not np.all(np.isfinite(duty_values)):
            raise ValueError("target and duty values must be finite")
        if np.any((target_values < 0) | (target_values > 1)):
            raise ValueError("target_values must remain between 0 and 1")
        if np.any((duty_values < 0) | (duty_values > 1)):
            raise ValueError("duty_values must remain between 0 and 1")
        if profile_duration_s <= 0 or quantum_s <= 0:
            raise ValueError("profile_duration_s and quantum_s must be positive")
        if quantum_s >= profile_duration_s / target_values.size:
            raise ValueError("quantum_s must be shorter than one target segment")
        if lead_s < 0 or tail_s < 0:
            raise ValueError("lead_s and tail_s cannot be negative")
        if min(hidden_size, feedforward_size, batch_size) < 1:
            raise ValueError("model and batch dimensions must be positive")
        if learning_rate <= 0 or gradient_scale <= 0:
            raise ValueError("learning_rate and gradient_scale must be positive")
        self.target_values = np.ascontiguousarray(target_values)
        self.duty_values = np.ascontiguousarray(duty_values)
        self.profile_duration_s = float(profile_duration_s)
        self.quantum_s = float(quantum_s)
        self.lead_s = float(lead_s)
        self.tail_s = float(tail_s)
        self.hidden_size = int(hidden_size)
        self.feedforward_size = int(feedforward_size)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.gradient_scale = float(gradient_scale)
        self.seed = int(seed)
        self.torch = None
        self.model = None
        self.optimizer = None
        self.input_batch = None
        self.training_target = None
        self._pending_gradient = False
        self._latest_loss = None

    @property
    def parameter_count(self) -> int:
        return 2 * self.hidden_size * self.feedforward_size

    @property
    def segment_duration_s(self) -> float:
        return self.profile_duration_s / self.target_values.size

    @property
    def total_duration_s(self) -> float:
        return self.lead_s + self.profile_duration_s + self.tail_s

    def setup(self) -> None:
        try:
            import torch
        except ImportError as error:
            raise ConfigurationError("training requires PyTorch; install the project dependencies") from error
        if not torch.cuda.is_available():
            raise ConfigurationError("drawing-with-traces requires a CUDA GPU")
        self.torch = torch
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.feedforward_size, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(self.feedforward_size, self.hidden_size, bias=False),
        ).cuda().to(torch.bfloat16)
        self.input_batch = torch.randn(
            self.batch_size, self.hidden_size, device="cuda", dtype=torch.bfloat16
        )
        self.training_target = torch.randn_like(self.input_batch)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        torch.cuda.synchronize()

    def require_setup(self):
        if self.torch is None or self.model is None or self.optimizer is None:
            raise RuntimeError("TrainingDutyWorkload.setup() has not completed")
        return self.torch

    def microstep(self) -> None:
        torch = self.require_setup()
        prediction = self.model(self.input_batch)
        loss = torch.nn.functional.mse_loss(prediction.float(), self.training_target.float())
        (loss * self.gradient_scale).backward()
        self._latest_loss = loss.detach()
        self._pending_gradient = True

    def warmup(self, iteration: int) -> None:
        del iteration
        self.optimizer.zero_grad(set_to_none=True)
        self.microstep()
        self.torch.cuda.synchronize()
        self.optimizer.zero_grad(set_to_none=True)
        self._pending_gradient = False

    def run(self, context):
        torch = self.require_setup()
        self.optimizer.zero_grad(set_to_none=True)
        self._pending_gradient = False
        context.labels.update(
            task="silhouette-shaped-model-training",
            profile_points=int(self.target_values.size),
            profile_duration_s=self.profile_duration_s,
            model_parameters=self.parameter_count,
        )
        context.add_artifact("target_values", self.target_values)
        context.add_artifact("duty_values", self.duty_values)

        with context.region("power_target.lead", duration_s=self.lead_s):
            sleep_until(time.monotonic_ns() + int(round(self.lead_s * 1e9)))

        segment_ns = int(round(self.segment_duration_s * 1e9))
        quantum_ns = int(round(self.quantum_s * 1e9))
        counts = np.zeros(self.target_values.size, dtype=np.int64)
        max_overrun_ns = 0
        with context.region(
            "power_target.profile",
            points=int(self.target_values.size),
            duration_s=self.profile_duration_s,
        ):
            for index, (target, duty) in enumerate(zip(self.target_values, self.duty_values)):
                segment_start = time.monotonic_ns()
                segment_end = segment_start + segment_ns
                with context.region(
                    "power_target.segment",
                    index=index,
                    target_normalized=float(target),
                    duty_cycle=float(duty),
                ):
                    while time.monotonic_ns() < segment_end:
                        quantum_start = time.monotonic_ns()
                        quantum_end = min(segment_end, quantum_start + quantum_ns)
                        active_ns = int(round((quantum_end - quantum_start) * float(duty)))
                        active_end = quantum_start + active_ns
                        while time.monotonic_ns() < active_end:
                            self.microstep()
                            torch.cuda.synchronize()
                            counts[index] += 1
                        sleep_until(quantum_end)
                        max_overrun_ns = max(max_overrun_ns, time.monotonic_ns() - quantum_end)
                sleep_until(segment_end)
        with context.region("power_target.tail", duration_s=self.tail_s):
            sleep_until(time.monotonic_ns() + int(round(self.tail_s * 1e9)))

        if max_overrun_ns > max(5_000_000, quantum_ns // 2):
            raise RuntimeError(
                f"training scheduler overran a quantum by {max_overrun_ns / 1e6:.3f} ms; "
                "discarding the timing-distorted capture"
            )
        loss = None if self._latest_loss is None else float(self._latest_loss)
        if loss is not None and not math.isfinite(loss):
            raise RuntimeError(f"training loss became non-finite: {loss}")
        context.labels.update(
            training_microsteps=int(counts.sum()),
            training_loss_before_update=loss,
            max_quantum_overrun_us=max_overrun_ns / 1e3,
        )
        context.add_artifact("microsteps_per_segment", counts)
        return {
            "training_microsteps": int(counts.sum()),
            "loss_before_update": loss,
            "max_quantum_overrun_us": max_overrun_ns / 1e3,
        }

    def on_accept(self, result: Any) -> None:
        del result
        torch = self.require_setup()
        if self._pending_gradient:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._pending_gradient = False

    def on_reject(self, error: BaseException) -> None:
        del error
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        self._pending_gradient = False

    def teardown(self) -> None:
        self.model = None
        self.optimizer = None
        self.input_batch = None
        self.training_target = None

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "drawing_with_traces.training_duty",
            "target_points": int(self.target_values.size),
            "target_sha256": hashlib_array(self.target_values),
            "duty_sha256": hashlib_array(self.duty_values),
            "profile_duration_s": self.profile_duration_s,
            "segment_duration_s": self.segment_duration_s,
            "quantum_s": self.quantum_s,
            "lead_s": self.lead_s,
            "tail_s": self.tail_s,
            "model": "Linear-GELU-Linear",
            "dtype": "bfloat16",
            "hidden_size": self.hidden_size,
            "feedforward_size": self.feedforward_size,
            "batch_size": self.batch_size,
            "parameter_count": self.parameter_count,
            "optimizer": "AdamW",
            "learning_rate": self.learning_rate,
            "gradient_scale": self.gradient_scale,
            "seed": self.seed,
            "transactional_update": True,
        }


def hashlib_array(values: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest()


class AdaptiveTrainingPowerWorkload(TrainingDutyWorkload):
    """Anticipatory feed-forward plus live NVML feedback for short profiles."""

    def __init__(
        self,
        target_values: np.ndarray,
        *,
        calibration_duty: np.ndarray,
        calibration_power_w: np.ndarray,
        profile_duration_s: float,
        control_s: float = 0.1,
        advance_s: float = 1.0,
        feedback_kp: float = 0.2,
        feedback_ki: float = 0.04,
        **kwargs,
    ):
        calibration_duty = np.asarray(calibration_duty, dtype=np.float32)
        calibration_power_w = np.asarray(calibration_power_w, dtype=np.float32)
        if calibration_duty.ndim != 1 or calibration_duty.size < 3:
            raise ValueError("calibration must contain at least three duty levels")
        if calibration_duty.shape != calibration_power_w.shape:
            raise ValueError("calibration duty and power arrays must have the same shape")
        if np.any(np.diff(calibration_duty) <= 0) or np.any(np.diff(calibration_power_w) < 0):
            raise ValueError("calibration duty must increase and calibrated power cannot decrease")
        if control_s <= 0 or control_s > profile_duration_s / 2:
            raise ValueError("control_s must be positive and substantially shorter than the profile")
        if advance_s < 0 or advance_s > kwargs.get("lead_s", 3.0):
            raise ValueError("advance_s must be non-negative and no longer than lead_s")
        if feedback_kp < 0 or feedback_ki < 0:
            raise ValueError("feedback gains cannot be negative")
        target_values = np.asarray(target_values, dtype=np.float32)
        low, high = float(calibration_power_w[0]), float(calibration_power_w[-1])
        target_power = low + target_values * (high - low)
        initial_duty = np.interp(target_power, calibration_power_w, calibration_duty).astype(np.float32)
        super().__init__(
            target_values,
            initial_duty,
            profile_duration_s=profile_duration_s,
            **kwargs,
        )
        self.calibration_duty = calibration_duty
        self.calibration_power_w = calibration_power_w
        self.control_s = float(control_s)
        self.advance_s = float(advance_s)
        self.feedback_kp = float(feedback_kp)
        self.feedback_ki = float(feedback_ki)
        self._nvml = None
        self._nvml_handle = None

    def setup(self) -> None:
        super().setup()
        try:
            import pynvml
        except ImportError as error:
            raise ConfigurationError("adaptive power control requires sidecapture[nvml]") from error
        self._nvml = pynvml
        self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    @property
    def power_span_w(self) -> float:
        return float(self.calibration_power_w[-1] - self.calibration_power_w[0])

    def target_at(self, profile_time_s: float) -> float:
        x = np.linspace(0, self.profile_duration_s, self.target_values.size)
        return float(np.interp(np.clip(profile_time_s, 0, self.profile_duration_s), x, self.target_values))

    def target_power_at(self, profile_time_s: float) -> float:
        normalized = self.target_at(profile_time_s)
        return float(self.calibration_power_w[0] + normalized * self.power_span_w)

    def read_power_w(self) -> float:
        try:
            return float(self._nvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0)
        except Exception:
            return float("nan")

    def run_pwm(self, duration_s: float, duty: float) -> tuple[int, int]:
        torch = self.require_setup()
        end_ns = time.monotonic_ns() + int(round(duration_s * 1e9))
        quantum_ns = int(round(self.quantum_s * 1e9))
        count = 0
        max_overrun_ns = 0
        while time.monotonic_ns() < end_ns:
            quantum_start = time.monotonic_ns()
            quantum_end = min(end_ns, quantum_start + quantum_ns)
            active_end = quantum_start + int(round((quantum_end - quantum_start) * duty))
            while time.monotonic_ns() < active_end:
                self.microstep()
                torch.cuda.synchronize()
                count += 1
            sleep_until(quantum_end)
            max_overrun_ns = max(max_overrun_ns, time.monotonic_ns() - quantum_end)
        return count, max_overrun_ns

    def run(self, context):
        self.optimizer.zero_grad(set_to_none=True)
        self._pending_gradient = False
        context.labels.update(
            task="adaptive-silhouette-shaped-model-training",
            profile_points=int(self.target_values.size),
            profile_duration_s=self.profile_duration_s,
            model_parameters=self.parameter_count,
            response_advance_s=self.advance_s,
        )
        context.add_artifact("target_values", self.target_values)
        context.add_artifact("feedforward_duty_values", self.duty_values)

        with context.region("power_target.lead_idle", duration_s=self.lead_s):
            sleep_until(time.monotonic_ns() + int(round(self.lead_s * 1e9)))

        integral = 0.0
        all_duty = []
        all_feedback = []
        all_target_power = []
        all_microsteps = []
        max_overrun_ns = 0

        def control_interval(profile_time_s: float, duration_s: float) -> None:
            nonlocal integral, max_overrun_ns
            current_target_w = self.target_power_at(profile_time_s)
            future_target_w = self.target_power_at(profile_time_s + self.advance_s)
            feedforward = float(
                np.interp(future_target_w, self.calibration_power_w, self.calibration_duty)
            )
            feedback_w = self.read_power_w()
            error = 0.0 if not np.isfinite(feedback_w) else (current_target_w - feedback_w) / self.power_span_w
            integral = float(np.clip(integral + error * duration_s, -1.0, 1.0))
            duty = float(
                np.clip(feedforward + self.feedback_kp * error + self.feedback_ki * integral, 0, 1)
            )
            count, overrun_ns = self.run_pwm(duration_s, duty)
            max_overrun_ns = max(max_overrun_ns, overrun_ns)
            all_duty.append(duty)
            all_feedback.append(feedback_w)
            all_target_power.append(current_target_w)
            all_microsteps.append(count)

        with context.region(
            "power_target.profile",
            points=int(self.target_values.size),
            duration_s=self.profile_duration_s,
            adaptive=True,
        ):
            elapsed = 0.0
            index = 0
            while elapsed < self.profile_duration_s:
                duration = min(self.control_s, self.profile_duration_s - elapsed)
                with context.region(
                    "power_target.control",
                    index=index,
                    profile_time_s=elapsed,
                    target_normalized=self.target_at(elapsed),
                ):
                    control_interval(elapsed, duration)
                elapsed += duration
                index += 1

        with context.region("power_target.tail", duration_s=self.tail_s):
            sleep_until(time.monotonic_ns() + int(round(self.tail_s * 1e9)))
        if max_overrun_ns > max(5_000_000, int(self.quantum_s * 1e9) // 2):
            raise RuntimeError(
                f"adaptive scheduler overran by {max_overrun_ns / 1e6:.3f} ms; discarding capture"
            )
        loss = None if self._latest_loss is None else float(self._latest_loss)
        if loss is not None and not math.isfinite(loss):
            raise RuntimeError(f"training loss became non-finite: {loss}")
        context.add_artifact("actual_duty", np.asarray(all_duty, dtype=np.float32))
        context.add_artifact("feedback_power_w", np.asarray(all_feedback, dtype=np.float32))
        context.add_artifact(
            "controller_target_power_w",
            np.asarray(all_target_power, dtype=np.float32),
        )
        context.add_artifact(
            "microsteps_per_control", np.asarray(all_microsteps, dtype=np.int64)
        )
        context.labels.update(
            training_microsteps=int(np.sum(all_microsteps)),
            training_loss_before_update=loss,
            max_quantum_overrun_us=max_overrun_ns / 1e3,
            mean_actual_duty=float(np.mean(all_duty)),
        )
        return {
            "training_microsteps": int(np.sum(all_microsteps)),
            "loss_before_update": loss,
            "max_quantum_overrun_us": max_overrun_ns / 1e3,
        }

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        metadata.update(
            {
                "controller": "causal_lookahead_feedforward_plus_nvml_pi",
                "control_s": self.control_s,
                "advance_s": self.advance_s,
                "feedback_kp": self.feedback_kp,
                "feedback_ki": self.feedback_ki,
                "calibration_duty": self.calibration_duty.tolist(),
                "calibration_power_w": self.calibration_power_w.tolist(),
            }
        )
        return metadata
