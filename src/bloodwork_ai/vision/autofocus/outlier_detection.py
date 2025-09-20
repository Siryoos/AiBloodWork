from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
from enum import Enum
import time


class OutlierType(Enum):
    """Types of autofocus outliers."""
    FOCUS_RANGE_EXCEEDED = "focus_range_exceeded"
    SURFACE_PREDICTION_ERROR = "surface_prediction_error"
    METRIC_SNR_LOW = "metric_snr_low"
    DOUBLE_PEAK = "double_peak"
    FLAT_RESPONSE = "flat_response"
    HARDWARE_ERROR = "hardware_error"
    TIMEOUT = "timeout"
    EXCESSIVE_DRIFT = "excessive_drift"


class FailureMode(Enum):
    """Autofocus failure modes."""
    NO_CONVERGENCE = "no_convergence"
    HARDWARE_FAULT = "hardware_fault"
    ILLUMINATION_FAILURE = "illumination_failure"
    STAGE_ERROR = "stage_error"
    CAMERA_ERROR = "camera_error"
    SOFTWARE_ERROR = "software_error"


@dataclass
class OutlierDetectionConfig:
    """Configuration for outlier detection."""

    # Focus range limits (um)
    max_focus_deviation_um: float = 8.0
    absolute_focus_limits_um: Tuple[float, float] = (-100.0, 100.0)

    # Surface prediction tolerances
    max_surface_residual_um: float = 5.0
    surface_prediction_confidence_threshold: float = 0.8

    # Metric quality thresholds
    min_metric_snr: float = 3.0
    min_metric_value: float = 1000.0  # Absolute minimum for meaningful focus

    # Curve analysis
    min_curve_peak_ratio: float = 1.5  # Peak must be 1.5x higher than neighbors
    max_flat_response_ratio: float = 0.1  # Max variation for "flat" response

    # Timing limits
    max_autofocus_time_ms: float = 500.0
    max_single_measurement_ms: float = 100.0

    # Drift detection
    max_drift_rate_um_per_min: float = 2.0
    drift_detection_window: int = 10  # Number of measurements for trend analysis


@dataclass
class OutlierResult:
    """Result of outlier detection analysis."""

    is_outlier: bool
    outlier_types: List[OutlierType] = field(default_factory=list)
    confidence: float = 1.0  # 0-1, confidence in the result
    metrics: Dict[str, float] = field(default_factory=dict)
    recommended_action: str = "proceed"  # proceed, retry, fallback, abort
    details: str = ""


@dataclass
class FailureHandlingConfig:
    """Configuration for failure handling strategies."""

    # Retry settings
    max_retries: int = 3
    retry_delay_ms: float = 100.0
    exponential_backoff: bool = True

    # Fallback strategies
    enable_coarse_fallback: bool = True
    enable_surface_fallback: bool = True
    enable_thermal_fallback: bool = True

    # Recovery timeouts
    hardware_recovery_timeout_ms: float = 5000.0
    camera_recovery_timeout_ms: float = 2000.0


class OutlierDetector:
    """Production-grade outlier detection for autofocus results."""

    def __init__(self, config: OutlierDetectionConfig):
        self.config = config
        self._focus_history: List[Tuple[float, float, float]] = []  # (time, x_pos, z_focus)
        self._metric_history: List[Tuple[float, float]] = []  # (time, metric_value)

    def analyze_focus_result(self,
                           z_result: float,
                           metric_value: float,
                           surface_prediction: Optional[float] = None,
                           focus_curve: Optional[List[Tuple[float, float]]] = None,
                           elapsed_ms: Optional[float] = None,
                           **kwargs) -> OutlierResult:
        """Analyze autofocus result for outliers.

        Args:
            z_result: Focus position result
            metric_value: Final metric value
            surface_prediction: Predicted focus from surface model
            focus_curve: List of (z, metric) points from focus sweep
            elapsed_ms: Time taken for autofocus
            **kwargs: Additional context (temperature, illumination, etc.)

        Returns:
            OutlierResult with analysis
        """
        outlier_types = []
        metrics = {}
        confidence = 1.0

        # Check focus range limits
        if self._check_focus_range_violation(z_result):
            outlier_types.append(OutlierType.FOCUS_RANGE_EXCEEDED)
            confidence *= 0.8

        # Check surface prediction agreement
        if surface_prediction is not None:
            surface_error = abs(z_result - surface_prediction)
            metrics["surface_residual_um"] = surface_error
            if surface_error > self.config.max_surface_residual_um:
                outlier_types.append(OutlierType.SURFACE_PREDICTION_ERROR)
                confidence *= 0.7

        # Check metric quality
        metric_analysis = self._analyze_metric_quality(metric_value, focus_curve)
        metrics.update(metric_analysis["metrics"])
        if metric_analysis["outlier_types"]:
            outlier_types.extend(metric_analysis["outlier_types"])
            confidence *= metric_analysis["confidence_factor"]

        # Check timing
        if elapsed_ms is not None and elapsed_ms > self.config.max_autofocus_time_ms:
            outlier_types.append(OutlierType.TIMEOUT)
            metrics["elapsed_ms"] = elapsed_ms

        # Check for drift patterns
        drift_analysis = self._analyze_drift_pattern(z_result)
        if drift_analysis["excessive_drift"]:
            outlier_types.append(OutlierType.EXCESSIVE_DRIFT)
            metrics.update(drift_analysis["metrics"])

        # Determine recommended action
        recommended_action = self._determine_action(outlier_types, confidence)

        return OutlierResult(
            is_outlier=len(outlier_types) > 0,
            outlier_types=outlier_types,
            confidence=confidence,
            metrics=metrics,
            recommended_action=recommended_action,
            details=self._generate_details(outlier_types, metrics)
        )

    def _check_focus_range_violation(self, z_result: float) -> bool:
        """Check if focus result violates range limits."""
        min_limit, max_limit = self.config.absolute_focus_limits_um
        return z_result < min_limit or z_result > max_limit

    def _analyze_metric_quality(self,
                               metric_value: float,
                               focus_curve: Optional[List[Tuple[float, float]]]) -> Dict[str, Any]:
        """Analyze focus metric quality."""
        outlier_types = []
        metrics = {"final_metric_value": metric_value}
        confidence_factor = 1.0

        # Check absolute metric value
        if metric_value < self.config.min_metric_value:
            outlier_types.append(OutlierType.METRIC_SNR_LOW)
            confidence_factor *= 0.6

        if focus_curve is None or len(focus_curve) < 3:
            return {
                "outlier_types": outlier_types,
                "metrics": metrics,
                "confidence_factor": confidence_factor
            }

        # Analyze focus curve
        z_values = np.array([point[0] for point in focus_curve])
        metric_values = np.array([point[1] for point in focus_curve])

        # Find peak
        peak_idx = np.argmax(metric_values)
        peak_value = metric_values[peak_idx]
        peak_z = z_values[peak_idx]

        # Check for flat response
        metric_range = np.ptp(metric_values)
        mean_metric = np.mean(metric_values)
        relative_range = metric_range / mean_metric if mean_metric > 0 else 0

        if relative_range < self.config.max_flat_response_ratio:
            outlier_types.append(OutlierType.FLAT_RESPONSE)
            confidence_factor *= 0.5

        # Check peak prominence
        if peak_idx > 0 and peak_idx < len(metric_values) - 1:
            left_neighbor = metric_values[peak_idx - 1]
            right_neighbor = metric_values[peak_idx + 1]
            avg_neighbor = (left_neighbor + right_neighbor) / 2

            peak_ratio = peak_value / avg_neighbor if avg_neighbor > 0 else float('inf')
            metrics["peak_ratio"] = peak_ratio

            if peak_ratio < self.config.min_curve_peak_ratio:
                outlier_types.append(OutlierType.METRIC_SNR_LOW)
                confidence_factor *= 0.7

        # Check for double peak (local maxima)
        local_maxima = self._find_local_maxima(metric_values)
        if len(local_maxima) > 1:
            # Check if there are multiple significant peaks
            sorted_peaks = sorted([metric_values[i] for i in local_maxima], reverse=True)
            if len(sorted_peaks) >= 2 and sorted_peaks[1] / sorted_peaks[0] > 0.8:
                outlier_types.append(OutlierType.DOUBLE_PEAK)
                metrics["secondary_peak_ratio"] = sorted_peaks[1] / sorted_peaks[0]

        metrics.update({
            "curve_range": metric_range,
            "curve_snr": peak_value / np.std(metric_values) if np.std(metric_values) > 0 else 0,
            "num_local_maxima": len(local_maxima)
        })

        return {
            "outlier_types": outlier_types,
            "metrics": metrics,
            "confidence_factor": confidence_factor
        }

    def _find_local_maxima(self, values: np.ndarray, min_prominence: float = 0.1) -> List[int]:
        """Find local maxima in metric curve."""
        maxima = []
        for i in range(1, len(values) - 1):
            if values[i] > values[i-1] and values[i] > values[i+1]:
                # Check prominence
                prominence = min(values[i] - values[i-1], values[i] - values[i+1])
                relative_prominence = prominence / values[i] if values[i] > 0 else 0
                if relative_prominence >= min_prominence:
                    maxima.append(i)
        return maxima

    def _analyze_drift_pattern(self, z_result: float) -> Dict[str, Any]:
        """Analyze focus drift patterns."""
        current_time = time.time()

        # Add to history
        self._focus_history.append((current_time, 0.0, z_result))  # x_pos=0 for now

        # Limit history size
        max_history = self.config.drift_detection_window * 2
        if len(self._focus_history) > max_history:
            self._focus_history = self._focus_history[-max_history:]

        # Need at least 3 points for trend analysis
        if len(self._focus_history) < 3:
            return {"excessive_drift": False, "metrics": {}}

        # Get recent measurements within time window
        window_s = 60.0  # 1 minute window
        cutoff_time = current_time - window_s
        recent_history = [(t, x, z) for t, x, z in self._focus_history if t >= cutoff_time]

        if len(recent_history) < 3:
            return {"excessive_drift": False, "metrics": {}}

        # Calculate drift rate
        times = np.array([t - recent_history[0][0] for t, x, z in recent_history])  # Relative times
        z_values = np.array([z for t, x, z in recent_history])

        # Linear fit to estimate drift rate
        if len(times) >= 2 and np.std(times) > 0:
            drift_rate = np.polyfit(times, z_values, 1)[0]  # um/s
            drift_rate_per_min = drift_rate * 60.0  # um/min

            excessive_drift = abs(drift_rate_per_min) > self.config.max_drift_rate_um_per_min

            return {
                "excessive_drift": excessive_drift,
                "metrics": {
                    "drift_rate_um_per_min": drift_rate_per_min,
                    "focus_std_um": float(np.std(z_values)),
                    "measurement_count": len(recent_history)
                }
            }

        return {"excessive_drift": False, "metrics": {}}

    def _determine_action(self, outlier_types: List[OutlierType], confidence: float) -> str:
        """Determine recommended action based on outlier analysis."""
        if not outlier_types:
            return "proceed"

        # Critical outliers that require immediate action
        critical_outliers = {
            OutlierType.FOCUS_RANGE_EXCEEDED,
            OutlierType.HARDWARE_ERROR,
            OutlierType.TIMEOUT
        }

        if any(ot in critical_outliers for ot in outlier_types):
            return "abort"

        # Outliers that suggest retry might help
        retry_outliers = {
            OutlierType.METRIC_SNR_LOW,
            OutlierType.FLAT_RESPONSE
        }

        if any(ot in retry_outliers for ot in outlier_types) and confidence > 0.5:
            return "retry"

        # Outliers that suggest fallback strategy
        fallback_outliers = {
            OutlierType.SURFACE_PREDICTION_ERROR,
            OutlierType.DOUBLE_PEAK,
            OutlierType.EXCESSIVE_DRIFT
        }

        if any(ot in fallback_outliers for ot in outlier_types):
            return "fallback"

        return "proceed"

    def _generate_details(self, outlier_types: List[OutlierType], metrics: Dict[str, float]) -> str:
        """Generate human-readable details about outliers."""
        if not outlier_types:
            return "No outliers detected"

        details = []
        for outlier_type in outlier_types:
            if outlier_type == OutlierType.FOCUS_RANGE_EXCEEDED:
                details.append("Focus position outside acceptable range")
            elif outlier_type == OutlierType.SURFACE_PREDICTION_ERROR:
                residual = metrics.get("surface_residual_um", 0)
                details.append(f"Surface prediction error: {residual:.1f} um")
            elif outlier_type == OutlierType.METRIC_SNR_LOW:
                snr = metrics.get("curve_snr", 0)
                details.append(f"Low metric SNR: {snr:.1f}")
            elif outlier_type == OutlierType.DOUBLE_PEAK:
                ratio = metrics.get("secondary_peak_ratio", 0)
                details.append(f"Multiple peaks detected, secondary ratio: {ratio:.2f}")
            elif outlier_type == OutlierType.FLAT_RESPONSE:
                details.append("Focus curve shows flat response")
            elif outlier_type == OutlierType.EXCESSIVE_DRIFT:
                drift = metrics.get("drift_rate_um_per_min", 0)
                details.append(f"Excessive drift: {drift:.1f} um/min")

        return "; ".join(details)


class FailureHandler:
    """Handles autofocus failures and implements recovery strategies."""

    def __init__(self, config: FailureHandlingConfig):
        self.config = config
        self.failure_counts: Dict[str, int] = {}
        self.last_failure_times: Dict[str, float] = {}

    def handle_failure(self,
                      failure_mode: FailureMode,
                      context: Dict[str, Any],
                      autofocus_system) -> Dict[str, Any]:
        """Handle autofocus failure with appropriate recovery strategy.

        Args:
            failure_mode: Type of failure encountered
            context: Context information (tile_id, position, etc.)
            autofocus_system: Reference to autofocus system

        Returns:
            Recovery result with status and actions taken
        """
        failure_key = f"{failure_mode.value}_{context.get('tile_id', 'unknown')}"

        # Track failure frequency
        self.failure_counts[failure_key] = self.failure_counts.get(failure_key, 0) + 1
        self.last_failure_times[failure_key] = time.time()

        # Determine recovery strategy
        recovery_strategy = self._select_recovery_strategy(failure_mode, context)

        # Execute recovery
        return self._execute_recovery(recovery_strategy, failure_mode, context, autofocus_system)

    def _select_recovery_strategy(self,
                                failure_mode: FailureMode,
                                context: Dict[str, Any]) -> str:
        """Select appropriate recovery strategy."""
        failure_count = self.failure_counts.get(
            f"{failure_mode.value}_{context.get('tile_id', 'unknown')}", 0
        )

        if failure_count >= self.config.max_retries:
            return "abort"

        if failure_mode == FailureMode.NO_CONVERGENCE:
            if self.config.enable_coarse_fallback:
                return "coarse_fallback"
            else:
                return "retry_with_expansion"

        elif failure_mode == FailureMode.HARDWARE_FAULT:
            return "hardware_recovery"

        elif failure_mode == FailureMode.ILLUMINATION_FAILURE:
            return "illumination_recovery"

        elif failure_mode == FailureMode.STAGE_ERROR:
            return "stage_recovery"

        elif failure_mode == FailureMode.CAMERA_ERROR:
            return "camera_recovery"

        else:
            return "generic_retry"

    def _execute_recovery(self,
                         strategy: str,
                         failure_mode: FailureMode,
                         context: Dict[str, Any],
                         autofocus_system) -> Dict[str, Any]:
        """Execute recovery strategy."""
        recovery_start = time.time()

        try:
            if strategy == "abort":
                return {
                    "status": "failed",
                    "action": "abort",
                    "reason": "Maximum retry limit exceeded",
                    "elapsed_ms": 0
                }

            elif strategy == "coarse_fallback":
                return self._coarse_fallback_recovery(context, autofocus_system)

            elif strategy == "hardware_recovery":
                return self._hardware_recovery(autofocus_system)

            elif strategy == "illumination_recovery":
                return self._illumination_recovery(autofocus_system)

            elif strategy == "camera_recovery":
                return self._camera_recovery(autofocus_system)

            elif strategy == "retry_with_expansion":
                return self._retry_with_expansion(context, autofocus_system)

            else:
                # Generic retry with delay
                time.sleep(self.config.retry_delay_ms / 1000.0)
                return {
                    "status": "retry",
                    "action": "generic_retry",
                    "elapsed_ms": (time.time() - recovery_start) * 1000
                }

        except Exception as e:
            return {
                "status": "failed",
                "action": strategy,
                "error": str(e),
                "elapsed_ms": (time.time() - recovery_start) * 1000
            }

    def _coarse_fallback_recovery(self, context: Dict[str, Any], autofocus_system) -> Dict[str, Any]:
        """Fallback to coarse-only autofocus."""
        try:
            # Temporarily modify config for coarse-only search
            original_fine_step = autofocus_system.config.search.fine_step_um
            autofocus_system.config.search.fine_step_um = autofocus_system.config.search.coarse_step_um

            # Run autofocus with coarse steps only
            result = autofocus_system.autofocus_at_position(
                x=context.get('x_um'),
                y=context.get('y_um'),
                z_guess=context.get('z_guess_um')
            )

            # Restore original config
            autofocus_system.config.search.fine_step_um = original_fine_step

            return {
                "status": "recovered",
                "action": "coarse_fallback",
                "result_z_um": result,
                "elapsed_ms": 0  # Not tracking detailed timing for fallback
            }

        except Exception as e:
            return {
                "status": "failed",
                "action": "coarse_fallback",
                "error": str(e)
            }

    def _hardware_recovery(self, autofocus_system) -> Dict[str, Any]:
        """Attempt hardware recovery."""
        # Basic hardware re-initialization
        try:
            # Home the Z-axis if possible
            if hasattr(autofocus_system.camera, 'home_focus'):
                autofocus_system.camera.home_focus()

            time.sleep(0.1)  # Allow hardware to settle

            return {
                "status": "recovered",
                "action": "hardware_recovery"
            }
        except Exception as e:
            return {
                "status": "failed",
                "action": "hardware_recovery",
                "error": str(e)
            }

    def _illumination_recovery(self, autofocus_system) -> Dict[str, Any]:
        """Attempt illumination recovery."""
        try:
            if autofocus_system._illumination_manager:
                # Reset to default illumination
                autofocus_system._illumination_manager.set_uniform(0.5)

            return {
                "status": "recovered",
                "action": "illumination_recovery"
            }
        except Exception as e:
            return {
                "status": "failed",
                "action": "illumination_recovery",
                "error": str(e)
            }

    def _camera_recovery(self, autofocus_system) -> Dict[str, Any]:
        """Attempt camera recovery."""
        try:
            # Try to acquire a test frame
            frame = autofocus_system.camera.get_frame()
            if frame is not None and frame.size > 0:
                return {
                    "status": "recovered",
                    "action": "camera_recovery"
                }
            else:
                return {
                    "status": "failed",
                    "action": "camera_recovery",
                    "error": "Camera not responding"
                }
        except Exception as e:
            return {
                "status": "failed",
                "action": "camera_recovery",
                "error": str(e)
            }

    def _retry_with_expansion(self, context: Dict[str, Any], autofocus_system) -> Dict[str, Any]:
        """Retry with expanded search range."""
        try:
            # Temporarily expand search bracket
            original_bracket = autofocus_system.config.search.bracket_um
            autofocus_system.config.search.bracket_um *= 2

            result = autofocus_system.autofocus_at_position(
                x=context.get('x_um'),
                y=context.get('y_um'),
                z_guess=context.get('z_guess_um')
            )

            # Restore original bracket
            autofocus_system.config.search.bracket_um = original_bracket

            return {
                "status": "recovered",
                "action": "retry_with_expansion",
                "result_z_um": result
            }

        except Exception as e:
            return {
                "status": "failed",
                "action": "retry_with_expansion",
                "error": str(e)
            }