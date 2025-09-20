from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json
from pathlib import Path

from .dual_lens import LensID, ParfocalMapping


class MappingModel(Enum):
    """Parfocal mapping model types."""
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    CUBIC = "cubic"
    ADAPTIVE = "adaptive"


@dataclass
class EnhancedParfocalMapping:
    """Enhanced parfocal mapping with adaptive accuracy and learning."""

    # Core mapping parameters
    model_type: MappingModel = MappingModel.ADAPTIVE
    coefficients: List[float] = field(default_factory=lambda: [0.0, 1.0, 0.0, 0.0])  # [offset, linear, quadratic, cubic]

    # Calibration metadata
    calibration_timestamp: float = 0.0
    calibration_temperature_c: float = 23.0
    num_calibration_points: int = 0
    rms_error_um: float = 0.0
    max_error_um: float = 0.0

    # Temperature compensation
    temp_coeff_linear_um_per_c: float = 0.0
    temp_coeff_offset_um_per_c: float = 0.0

    # Adaptive learning
    validation_points: List[Tuple[float, float, float]] = field(default_factory=list)  # (z_a, z_b, timestamp)
    learning_rate: float = 0.1
    confidence_threshold: float = 0.95

    # Performance tracking
    mapping_history: List[Dict[str, Any]] = field(default_factory=list)
    accuracy_trend: List[float] = field(default_factory=list)

    def map_lens_a_to_b(self, z_a_um: float, temperature_c: float = 23.0) -> float:
        """Enhanced A→B mapping with adaptive accuracy."""
        # Apply temperature compensation
        temp_delta = temperature_c - self.calibration_temperature_c
        temp_offset = self.temp_coeff_offset_um_per_c * temp_delta
        temp_linear = self.temp_coeff_linear_um_per_c * temp_delta

        # Apply mapping model
        if self.model_type == MappingModel.LINEAR:
            z_b_um = (self.coefficients[0] + temp_offset +
                     (self.coefficients[1] + temp_linear) * z_a_um)

        elif self.model_type == MappingModel.QUADRATIC:
            z_b_um = (self.coefficients[0] + temp_offset +
                     (self.coefficients[1] + temp_linear) * z_a_um +
                     self.coefficients[2] * z_a_um**2)

        elif self.model_type == MappingModel.CUBIC:
            z_b_um = (self.coefficients[0] + temp_offset +
                     (self.coefficients[1] + temp_linear) * z_a_um +
                     self.coefficients[2] * z_a_um**2 +
                     self.coefficients[3] * z_a_um**3)

        elif self.model_type == MappingModel.ADAPTIVE:
            # Use best fit model based on current accuracy
            z_b_um = self._adaptive_mapping(z_a_um, temperature_c)

        else:
            z_b_um = z_a_um  # Fallback

        # Record mapping for learning
        self._record_mapping(z_a_um, z_b_um, temperature_c)

        return z_b_um

    def map_lens_b_to_a(self, z_b_um: float, temperature_c: float = 23.0) -> float:
        """Enhanced B→A mapping with iterative solution."""
        # For non-linear mappings, use iterative solver
        if self.model_type in [MappingModel.QUADRATIC, MappingModel.CUBIC, MappingModel.ADAPTIVE]:
            return self._inverse_mapping_iterative(z_b_um, temperature_c)
        else:
            # Linear case - direct solution
            temp_delta = temperature_c - self.calibration_temperature_c
            temp_offset = self.temp_coeff_offset_um_per_c * temp_delta
            temp_linear = self.temp_coeff_linear_um_per_c * temp_delta

            linear_coeff = self.coefficients[1] + temp_linear
            if abs(linear_coeff) > 1e-6:
                z_a_um = (z_b_um - self.coefficients[0] - temp_offset) / linear_coeff
            else:
                z_a_um = 0.0  # Degenerate case

        return z_a_um

    def _adaptive_mapping(self, z_a_um: float, temperature_c: float) -> float:
        """Adaptive mapping that selects best model based on local accuracy."""
        # Try different models and select based on confidence
        models_to_try = [MappingModel.LINEAR, MappingModel.QUADRATIC, MappingModel.CUBIC]

        best_result = None
        best_confidence = 0.0

        for model in models_to_try:
            temp_model_type = self.model_type
            self.model_type = model

            try:
                result = self.map_lens_a_to_b(z_a_um, temperature_c)
                confidence = self._estimate_local_confidence(z_a_um, result)

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_result = result

            finally:
                self.model_type = temp_model_type

        # Use quadratic as default if no good confidence
        if best_result is None or best_confidence < self.confidence_threshold:
            temp_delta = temperature_c - self.calibration_temperature_c
            temp_offset = self.temp_coeff_offset_um_per_c * temp_delta
            temp_linear = self.temp_coeff_linear_um_per_c * temp_delta

            best_result = (self.coefficients[0] + temp_offset +
                          (self.coefficients[1] + temp_linear) * z_a_um +
                          self.coefficients[2] * z_a_um**2)

        return best_result

    def _inverse_mapping_iterative(self, z_b_um: float, temperature_c: float) -> float:
        """Iterative solver for inverse mapping."""
        # Newton's method for finding z_a such that f(z_a) = z_b
        z_a_estimate = z_b_um / self.coefficients[1] if abs(self.coefficients[1]) > 1e-6 else 0.0

        for iteration in range(10):  # Max 10 iterations
            # Evaluate function and derivative
            f_val = self.map_lens_a_to_b(z_a_estimate, temperature_c) - z_b_um

            # Compute numerical derivative
            h = 0.01
            f_plus = self.map_lens_a_to_b(z_a_estimate + h, temperature_c)
            f_minus = self.map_lens_a_to_b(z_a_estimate - h, temperature_c)
            df_dz = (f_plus - f_minus) / (2 * h)

            # Newton update
            if abs(df_dz) > 1e-6:
                z_a_new = z_a_estimate - f_val / df_dz
            else:
                break  # Derivative too small

            # Check convergence
            if abs(z_a_new - z_a_estimate) < 0.001:  # 1nm convergence
                break

            z_a_estimate = z_a_new

        return z_a_estimate

    def _estimate_local_confidence(self, z_a_um: float, z_b_predicted: float) -> float:
        """Estimate confidence of mapping prediction based on nearby validation data."""
        if not self.validation_points:
            return 0.5  # Default moderate confidence

        # Find nearby validation points
        nearby_points = []
        search_radius = 2.0  # μm

        for z_a_val, z_b_val, timestamp in self.validation_points:
            if abs(z_a_val - z_a_um) <= search_radius:
                nearby_points.append((z_a_val, z_b_val, timestamp))

        if not nearby_points:
            return 0.5

        # Calculate local accuracy
        errors = []
        for z_a_val, z_b_actual, timestamp in nearby_points:
            z_b_pred = self.map_lens_a_to_b(z_a_val)
            error = abs(z_b_pred - z_b_actual)
            errors.append(error)

        # Convert error to confidence (lower error = higher confidence)
        mean_error = np.mean(errors)
        confidence = np.exp(-mean_error / 0.5)  # 0.5μm characteristic error scale

        return min(confidence, 1.0)

    def _record_mapping(self, z_a_um: float, z_b_um: float, temperature_c: float) -> None:
        """Record mapping for performance tracking."""
        record = {
            "timestamp": time.time(),
            "z_a_um": z_a_um,
            "z_b_um": z_b_um,
            "temperature_c": temperature_c,
            "model_type": self.model_type.value
        }

        self.mapping_history.append(record)

        # Limit history size
        if len(self.mapping_history) > 1000:
            self.mapping_history = self.mapping_history[-500:]

    def add_validation_point(self, z_a_um: float, z_b_actual: float) -> None:
        """Add validation point for adaptive learning."""
        timestamp = time.time()
        self.validation_points.append((z_a_um, z_b_actual, timestamp))

        # Limit validation points
        if len(self.validation_points) > 200:
            self.validation_points = self.validation_points[-100:]

        # Update accuracy trend
        z_b_predicted = self.map_lens_a_to_b(z_a_um)
        error = abs(z_b_predicted - z_b_actual)
        self.accuracy_trend.append(error)

        if len(self.accuracy_trend) > 50:
            self.accuracy_trend = self.accuracy_trend[-25:]

    def calibrate_enhanced(self, calibration_data: List[Tuple[float, float, float]]) -> Dict[str, Any]:
        """Enhanced calibration with model selection and validation."""
        if len(calibration_data) < 6:
            raise ValueError("Need at least 6 calibration points")

        # Extract data
        points = np.array(calibration_data)
        z_a_values = points[:, 0]
        z_b_values = points[:, 1]
        temperatures = points[:, 2]

        # Try different models and select best
        models = [MappingModel.LINEAR, MappingModel.QUADRATIC, MappingModel.CUBIC]
        best_model = None
        best_rms = float('inf')
        best_coeffs = None

        for model in models:
            try:
                coeffs, rms_error = self._fit_model(z_a_values, z_b_values, model)
                if rms_error < best_rms:
                    best_rms = rms_error
                    best_model = model
                    best_coeffs = coeffs
            except Exception:
                continue

        if best_model is None:
            raise ValueError("Calibration failed for all models")

        # Update mapping parameters
        self.model_type = best_model
        self.coefficients = best_coeffs
        self.rms_error_um = best_rms
        self.calibration_timestamp = time.time()
        self.calibration_temperature_c = np.mean(temperatures)
        self.num_calibration_points = len(calibration_data)

        # Estimate temperature compensation if temperature variation exists
        if np.std(temperatures) > 1.0:  # At least 1°C variation
            self._estimate_temperature_compensation(calibration_data)

        # Calculate maximum error
        errors = []
        for z_a, z_b_actual, temp in calibration_data:
            z_b_pred = self.map_lens_a_to_b(z_a, temp)
            errors.append(abs(z_b_pred - z_b_actual))

        self.max_error_um = np.max(errors)

        return {
            "model_type": best_model.value,
            "rms_error_um": best_rms,
            "max_error_um": self.max_error_um,
            "num_points": len(calibration_data),
            "coefficients": best_coeffs,
            "temperature_compensation": {
                "offset_um_per_c": self.temp_coeff_offset_um_per_c,
                "linear_um_per_c": self.temp_coeff_linear_um_per_c
            }
        }

    def _fit_model(self, z_a_values: np.ndarray, z_b_values: np.ndarray, model: MappingModel) -> Tuple[List[float], float]:
        """Fit specific mapping model."""
        if model == MappingModel.LINEAR:
            # z_b = a + b*z_a
            design_matrix = np.column_stack([np.ones(len(z_a_values)), z_a_values])
            coeffs = [0.0, 0.0, 0.0, 0.0]

        elif model == MappingModel.QUADRATIC:
            # z_b = a + b*z_a + c*z_a^2
            design_matrix = np.column_stack([
                np.ones(len(z_a_values)),
                z_a_values,
                z_a_values**2
            ])
            coeffs = [0.0, 0.0, 0.0, 0.0]

        elif model == MappingModel.CUBIC:
            # z_b = a + b*z_a + c*z_a^2 + d*z_a^3
            design_matrix = np.column_stack([
                np.ones(len(z_a_values)),
                z_a_values,
                z_a_values**2,
                z_a_values**3
            ])
            coeffs = [0.0, 0.0, 0.0, 0.0]

        else:
            raise ValueError(f"Unsupported model: {model}")

        # Solve least squares
        fitted_coeffs, residuals, rank, singular_values = np.linalg.lstsq(
            design_matrix, z_b_values, rcond=None
        )

        # Update coefficient array
        for i, coeff in enumerate(fitted_coeffs):
            if i < len(coeffs):
                coeffs[i] = float(coeff)

        # Calculate RMS error
        predicted = design_matrix @ fitted_coeffs
        rms_error = np.sqrt(np.mean((z_b_values - predicted)**2))

        return coeffs, float(rms_error)

    def _estimate_temperature_compensation(self, calibration_data: List[Tuple[float, float, float]]) -> None:
        """Estimate temperature compensation coefficients."""
        # Group data by temperature
        temp_groups = {}
        for z_a, z_b, temp in calibration_data:
            temp_rounded = round(temp)
            if temp_rounded not in temp_groups:
                temp_groups[temp_rounded] = []
            temp_groups[temp_rounded].append((z_a, z_b))

        if len(temp_groups) < 2:
            return  # Need at least 2 temperatures

        # For each temperature, fit basic linear model
        temp_coeffs = {}
        for temp, points in temp_groups.items():
            if len(points) >= 3:
                z_a_vals = np.array([p[0] for p in points])
                z_b_vals = np.array([p[1] for p in points])

                # Fit z_b = offset + linear*z_a
                design = np.column_stack([np.ones(len(z_a_vals)), z_a_vals])
                coeffs, _, _, _ = np.linalg.lstsq(design, z_b_vals, rcond=None)
                temp_coeffs[temp] = {"offset": coeffs[0], "linear": coeffs[1]}

        # Estimate temperature dependence
        if len(temp_coeffs) >= 2:
            temps = np.array(list(temp_coeffs.keys()))
            offsets = np.array([temp_coeffs[t]["offset"] for t in temps])
            linears = np.array([temp_coeffs[t]["linear"] for t in temps])

            # Linear fit of coefficients vs temperature
            self.temp_coeff_offset_um_per_c = np.polyfit(temps, offsets, 1)[0]
            self.temp_coeff_linear_um_per_c = np.polyfit(temps, linears, 1)[0]

    def get_mapping_accuracy_report(self) -> Dict[str, Any]:
        """Get comprehensive accuracy report."""
        recent_accuracy = self.accuracy_trend[-10:] if self.accuracy_trend else []

        report = {
            "calibration": {
                "model_type": self.model_type.value,
                "rms_error_um": self.rms_error_um,
                "max_error_um": self.max_error_um,
                "num_calibration_points": self.num_calibration_points,
                "calibration_age_hours": (time.time() - self.calibration_timestamp) / 3600
            },
            "recent_performance": {
                "num_validation_points": len(self.validation_points),
                "recent_avg_error_um": np.mean(recent_accuracy) if recent_accuracy else None,
                "recent_max_error_um": np.max(recent_accuracy) if recent_accuracy else None,
                "accuracy_trend": "improving" if self._is_accuracy_improving() else "stable"
            },
            "temperature_compensation": {
                "offset_coefficient_um_per_c": self.temp_coeff_offset_um_per_c,
                "linear_coefficient_um_per_c": self.temp_coeff_linear_um_per_c,
                "thermal_stability_um_per_c": abs(self.temp_coeff_offset_um_per_c) + abs(self.temp_coeff_linear_um_per_c)
            },
            "coefficients": self.coefficients,
            "confidence_metrics": {
                "overall_confidence": self._estimate_overall_confidence(),
                "calibration_freshness": max(0, 1 - (time.time() - self.calibration_timestamp) / (24 * 3600))  # 24h decay
            }
        }

        return report

    def _is_accuracy_improving(self) -> bool:
        """Check if accuracy trend is improving."""
        if len(self.accuracy_trend) < 5:
            return False

        recent = self.accuracy_trend[-5:]
        older = self.accuracy_trend[-10:-5] if len(self.accuracy_trend) >= 10 else []

        if not older:
            return False

        return np.mean(recent) < np.mean(older)

    def _estimate_overall_confidence(self) -> float:
        """Estimate overall mapping confidence."""
        calibration_confidence = np.exp(-self.rms_error_um / 0.3)  # 0.3μm scale

        freshness_hours = (time.time() - self.calibration_timestamp) / 3600
        freshness_confidence = np.exp(-freshness_hours / 24)  # 24h decay

        validation_confidence = 1.0
        if self.validation_points:
            recent_errors = self.accuracy_trend[-10:] if self.accuracy_trend else []
            if recent_errors:
                validation_confidence = np.exp(-np.mean(recent_errors) / 0.3)

        overall_confidence = (calibration_confidence * freshness_confidence * validation_confidence) ** (1/3)

        return min(overall_confidence, 1.0)

    def save_mapping(self, filepath: str) -> None:
        """Save mapping parameters to file."""
        data = {
            "model_type": self.model_type.value,
            "coefficients": self.coefficients,
            "calibration_timestamp": self.calibration_timestamp,
            "calibration_temperature_c": self.calibration_temperature_c,
            "num_calibration_points": self.num_calibration_points,
            "rms_error_um": self.rms_error_um,
            "max_error_um": self.max_error_um,
            "temp_coeff_offset_um_per_c": self.temp_coeff_offset_um_per_c,
            "temp_coeff_linear_um_per_c": self.temp_coeff_linear_um_per_c,
            "validation_points": self.validation_points,
            "accuracy_trend": self.accuracy_trend
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_mapping(cls, filepath: str) -> 'EnhancedParfocalMapping':
        """Load mapping parameters from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        mapping = cls()
        mapping.model_type = MappingModel(data["model_type"])
        mapping.coefficients = data["coefficients"]
        mapping.calibration_timestamp = data["calibration_timestamp"]
        mapping.calibration_temperature_c = data["calibration_temperature_c"]
        mapping.num_calibration_points = data["num_calibration_points"]
        mapping.rms_error_um = data["rms_error_um"]
        mapping.max_error_um = data["max_error_um"]
        mapping.temp_coeff_offset_um_per_c = data["temp_coeff_offset_um_per_c"]
        mapping.temp_coeff_linear_um_per_c = data["temp_coeff_linear_um_per_c"]
        mapping.validation_points = data.get("validation_points", [])
        mapping.accuracy_trend = data.get("accuracy_trend", [])

        return mapping


def create_enhanced_parfocal_mapping(calibration_data: List[Tuple[float, float, float]]) -> EnhancedParfocalMapping:
    """Factory function to create enhanced parfocal mapping from calibration data."""
    mapping = EnhancedParfocalMapping()
    mapping.calibrate_enhanced(calibration_data)
    return mapping