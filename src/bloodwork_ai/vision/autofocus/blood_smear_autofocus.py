from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Callable
import numpy as np

from .camera_interface import CameraInterface
from .stage_interface import StageInterface
from .metrics import (
    tenengrad, variance_of_laplacian, brenner_gradient,
    high_frequency_energy, normalized_dct_energy, metric_fusion
)
from .strategies.coarse_to_fine import CoarseToFineStrategy
from .surface import FocusSurfaceBuilder, FocusSurfaceModel
from .thermal import DriftTracker, ThermalCompensationModel, TemperatureSensor
from .illumination import IlluminationManager, IlluminationController
from .diagnostics import DiagnosticsLogger
from .config import AutofocusConfig, CalibrationData
from .validation import ImageQualityValidator


@dataclass
class BloodSmearAutofocus:
    """Complete autofocus system optimized for blood smear analysis.

    Implements the full system architecture described in the specifications:
    - Multi-metric focus assessment with fusion
    - Coarse-to-fine search with parabolic refinement
    - Predictive focus surface mapping
    - Thermal drift compensation
    - Illumination optimization
    - Comprehensive validation and diagnostics
    """

    camera: CameraInterface
    config: AutofocusConfig = field(default_factory=AutofocusConfig.create_blood_smear_config)

    # Optional components
    stage: Optional[StageInterface] = None
    illumination: Optional[IlluminationController] = None
    temperature_sensor: Optional[TemperatureSensor] = None

    # Internal state
    _drift_tracker: Optional[DriftTracker] = None
    _illumination_manager: Optional[IlluminationManager] = None
    _diagnostics: Optional[DiagnosticsLogger] = None
    _surface_model: Optional[FocusSurfaceModel] = None
    _calibration: Optional[CalibrationData] = None
    _validator: Optional[ImageQualityValidator] = None

    def __post_init__(self) -> None:
        # Initialize drift tracking if temperature sensor available
        if self.temperature_sensor and self.config.thermal.enable_compensation:
            self._drift_tracker = DriftTracker(
                temperature_sensor=self.temperature_sensor,
                thermal_model=ThermalCompensationModel(
                    thermal_coeff=self.config.thermal.thermal_coeff_um_per_c,
                    ref_temperature=self.config.thermal.ref_temperature_c
                )
            )

        # Initialize illumination management
        if self.illumination:
            self._illumination_manager = IlluminationManager(
                controller=self.illumination,
                settle_time_s=self.config.hardware.illumination_settle_time_s
            )

        # Initialize diagnostics if enabled
        if self.config.enable_diagnostics and self.config.diagnostics_path:
            self._diagnostics = DiagnosticsLogger(
                csv_path=self.config.diagnostics_path,
                extras={
                    "system": "blood_smear_autofocus",
                    "version": "1.0"
                }
            )

        # Initialize validator
        self._validator = ImageQualityValidator()

    def _get_focus_metric(self) -> Callable[[np.ndarray], float]:
        """Get the configured focus metric function."""
        metric_name = self.config.metric.primary_metric

        if metric_name == "tenengrad":
            return lambda img: tenengrad(img, ksize=self.config.metric.tenengrad_ksize)
        elif metric_name == "variance_of_laplacian":
            return variance_of_laplacian
        elif metric_name == "brenner_gradient":
            return brenner_gradient
        elif metric_name == "high_frequency_energy":
            return lambda img: high_frequency_energy(img, self.config.metric.hf_cutoff_ratio)
        elif metric_name == "normalized_dct_energy":
            return lambda img: normalized_dct_energy(img, self.config.metric.dct_cutoff_ratio)
        elif metric_name == "metric_fusion":
            return lambda img: metric_fusion(img, self.config.metric.fusion_weights)
        else:
            # Default to tenengrad
            return tenengrad

    def _optimize_illumination(self) -> None:
        """Optimize illumination for best contrast if illumination manager available."""
        if self._illumination_manager is None:
            return

        metric_fn = self._get_focus_metric()

        try:
            best_pattern = self._illumination_manager.optimize_for_contrast(
                self.camera, metric_fn
            )
            if self._diagnostics:
                self._diagnostics.log_measurement(
                    phase="illumination_optimization",
                    z=self.camera.get_focus(),
                    value=0.0,  # Placeholder
                    pattern=best_pattern.name if best_pattern else "unknown"
                )
        except Exception as e:
            if self._diagnostics:
                self._diagnostics.log_measurement(
                    phase="illumination_optimization_error",
                    z=self.camera.get_focus(),
                    value=0.0,
                    error=str(e)
                )

    def autofocus_at_position(self,
                            x: Optional[float] = None,
                            y: Optional[float] = None,
                            z_guess: Optional[float] = None,
                            use_surface_prediction: bool = True) -> float:
        """Perform autofocus at specified XY position.

        Args:
            x, y: Stage coordinates (if stage available)
            z_guess: Initial focus guess (if None, uses current or predicted)
            use_surface_prediction: Whether to use surface model for prediction

        Returns:
            Best focus position found
        """
        start_time = time.time()

        # Move to XY position if stage available
        if self.stage and x is not None and y is not None:
            self.stage.move_xy(x, y)

        # Get initial z guess
        if z_guess is None:
            z_guess = self.camera.get_focus()

            # Use surface prediction if available
            if (use_surface_prediction and self._surface_model and
                x is not None and y is not None):
                try:
                    z_predicted = self._surface_model.predict(x, y)
                    z_guess = z_predicted
                    if self._diagnostics:
                        self._diagnostics.log_measurement(
                            phase="surface_prediction",
                            z=z_predicted,
                            value=0.0,
                            x=x, y=y
                        )
                except Exception as e:
                    if self._diagnostics:
                        self._diagnostics.log_measurement(
                            phase="surface_prediction_error",
                            z=z_guess,
                            value=0.0,
                            error=str(e)
                        )

            # Apply thermal compensation if available
            if self._drift_tracker:
                try:
                    z_compensated = self._drift_tracker.get_predicted_focus(z_guess)
                    z_guess = z_compensated
                    if self._diagnostics:
                        self._diagnostics.log_measurement(
                            phase="thermal_compensation",
                            z=z_compensated,
                            value=0.0
                        )
                except Exception:
                    pass

        # Create autofocus strategy
        metric_fn = self._get_focus_metric()
        strategy = CoarseToFineStrategy(
            camera=self.camera,
            bracket=self.config.search.bracket_um,
            coarse=self.config.search.coarse_step_um,
            fine=self.config.search.fine_step_um,
            metric=metric_fn,
            settle_time_s=self.config.hardware.settle_time_s,
            roi_fracs=self.config.search.roi_fractions,
            max_micro_iters=self.config.search.max_micro_iters,
            diag=self._diagnostics
        )

        # Set initial position
        self.camera.set_focus(z_guess)
        time.sleep(self.config.hardware.settle_time_s)

        # Run autofocus
        try:
            best_z = strategy.find_best_focus(timeout_s=self.config.timeout_s)

            # Record measurement for drift tracking
            if self._drift_tracker:
                current_temp = (self.temperature_sensor.get_temperature()
                              if self.temperature_sensor else None)
                self._drift_tracker.record_focus(best_z, current_temp)

            # Log final result
            if self._diagnostics:
                duration = time.time() - start_time
                self._diagnostics.log_measurement(
                    phase="autofocus_complete",
                    z=best_z,
                    value=strategy.last_best_value or 0.0,
                    duration_s=duration,
                    x=x, y=y
                )

            return best_z

        except Exception as e:
            if self._diagnostics:
                self._diagnostics.log_measurement(
                    phase="autofocus_error",
                    z=z_guess,
                    value=0.0,
                    error=str(e)
                )
            raise

    def build_focus_surface(self,
                          tile_bbox: Tuple[float, float, float, float],
                          grid_points: Optional[int] = None) -> Tuple[FocusSurfaceModel, List[Tuple[float, float, float]]]:
        """Build focus surface model for a tile region.

        Args:
            tile_bbox: (x0, y0, x1, y1) bounding box of tile
            grid_points: Number of grid points per axis (if None, uses config)

        Returns:
            Tuple of (surface_model, sample_points)
        """
        if self.stage is None:
            raise RuntimeError("Stage interface required for surface building")

        grid = grid_points or self.config.surface.grid_points

        # Optimize illumination first
        self._optimize_illumination()

        # Create surface builder
        builder = FocusSurfaceBuilder(
            camera=self.camera,
            stage=self.stage,
            autofocus_fn=lambda z_guess: self.autofocus_at_position(
                z_guess=z_guess,
                use_surface_prediction=False  # Don't use surface during building
            )
        )

        # Build surface
        surface_model, samples = builder.build(
            tile_bbox, grid, self.config.surface.model_type
        )

        # Store for future use
        self._surface_model = surface_model

        if self._diagnostics:
            self._diagnostics.log_measurement(
                phase="surface_built",
                z=0.0,
                value=len(samples),
                model_type=self.config.surface.model_type,
                grid_points=grid
            )

        return surface_model, samples

    def validate_performance(self,
                           test_positions: Optional[List[float]] = None,
                           num_repeats: int = 5) -> Dict[str, Any]:
        """Validate autofocus performance.

        Args:
            test_positions: Z positions to test (if None, uses config)
            num_repeats: Number of repeats per position

        Returns:
            Validation results dictionary
        """
        if test_positions is None:
            test_positions = list(self.config.validation.accuracy_test_positions)

        results = {
            "timestamp": time.time(),
            "test_positions": test_positions,
            "num_repeats": num_repeats,
            "measurements": [],
            "summary": {}
        }

        all_errors = []
        all_repeatabilities = []

        for z_true in test_positions:
            position_results = []

            for repeat in range(num_repeats):
                # Add random offset to starting position
                z_start = z_true + np.random.normal(0, 2.0)

                try:
                    z_measured = self.autofocus_at_position(z_guess=z_start)
                    error = abs(z_measured - z_true)
                    all_errors.append(error)

                    position_results.append({
                        "repeat": repeat,
                        "z_true": z_true,
                        "z_start": z_start,
                        "z_measured": z_measured,
                        "error_um": error
                    })

                except Exception as e:
                    position_results.append({
                        "repeat": repeat,
                        "z_true": z_true,
                        "z_start": z_start,
                        "error": str(e)
                    })

            # Calculate repeatability for this position
            valid_z = [r["z_measured"] for r in position_results
                      if "z_measured" in r]
            if len(valid_z) >= 2:
                repeatability = float(np.std(valid_z))
                all_repeatabilities.append(repeatability)
            else:
                repeatability = None

            results["measurements"].append({
                "z_true": z_true,
                "results": position_results,
                "repeatability_um": repeatability,
                "success_rate": len(valid_z) / num_repeats
            })

        # Calculate summary statistics
        if all_errors:
            max_error = float(np.max(all_errors))
            mean_error = float(np.mean(all_errors))
            max_repeatability = (float(np.max(all_repeatabilities))
                               if all_repeatabilities else None)

            results["summary"] = {
                "max_error_um": max_error,
                "mean_error_um": mean_error,
                "max_repeatability_um": max_repeatability,
                "accuracy_pass": max_error <= self.config.validation.max_focus_error_um,
                "repeatability_pass": (max_repeatability <= self.config.validation.max_repeatability_um
                                     if max_repeatability is not None else False),
                "overall_pass": (max_error <= self.config.validation.max_focus_error_um and
                               (max_repeatability <= self.config.validation.max_repeatability_um
                                if max_repeatability is not None else False))
            }

        return results

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "timestamp": time.time(),
            "focus_position": self.camera.get_focus(),
            "focus_range": self.camera.get_focus_range(),
        }

        # Add stage position if available
        if self.stage:
            status["stage_position"] = self.stage.get_xy()

        # Add thermal information
        if self._drift_tracker:
            status["thermal"] = self._drift_tracker.get_drift_statistics()

        # Add illumination status
        if self._illumination_manager:
            pattern = self._illumination_manager.get_pattern()
            status["illumination"] = {
                "current_pattern": pattern.name if pattern else None,
                "led_count": self.illumination.get_led_count()
            }

        # Add surface model info
        if self._surface_model:
            status["surface_model"] = {
                "type": self._surface_model.kind,
                "origin": self._surface_model.ref_origin
            }

        return status

    def close(self) -> None:
        """Clean up resources."""
        if self._diagnostics:
            self._diagnostics.close()


# Factory function for easy setup
def create_blood_smear_autofocus(camera: CameraInterface,
                                stage: Optional[StageInterface] = None,
                                illumination: Optional[IlluminationController] = None,
                                temperature_sensor: Optional[TemperatureSensor] = None,
                                config: Optional[AutofocusConfig] = None) -> BloodSmearAutofocus:
    """Factory function to create a configured blood smear autofocus system.

    Args:
        camera: Camera interface (required)
        stage: Stage interface for XY movement (optional)
        illumination: Illumination controller (optional)
        temperature_sensor: Temperature sensor for drift compensation (optional)
        config: System configuration (if None, uses blood smear defaults)

    Returns:
        Configured BloodSmearAutofocus instance
    """
    if config is None:
        config = AutofocusConfig.create_blood_smear_config()

    return BloodSmearAutofocus(
        camera=camera,
        stage=stage,
        illumination=illumination,
        temperature_sensor=temperature_sensor,
        config=config
    )