from __future__ import annotations

import time
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

try:
    import grpc
    from google.protobuf import json_format
    HAS_GRPC = True
except ImportError:
    HAS_GRPC = False
    grpc = None


@dataclass
class ROISpec:
    """Region of Interest specification."""
    pattern: str = "CENTER_PLUS_CORNERS"  # CENTER, CENTER_PLUS_CORNERS, CUSTOM
    size_um: float = 80.0
    custom_rois: Optional[List[Tuple[float, float, float, float]]] = None


@dataclass
class AutofocusRequest:
    """Autofocus request following the roadmap specification."""

    # Required fields
    tile_id: str
    x_um: float
    y_um: float

    # Optional fields with defaults
    z_guess_um: Optional[float] = None
    illum_profile: str = "LED_ANGLE_25"
    policy: str = "RBC_LAYER"  # RBC_LAYER, WBC_NUCLEUS, ADAPTIVE
    roi_spec: ROISpec = None

    # Advanced options
    timeout_ms: Optional[int] = None
    use_surface_prediction: bool = True
    enable_thermal_compensation: bool = True
    metric_weights: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if self.roi_spec is None:
            self.roi_spec = ROISpec()

    @classmethod
    def from_json(cls, json_data: Union[str, Dict[str, Any]]) -> "AutofocusRequest":
        """Create request from JSON data."""
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data

        # Handle nested ROI spec
        roi_data = data.get("roi_spec", {})
        if isinstance(roi_data, dict):
            roi_spec = ROISpec(**roi_data)
        else:
            roi_spec = ROISpec()

        # Remove roi_spec from data to avoid duplicate
        data = data.copy()
        data.pop("roi_spec", None)

        return cls(roi_spec=roi_spec, **data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        return result


@dataclass
class MetricResult:
    """Focus metric results."""
    tenengrad: Optional[float] = None
    hf_energy: Optional[float] = None
    brenner: Optional[float] = None
    laplacian: Optional[float] = None
    dct_energy: Optional[float] = None
    fusion_score: Optional[float] = None


@dataclass
class FocusSample:
    """Single focus measurement sample."""
    z: float
    metric: float
    timestamp_ms: Optional[float] = None


@dataclass
class AutofocusResponse:
    """Autofocus response following the roadmap specification."""

    # Core results
    z_af_um: float
    metric: MetricResult
    elapsed_ms: float

    # Optional detailed results
    samples: Optional[List[FocusSample]] = None
    surface_residual_um: Optional[float] = None
    thermal_compensation_um: Optional[float] = None

    # Status and flags
    status: str = "success"  # success, failed, timeout, error
    flags: List[str] = None
    error_message: Optional[str] = None

    # Traceability
    request_id: Optional[str] = None
    timestamp: Optional[float] = None
    algorithm_version: str = "1.0"
    config_hash: Optional[str] = None

    def __post_init__(self):
        if self.flags is None:
            self.flags = []
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class BatchAutofocusRequest:
    """Batch autofocus request for multiple tiles."""
    requests: List[AutofocusRequest]
    batch_id: Optional[str] = None
    parallel_processing: bool = True
    max_workers: int = 4


@dataclass
class BatchAutofocusResponse:
    """Batch autofocus response."""
    responses: List[AutofocusResponse]
    total_elapsed_ms: float
    success_count: int
    failure_count: int
    batch_id: Optional[str] = None
    batch_status: str = "completed"


class AutofocusAPIServer:
    """Production autofocus API server implementation."""

    def __init__(self, autofocus_system, enable_telemetry: bool = True):
        """Initialize API server.

        Args:
            autofocus_system: BloodSmearAutofocus instance
            enable_telemetry: Whether to log all requests/responses
        """
        self.autofocus_system = autofocus_system
        self.enable_telemetry = enable_telemetry
        self.request_count = 0
        self.telemetry_log: List[Dict[str, Any]] = []

    def process_autofocus_request(self, request: AutofocusRequest) -> AutofocusResponse:
        """Process single autofocus request."""
        start_time = time.time()
        self.request_count += 1

        try:
            # Validate request
            self._validate_request(request)

            # Configure system based on request
            self._apply_request_config(request)

            # Execute autofocus
            result = self._execute_autofocus(request)

            # Create response
            response = self._create_response(request, result, start_time)

            # Log telemetry
            if self.enable_telemetry:
                self._log_telemetry(request, response)

            return response

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            error_response = AutofocusResponse(
                z_af_um=request.z_guess_um or 0.0,
                metric=MetricResult(),
                elapsed_ms=elapsed_ms,
                status="error",
                error_message=str(e),
                flags=["PROCESSING_ERROR"]
            )

            if self.enable_telemetry:
                self._log_telemetry(request, error_response)

            return error_response

    def _validate_request(self, request: AutofocusRequest) -> None:
        """Validate autofocus request."""
        if not request.tile_id:
            raise ValueError("tile_id is required")

        # Validate coordinates (basic sanity check)
        if abs(request.x_um) > 1e6 or abs(request.y_um) > 1e6:
            raise ValueError("Coordinates out of reasonable range")

        # Validate illumination profile
        valid_profiles = [
            "BRIGHTFIELD", "DARKFIELD", "LED_ANGLE_0", "LED_ANGLE_25",
            "LED_ANGLE_45", "LED_ANGLE_90", "MULTI_ANGLE"
        ]
        if request.illum_profile not in valid_profiles:
            raise ValueError(f"Invalid illumination profile: {request.illum_profile}")

        # Validate policy
        valid_policies = ["RBC_LAYER", "WBC_NUCLEUS", "ADAPTIVE"]
        if request.policy not in valid_policies:
            raise ValueError(f"Invalid policy: {request.policy}")

    def _apply_request_config(self, request: AutofocusRequest) -> None:
        """Apply request-specific configuration to autofocus system."""
        # Set illumination if available
        if hasattr(self.autofocus_system, '_illumination_manager'):
            self._set_illumination_from_profile(request.illum_profile)

        # Apply metric weights if provided
        if request.metric_weights and hasattr(self.autofocus_system, 'config'):
            self.autofocus_system.config.metric.fusion_weights.update(request.metric_weights)

    def _set_illumination_from_profile(self, profile: str) -> None:
        """Set illumination based on profile name."""
        if not hasattr(self.autofocus_system, '_illumination_manager'):
            return

        manager = self.autofocus_system._illumination_manager
        if profile == "BRIGHTFIELD":
            manager.set_pattern(manager.controller.get_led_count())
        elif profile == "DARKFIELD":
            from .illumination import IlluminationPatterns
            pattern = IlluminationPatterns.darkfield(0.7, manager.controller.get_led_count())
            manager.set_pattern(pattern)
        # Add more profile mappings as needed

    def _execute_autofocus(self, request: AutofocusRequest) -> Dict[str, Any]:
        """Execute the autofocus operation."""
        # Move to position and run autofocus
        best_z = self.autofocus_system.autofocus_at_position(
            x=request.x_um,
            y=request.y_um,
            z_guess=request.z_guess_um,
            use_surface_prediction=request.use_surface_prediction
        )

        # Get additional metrics if available
        metrics = self._collect_metrics(request)

        # Get surface residual if surface model exists
        surface_residual = None
        if self.autofocus_system._surface_model and hasattr(self.autofocus_system, 'stage'):
            try:
                predicted_z = self.autofocus_system._surface_model.predict(
                    request.x_um, request.y_um
                )
                surface_residual = abs(best_z - predicted_z)
            except Exception:
                pass

        return {
            "z_af_um": best_z,
            "metrics": metrics,
            "surface_residual_um": surface_residual
        }

    def _collect_metrics(self, request: AutofocusRequest) -> MetricResult:
        """Collect focus metrics from the current frame."""
        try:
            frame = self.autofocus_system.camera.get_frame()

            from .metrics import (
                tenengrad, variance_of_laplacian, brenner_gradient,
                high_frequency_energy, block_dct_energy, metric_fusion
            )

            # Create ROI fractions from request
            roi_fracs = None
            if request.roi_spec.pattern == "CENTER":
                roi_fracs = (0.375, 0.375, 0.25, 0.25)  # Center 25%
            elif request.roi_spec.pattern == "CENTER_PLUS_CORNERS":
                # Use center ROI for now (could be extended to multi-ROI)
                roi_fracs = (0.25, 0.25, 0.5, 0.5)

            return MetricResult(
                tenengrad=tenengrad(frame, roi_fracs=roi_fracs),
                hf_energy=high_frequency_energy(frame),
                brenner=brenner_gradient(frame, roi_fracs=roi_fracs),
                laplacian=variance_of_laplacian(frame),
                dct_energy=block_dct_energy(frame, roi_fracs=roi_fracs),
                fusion_score=metric_fusion(frame)
            )

        except Exception:
            return MetricResult()

    def _create_response(self,
                        request: AutofocusRequest,
                        result: Dict[str, Any],
                        start_time: float) -> AutofocusResponse:
        """Create autofocus response from results."""
        elapsed_ms = (time.time() - start_time) * 1000

        flags = []
        if elapsed_ms > 150:  # Target is 80-150ms
            flags.append("SLOW_RESPONSE")

        return AutofocusResponse(
            z_af_um=result["z_af_um"],
            metric=result["metrics"],
            elapsed_ms=elapsed_ms,
            surface_residual_um=result.get("surface_residual_um"),
            status="success",
            flags=flags,
            request_id=request.tile_id
        )

    def _log_telemetry(self, request: AutofocusRequest, response: AutofocusResponse) -> None:
        """Log telemetry data."""
        telemetry_entry = {
            "timestamp": time.time(),
            "tile_id": request.tile_id,
            "x_um": request.x_um,
            "y_um": request.y_um,
            "z_af_um": response.z_af_um,
            "elapsed_ms": response.elapsed_ms,
            "status": response.status,
            "illum_profile": request.illum_profile,
            "policy": request.policy,
            "flags": response.flags
        }

        self.telemetry_log.append(telemetry_entry)

        # Limit telemetry log size
        if len(self.telemetry_log) > 10000:
            self.telemetry_log = self.telemetry_log[-5000:]

    def get_telemetry_summary(self) -> Dict[str, Any]:
        """Get telemetry summary statistics."""
        if not self.telemetry_log:
            return {"status": "no_data"}

        recent_entries = self.telemetry_log[-1000:]  # Last 1000 requests
        elapsed_times = [e["elapsed_ms"] for e in recent_entries if "elapsed_ms" in e]
        success_count = sum(1 for e in recent_entries if e.get("status") == "success")

        import numpy as np

        return {
            "total_requests": len(self.telemetry_log),
            "recent_requests": len(recent_entries),
            "success_rate": success_count / len(recent_entries) if recent_entries else 0,
            "avg_elapsed_ms": float(np.mean(elapsed_times)) if elapsed_times else 0,
            "p95_elapsed_ms": float(np.percentile(elapsed_times, 95)) if elapsed_times else 0,
            "throughput_target_met": (
                float(np.percentile(elapsed_times, 95)) <= 150 if elapsed_times else False
            )
        }

    def export_telemetry(self, filepath: Union[str, Path]) -> None:
        """Export telemetry data to file."""
        import csv

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', newline='') as f:
            if not self.telemetry_log:
                return

            fieldnames = self.telemetry_log[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.telemetry_log)


# Factory function for easy API setup
def create_autofocus_api(autofocus_system,
                        enable_telemetry: bool = True) -> AutofocusAPIServer:
    """Create autofocus API server instance.

    Args:
        autofocus_system: BloodSmearAutofocus instance
        enable_telemetry: Whether to enable telemetry logging

    Returns:
        Configured API server
    """
    return AutofocusAPIServer(autofocus_system, enable_telemetry)


# Example usage functions
def example_json_request() -> str:
    """Example JSON request for documentation."""
    request = AutofocusRequest(
        tile_id="T123",
        x_um=215000.0,
        y_um=74000.0,
        z_guess_um=120.0,
        illum_profile="LED_ANGLE_25",
        policy="RBC_LAYER",
        roi_spec=ROISpec(pattern="CENTER_PLUS_CORNERS", size_um=80)
    )
    return json.dumps(request.to_dict(), indent=2)


def example_json_response() -> str:
    """Example JSON response for documentation."""
    response = AutofocusResponse(
        z_af_um=121.1,
        metric=MetricResult(
            tenengrad=8.42e6,
            hf_energy=1.27e5,
            brenner=3.45e5,
            laplacian=2.18e4
        ),
        elapsed_ms=94,
        surface_residual_um=0.6,
        status="success",
        flags=[]
    )
    return response.to_json()