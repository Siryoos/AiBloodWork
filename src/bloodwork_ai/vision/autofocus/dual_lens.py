from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Protocol, Any
from enum import Enum
import numpy as np

from .config import AutofocusConfig, MetricConfig
from .illumination import IlluminationController


class LensID(Enum):
    """Lens identifiers for dual-lens system."""
    LENS_A = "lens_a"  # 10-20x low NA for scanning
    LENS_B = "lens_b"  # 40-60x high NA for detailed analysis


@dataclass
class LensProfile:
    """Physical and optical properties of a lens."""

    # Identification
    lens_id: LensID
    name: str
    serial_number: Optional[str] = None

    # Optical properties
    magnification: float = 20.0  # e.g., 20x
    numerical_aperture: float = 0.4  # e.g., 0.4 NA
    working_distance_mm: float = 2.0
    field_of_view_um: float = 500.0
    depth_of_field_um: float = 2.0

    # Mechanical properties
    z_range_um: Tuple[float, float] = (-50.0, 50.0)
    z_resolution_um: float = 0.1
    parfocal_offset_um: float = 0.0  # Offset from reference plane

    # Autofocus configuration
    af_config: AutofocusConfig = field(default_factory=lambda: AutofocusConfig.create_blood_smear_config())
    metric_weights: Dict[str, float] = field(default_factory=lambda: {"tenengrad": 0.4, "laplacian": 0.3, "brenner": 0.3})

    # Illumination preferences
    preferred_illum_pattern: str = "BRIGHTFIELD"
    illum_intensity_factor: float = 1.0

    # Performance characteristics
    focus_speed_um_per_s: float = 200.0
    settle_time_ms: float = 10.0

    def get_z_center(self) -> float:
        """Get center of Z range."""
        return (self.z_range_um[0] + self.z_range_um[1]) / 2.0

    def is_z_in_range(self, z_um: float) -> bool:
        """Check if Z position is within lens range."""
        return self.z_range_um[0] <= z_um <= self.z_range_um[1]


@dataclass
class ParfocalMapping:
    """Cross-lens parfocal mapping for coordinated focus."""

    # Mapping coefficients (lens_a_z = a + b * lens_b_z + c * lens_b_z^2)
    linear_coeff: float = 1.0  # b coefficient
    quadratic_coeff: float = 0.0  # c coefficient
    offset_um: float = 0.0  # a coefficient (constant offset)

    # Calibration metadata
    calibration_timestamp: float = 0.0
    temperature_c: float = 23.0
    num_calibration_points: int = 0
    rms_error_um: float = 0.0

    # Temperature compensation
    temp_coeff_um_per_c: float = 0.0

    def map_lens_b_to_a(self, z_b_um: float, temperature_c: float = 23.0) -> float:
        """Map Lens-B focus position to Lens-A equivalent."""
        # Apply temperature compensation
        temp_offset = (temperature_c - self.temperature_c) * self.temp_coeff_um_per_c

        # Apply polynomial mapping
        z_a_um = (self.offset_um +
                  self.linear_coeff * z_b_um +
                  self.quadratic_coeff * z_b_um**2 +
                  temp_offset)

        return z_a_um

    def map_lens_a_to_b(self, z_a_um: float, temperature_c: float = 23.0) -> float:
        """Map Lens-A focus position to Lens-B equivalent (inverse mapping)."""
        # For now, use linear approximation of inverse
        # More sophisticated inverse could be implemented if needed
        temp_offset = (temperature_c - self.temperature_c) * self.temp_coeff_um_per_c
        z_a_corrected = z_a_um - self.offset_um - temp_offset

        if abs(self.linear_coeff) > 1e-6:
            z_b_um = z_a_corrected / self.linear_coeff
            # Apply quadratic correction iteratively if significant
            if abs(self.quadratic_coeff) > 1e-6:
                for _ in range(3):  # Newton's method iterations
                    f = self.linear_coeff * z_b_um + self.quadratic_coeff * z_b_um**2 - z_a_corrected
                    df = self.linear_coeff + 2 * self.quadratic_coeff * z_b_um
                    if abs(df) > 1e-6:
                        z_b_um -= f / df
            return z_b_um
        else:
            return 0.0  # Degenerate case

    def get_mapping_accuracy_um(self) -> float:
        """Get expected mapping accuracy."""
        return max(0.1, self.rms_error_um * 2.0)  # 2-sigma estimate


class CameraInterface(Protocol):
    """Protocol for camera interface in dual-lens system."""

    def get_frame(self) -> np.ndarray:
        """Capture frame from current active lens."""
        ...

    def set_active_lens(self, lens_id: LensID) -> None:
        """Switch to specified lens."""
        ...

    def get_active_lens(self) -> LensID:
        """Get currently active lens."""
        ...

    def set_focus(self, z_um: float) -> None:
        """Set focus position for active lens."""
        ...

    def get_focus(self) -> float:
        """Get focus position of active lens."""
        ...


@dataclass
class DualLensHandoffResult:
    """Result of cross-lens handoff operation."""

    # Handoff success
    success: bool
    elapsed_ms: float

    # Focus positions
    source_lens: LensID
    target_lens: LensID
    source_z_um: float
    target_z_um: float

    # Mapping validation
    predicted_z_um: float
    actual_z_um: float
    mapping_error_um: float

    # Performance metrics
    focus_metric_source: float
    focus_metric_target: float
    metric_consistency: float

    # Status flags
    flags: List[str] = field(default_factory=list)

    def meets_performance_target(self) -> bool:
        """Check if handoff meets ≤300ms target."""
        return self.elapsed_ms <= 300.0


class DualLensAutofocusManager:
    """Main controller for dual-lens autofocus system."""

    def __init__(self,
                 camera: CameraInterface,
                 stage_controller,
                 illumination: IlluminationController,
                 lens_a_profile: LensProfile,
                 lens_b_profile: LensProfile,
                 parfocal_mapping: Optional[ParfocalMapping] = None):
        """Initialize dual-lens autofocus system.

        Args:
            camera: Dual-lens camera interface
            stage_controller: XYZ stage controller
            illumination: Illumination controller
            lens_a_profile: Lens-A (scanning) profile
            lens_b_profile: Lens-B (detailed) profile
            parfocal_mapping: Cross-lens parfocal mapping
        """
        self.camera = camera
        self.stage = stage_controller
        self.illumination = illumination

        # Lens profiles
        self.profiles = {
            LensID.LENS_A: lens_a_profile,
            LensID.LENS_B: lens_b_profile
        }

        # Parfocal mapping
        self.parfocal_mapping = parfocal_mapping or ParfocalMapping()

        # Current state
        self.active_lens = camera.get_active_lens()
        self._lock = threading.Lock()

        # Performance tracking
        self.handoff_history: List[DualLensHandoffResult] = []
        self.calibration_data: Dict[str, Any] = {}

        # Temperature monitoring
        self._last_temperature_c: float = 23.0

    def autofocus_scanning(self, x_um: float, y_um: float,
                          z_guess_um: Optional[float] = None) -> float:
        """Perform autofocus optimized for scanning with Lens-A.

        Args:
            x_um, y_um: Target position
            z_guess_um: Initial focus guess

        Returns:
            Final focus position in micrometers
        """
        with self._lock:
            # Ensure Lens-A is active
            if self.active_lens != LensID.LENS_A:
                self._switch_to_lens(LensID.LENS_A)

            # Move to position
            self.stage.move_xy(x_um, y_um)

            # Set up illumination for scanning
            profile = self.profiles[LensID.LENS_A]
            self._configure_illumination_for_lens(LensID.LENS_A)

            # Perform autofocus with scanning-optimized parameters
            config = profile.af_config

            # Use coarse search for speed
            z_range = profile.z_range_um
            if z_guess_um is not None and profile.is_z_in_range(z_guess_um):
                # Narrow search around guess
                search_range = min(10.0, config.search.coarse_range_um)
                z_range = (max(z_range[0], z_guess_um - search_range/2),
                          min(z_range[1], z_guess_um + search_range/2))

            z_af_um = self._perform_autofocus(z_range, config)

            return z_af_um

    def autofocus_detailed(self, x_um: float, y_um: float,
                          z_guess_um: Optional[float] = None) -> float:
        """Perform high-precision autofocus with Lens-B.

        Args:
            x_um, y_um: Target position
            z_guess_um: Initial focus guess

        Returns:
            Final focus position in micrometers
        """
        with self._lock:
            # Ensure Lens-B is active
            if self.active_lens != LensID.LENS_B:
                self._switch_to_lens(LensID.LENS_B)

            # Move to position
            self.stage.move_xy(x_um, y_um)

            # Configure for high-precision analysis
            profile = self.profiles[LensID.LENS_B]
            self._configure_illumination_for_lens(LensID.LENS_B)

            # Perform high-precision autofocus
            config = profile.af_config
            z_range = profile.z_range_um

            if z_guess_um is not None and profile.is_z_in_range(z_guess_um):
                # Use guess to narrow search
                search_range = config.search.coarse_range_um
                z_range = (max(z_range[0], z_guess_um - search_range/2),
                          min(z_range[1], z_guess_um + search_range/2))

            z_af_um = self._perform_autofocus(z_range, config)

            return z_af_um

    def handoff_a_to_b(self, source_z_um: float) -> DualLensHandoffResult:
        """Fast handoff from Lens-A to Lens-B with parfocal mapping.

        Target: ≤300ms total handoff time

        Args:
            source_z_um: Focus position from Lens-A

        Returns:
            Handoff result with performance metrics
        """
        start_time = time.time()

        try:
            with self._lock:
                # Sync active lens state with camera
                self._sync_active_lens()

                # Ensure we're starting from Lens-A
                if self.active_lens != LensID.LENS_A:
                    raise ValueError("Handoff requires starting from Lens-A")

                # Predict Lens-B focus position using parfocal mapping
                predicted_z_b = self.parfocal_mapping.map_lens_a_to_b(
                    source_z_um, self._last_temperature_c)

                # Validate predicted position is within Lens-B range
                profile_b = self.profiles[LensID.LENS_B]
                if not profile_b.is_z_in_range(predicted_z_b):
                    # Clamp to range
                    predicted_z_b = np.clip(predicted_z_b,
                                          profile_b.z_range_um[0],
                                          profile_b.z_range_um[1])

                # Measure focus quality at source
                frame_a = self.camera.get_frame()
                metric_a = self._compute_focus_metric(frame_a, LensID.LENS_A)

                # Switch to Lens-B
                self._switch_to_lens(LensID.LENS_B)

                # Set predicted focus position
                self.camera.set_focus(predicted_z_b)
                time.sleep(profile_b.settle_time_ms / 1000.0)  # Allow settling

                # Configure illumination for Lens-B
                self._configure_illumination_for_lens(LensID.LENS_B)

                # Fine-tune focus with limited search (±1μm for speed)
                fine_range = (predicted_z_b - 1.0, predicted_z_b + 1.0)
                fine_range = (max(fine_range[0], profile_b.z_range_um[0]),
                             min(fine_range[1], profile_b.z_range_um[1]))

                actual_z_b = self._perform_fine_autofocus(fine_range, profile_b.af_config)

                # Measure focus quality at target
                frame_b = self.camera.get_frame()
                metric_b = self._compute_focus_metric(frame_b, LensID.LENS_B)

                # Calculate performance metrics
                mapping_error = abs(actual_z_b - predicted_z_b)
                metric_consistency = min(metric_a, metric_b) / max(metric_a, metric_b) if max(metric_a, metric_b) > 0 else 0

                elapsed_ms = (time.time() - start_time) * 1000.0

                # Create result
                result = DualLensHandoffResult(
                    success=True,
                    elapsed_ms=elapsed_ms,
                    source_lens=LensID.LENS_A,
                    target_lens=LensID.LENS_B,
                    source_z_um=source_z_um,
                    target_z_um=actual_z_b,
                    predicted_z_um=predicted_z_b,
                    actual_z_um=actual_z_b,
                    mapping_error_um=mapping_error,
                    focus_metric_source=metric_a,
                    focus_metric_target=metric_b,
                    metric_consistency=metric_consistency
                )

                # Add performance flags
                if elapsed_ms > 300.0:
                    result.flags.append("SLOW_HANDOFF")
                if mapping_error > 1.0:
                    result.flags.append("POOR_MAPPING")
                if metric_consistency < 0.8:
                    result.flags.append("METRIC_INCONSISTENCY")

                # Store in history
                self.handoff_history.append(result)
                if len(self.handoff_history) > 1000:
                    self.handoff_history = self.handoff_history[-500:]  # Keep recent history

                return result

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000.0
            return DualLensHandoffResult(
                success=False,
                elapsed_ms=elapsed_ms,
                source_lens=LensID.LENS_A,
                target_lens=LensID.LENS_B,
                source_z_um=source_z_um,
                target_z_um=0.0,
                predicted_z_um=0.0,
                actual_z_um=0.0,
                mapping_error_um=999.0,
                focus_metric_source=0.0,
                focus_metric_target=0.0,
                metric_consistency=0.0,
                flags=[f"ERROR:{str(e)}"]
            )

    def handoff_b_to_a(self, source_z_um: float) -> DualLensHandoffResult:
        """Fast handoff from Lens-B to Lens-A."""
        start_time = time.time()

        try:
            with self._lock:
                # Sync active lens state with camera
                self._sync_active_lens()

                if self.active_lens != LensID.LENS_B:
                    raise ValueError("Handoff requires starting from Lens-B")

                # Map B→A using inverse mapping
                predicted_z_a = self.parfocal_mapping.map_lens_b_to_a(
                    source_z_um, self._last_temperature_c)

                profile_a = self.profiles[LensID.LENS_A]
                if not profile_a.is_z_in_range(predicted_z_a):
                    predicted_z_a = np.clip(predicted_z_a,
                                          profile_a.z_range_um[0],
                                          profile_a.z_range_um[1])

                # Measure source metric
                frame_b = self.camera.get_frame()
                metric_b = self._compute_focus_metric(frame_b, LensID.LENS_B)

                # Switch to Lens-A
                self._switch_to_lens(LensID.LENS_A)
                self.camera.set_focus(predicted_z_a)
                time.sleep(profile_a.settle_time_ms / 1000.0)

                self._configure_illumination_for_lens(LensID.LENS_A)

                # Fine-tune
                fine_range = (predicted_z_a - 1.5, predicted_z_a + 1.5)  # Slightly larger for lower magnification
                fine_range = (max(fine_range[0], profile_a.z_range_um[0]),
                             min(fine_range[1], profile_a.z_range_um[1]))

                actual_z_a = self._perform_fine_autofocus(fine_range, profile_a.af_config)

                frame_a = self.camera.get_frame()
                metric_a = self._compute_focus_metric(frame_a, LensID.LENS_A)

                mapping_error = abs(actual_z_a - predicted_z_a)
                metric_consistency = min(metric_a, metric_b) / max(metric_a, metric_b) if max(metric_a, metric_b) > 0 else 0

                elapsed_ms = (time.time() - start_time) * 1000.0

                result = DualLensHandoffResult(
                    success=True,
                    elapsed_ms=elapsed_ms,
                    source_lens=LensID.LENS_B,
                    target_lens=LensID.LENS_A,
                    source_z_um=source_z_um,
                    target_z_um=actual_z_a,
                    predicted_z_um=predicted_z_a,
                    actual_z_um=actual_z_a,
                    mapping_error_um=mapping_error,
                    focus_metric_source=metric_b,
                    focus_metric_target=metric_a,
                    metric_consistency=metric_consistency
                )

                if elapsed_ms > 300.0:
                    result.flags.append("SLOW_HANDOFF")
                if mapping_error > 1.5:  # Slightly more tolerance for B→A
                    result.flags.append("POOR_MAPPING")
                if metric_consistency < 0.7:
                    result.flags.append("METRIC_INCONSISTENCY")

                self.handoff_history.append(result)
                return result

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000.0
            return DualLensHandoffResult(
                success=False,
                elapsed_ms=elapsed_ms,
                source_lens=LensID.LENS_B,
                target_lens=LensID.LENS_A,
                source_z_um=source_z_um,
                target_z_um=0.0,
                predicted_z_um=0.0,
                actual_z_um=0.0,
                mapping_error_um=999.0,
                focus_metric_source=0.0,
                focus_metric_target=0.0,
                metric_consistency=0.0,
                flags=[f"ERROR:{str(e)}"]
            )

    def _switch_to_lens(self, lens_id: LensID) -> None:
        """Switch active lens and update state."""
        if self.active_lens != lens_id:
            self.camera.set_active_lens(lens_id)
            self.active_lens = lens_id
            # Small delay for lens switching mechanics
            time.sleep(0.05)

    def _sync_active_lens(self) -> None:
        """Sync active lens state with camera."""
        self.active_lens = self.camera.get_active_lens()

    def _configure_illumination_for_lens(self, lens_id: LensID) -> None:
        """Configure illumination optimized for specific lens."""
        profile = self.profiles[lens_id]

        # Set intensity based on lens characteristics
        base_intensity = 0.5 * profile.illum_intensity_factor

        # Apply lens-specific pattern
        if hasattr(self.illumination, 'set_pattern_by_name'):
            self.illumination.set_pattern_by_name(profile.preferred_illum_pattern)

        if hasattr(self.illumination, 'set_intensity'):
            self.illumination.set_intensity(base_intensity)

    def _perform_autofocus(self, z_range: Tuple[float, float], config: AutofocusConfig) -> float:
        """Perform autofocus within specified range."""
        from .blood_smear_autofocus import BloodSmearAutofocus

        # Create temporary autofocus instance for this operation
        # In production, this would be optimized to reuse instances
        af_system = BloodSmearAutofocus(
            camera=self.camera,
            stage=self.stage,
            illumination=self.illumination,
            config=config
        )

        # Set search range
        original_range = config.search.coarse_range_um
        config.search.coarse_range_um = z_range[1] - z_range[0]

        try:
            result = af_system.autofocus_at_position()
            return result
        finally:
            config.search.coarse_range_um = original_range
            af_system.close()

    def _perform_fine_autofocus(self, z_range: Tuple[float, float], config: AutofocusConfig) -> float:
        """Perform fine autofocus with limited search range for speed."""
        # Simple hill-climbing for fine adjustment
        z_start = (z_range[0] + z_range[1]) / 2.0
        self.camera.set_focus(z_start)

        best_z = z_start
        best_metric = self._compute_focus_metric(self.camera.get_frame(), self.active_lens)

        # Search with 0.2μm steps
        step_size = 0.2
        search_positions = np.arange(z_range[0], z_range[1] + step_size, step_size)

        for z in search_positions:
            self.camera.set_focus(z)
            frame = self.camera.get_frame()
            metric = self._compute_focus_metric(frame, self.active_lens)

            if metric > best_metric:
                best_metric = metric
                best_z = z

        # Set final position
        self.camera.set_focus(best_z)
        return best_z

    def _compute_focus_metric(self, frame: np.ndarray, lens_id: LensID) -> float:
        """Compute focus metric weighted for specific lens."""
        from .metrics import tenengrad, variance_of_laplacian, brenner_gradient

        profile = self.profiles[lens_id]
        weights = profile.metric_weights

        # Compute individual metrics
        tenengrad_score = tenengrad(frame) if "tenengrad" in weights else 0
        laplacian_score = variance_of_laplacian(frame) if "laplacian" in weights else 0
        brenner_score = brenner_gradient(frame) if "brenner" in weights else 0

        # Weighted combination
        total_score = (weights.get("tenengrad", 0) * tenengrad_score +
                      weights.get("laplacian", 0) * laplacian_score +
                      weights.get("brenner", 0) * brenner_score)

        return total_score

    def get_handoff_performance_stats(self) -> Dict[str, float]:
        """Get handoff performance statistics."""
        if not self.handoff_history:
            return {"status": "no_data"}

        recent_handoffs = [h for h in self.handoff_history[-50:] if h.success]

        if not recent_handoffs:
            return {"status": "no_successful_handoffs"}

        times = [h.elapsed_ms for h in recent_handoffs]
        errors = [h.mapping_error_um for h in recent_handoffs]

        return {
            "avg_handoff_time_ms": np.mean(times),
            "p95_handoff_time_ms": np.percentile(times, 95),
            "max_handoff_time_ms": np.max(times),
            "target_met_rate": np.mean([h.meets_performance_target() for h in recent_handoffs]),
            "avg_mapping_error_um": np.mean(errors),
            "p95_mapping_error_um": np.percentile(errors, 95),
            "handoff_count": len(recent_handoffs)
        }

    def update_temperature(self, temperature_c: float) -> None:
        """Update system temperature for thermal compensation."""
        self._last_temperature_c = temperature_c

    def close(self) -> None:
        """Clean up dual-lens system."""
        pass