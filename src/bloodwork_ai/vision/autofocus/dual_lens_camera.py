from __future__ import annotations

import time
import threading
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .dual_lens import LensID, LensProfile


class AcquisitionMode(Enum):
    """Camera acquisition modes for dual-lens system."""
    SINGLE_LENS = "single_lens"          # Use one lens at a time
    SIMULTANEOUS = "simultaneous"        # Capture from both lenses simultaneously
    ALTERNATING = "alternating"          # Rapid alternation between lenses
    FOCUS_BRACKETING = "focus_bracketing" # Multiple focus positions per lens


@dataclass
class DualLensFrame:
    """Frame data from dual-lens system."""

    # Image data
    lens_a_frame: Optional[np.ndarray] = None
    lens_b_frame: Optional[np.ndarray] = None

    # Acquisition metadata
    timestamp: float = field(default_factory=time.time)
    active_lens: Optional[LensID] = None
    acquisition_mode: AcquisitionMode = AcquisitionMode.SINGLE_LENS

    # Focus positions
    lens_a_z_um: Optional[float] = None
    lens_b_z_um: Optional[float] = None

    # Illumination state
    illumination_pattern: str = "UNKNOWN"
    illumination_intensity: float = 0.0

    # Quality metrics
    lens_a_metric: Optional[float] = None
    lens_b_metric: Optional[float] = None

    def has_lens_a_data(self) -> bool:
        return self.lens_a_frame is not None

    def has_lens_b_data(self) -> bool:
        return self.lens_b_frame is not None

    def is_simultaneous(self) -> bool:
        return self.has_lens_a_data() and self.has_lens_b_data()


class DualLensCameraController:
    """Advanced camera controller for dual-lens autofocus system."""

    def __init__(self,
                 lens_a_profile: LensProfile,
                 lens_b_profile: LensProfile):
        """Initialize dual-lens camera controller.

        Args:
            lens_a_profile: Configuration for Lens-A (scanning)
            lens_b_profile: Configuration for Lens-B (detailed)
        """
        self.profiles = {
            LensID.LENS_A: lens_a_profile,
            LensID.LENS_B: lens_b_profile
        }

        # Current state
        self._active_lens = LensID.LENS_A
        self._focus_positions = {
            LensID.LENS_A: 0.0,
            LensID.LENS_B: 0.0
        }

        # Hardware simulation state
        self._lens_switch_time_ms = 50.0  # Time to switch between lenses
        self._focus_move_speed_um_per_ms = 0.2  # Focus movement speed

        # Synchronization
        self._lock = threading.Lock()
        self._last_switch_time = 0.0

        # Performance tracking
        self.acquisition_history: List[Dict[str, Any]] = []

        # Focus surface models (will be populated by calibration)
        self._focus_surfaces = {
            LensID.LENS_A: None,
            LensID.LENS_B: None
        }

    def set_active_lens(self, lens_id: LensID) -> None:
        """Switch to specified lens with realistic timing."""
        with self._lock:
            if self._active_lens != lens_id:
                # Simulate lens switching time
                current_time = time.time()
                if current_time - self._last_switch_time < (self._lens_switch_time_ms / 1000.0):
                    remaining_time = (self._lens_switch_time_ms / 1000.0) - (current_time - self._last_switch_time)
                    time.sleep(remaining_time)

                self._active_lens = lens_id
                self._last_switch_time = time.time()

    def get_active_lens(self) -> LensID:
        """Get currently active lens."""
        return self._active_lens

    def set_focus(self, z_um: float, lens_id: Optional[LensID] = None) -> None:
        """Set focus position for specified lens (or active lens)."""
        target_lens = lens_id or self._active_lens

        with self._lock:
            # Validate range
            profile = self.profiles[target_lens]
            if not profile.is_z_in_range(z_um):
                raise ValueError(f"Focus position {z_um:.2f} μm out of range for {target_lens.value}")

            # Simulate focus movement time
            current_pos = self._focus_positions[target_lens]
            move_distance = abs(z_um - current_pos)
            move_time_s = move_distance / (profile.focus_speed_um_per_s)

            if move_time_s > 0.001:  # Only sleep for significant moves
                time.sleep(min(move_time_s, 0.1))  # Cap at 100ms for simulation

            self._focus_positions[target_lens] = z_um

    def get_focus(self, lens_id: Optional[LensID] = None) -> float:
        """Get focus position of specified lens (or active lens)."""
        target_lens = lens_id or self._active_lens
        return self._focus_positions[target_lens]

    def get_frame(self) -> np.ndarray:
        """Capture frame from active lens."""
        with self._lock:
            frame = self._simulate_lens_frame(self._active_lens)

            # Track acquisition
            self.acquisition_history.append({
                "timestamp": time.time(),
                "lens": self._active_lens,
                "z_um": self._focus_positions[self._active_lens],
                "mode": AcquisitionMode.SINGLE_LENS
            })

            return frame

    def get_dual_frame(self,
                      lens_a_z_um: Optional[float] = None,
                      lens_b_z_um: Optional[float] = None,
                      mode: AcquisitionMode = AcquisitionMode.ALTERNATING) -> DualLensFrame:
        """Capture frames from both lenses with specified focus positions.

        Args:
            lens_a_z_um: Focus position for Lens-A (None = current)
            lens_b_z_um: Focus position for Lens-B (None = current)
            mode: Acquisition mode

        Returns:
            DualLensFrame with data from both lenses
        """
        result = DualLensFrame(
            acquisition_mode=mode,
            timestamp=time.time()
        )

        if mode == AcquisitionMode.SIMULTANEOUS:
            # Simultaneous capture (would require specialized hardware)
            with self._lock:
                if lens_a_z_um is not None:
                    self.set_focus(lens_a_z_um, LensID.LENS_A)
                if lens_b_z_um is not None:
                    self.set_focus(lens_b_z_um, LensID.LENS_B)

                # Simulate simultaneous capture
                result.lens_a_frame = self._simulate_lens_frame(LensID.LENS_A)
                result.lens_b_frame = self._simulate_lens_frame(LensID.LENS_B)
                result.lens_a_z_um = self._focus_positions[LensID.LENS_A]
                result.lens_b_z_um = self._focus_positions[LensID.LENS_B]

        elif mode == AcquisitionMode.ALTERNATING:
            # Rapid alternation between lenses

            # Capture from Lens-A
            self.set_active_lens(LensID.LENS_A)
            if lens_a_z_um is not None:
                self.set_focus(lens_a_z_um)
            result.lens_a_frame = self.get_frame()
            result.lens_a_z_um = self._focus_positions[LensID.LENS_A]

            # Capture from Lens-B
            self.set_active_lens(LensID.LENS_B)
            if lens_b_z_um is not None:
                self.set_focus(lens_b_z_um)
            result.lens_b_frame = self.get_frame()
            result.lens_b_z_um = self._focus_positions[LensID.LENS_B]

        else:
            raise ValueError(f"Unsupported acquisition mode: {mode}")

        return result

    def focus_bracketing_sequence(self,
                                lens_id: LensID,
                                center_z_um: float,
                                range_um: float,
                                num_steps: int = 5) -> List[Tuple[float, np.ndarray]]:
        """Capture focus bracketing sequence.

        Args:
            lens_id: Target lens
            center_z_um: Center focus position
            range_um: Total focus range
            num_steps: Number of focus steps

        Returns:
            List of (focus_position, frame) tuples
        """
        self.set_active_lens(lens_id)

        # Generate focus positions
        half_range = range_um / 2.0
        focus_positions = np.linspace(center_z_um - half_range,
                                    center_z_um + half_range,
                                    num_steps)

        # Validate positions
        profile = self.profiles[lens_id]
        focus_positions = [z for z in focus_positions if profile.is_z_in_range(z)]

        # Capture sequence
        sequence = []
        for z_um in focus_positions:
            self.set_focus(z_um)
            frame = self.get_frame()
            sequence.append((z_um, frame))

            # Track for performance analysis
            self.acquisition_history.append({
                "timestamp": time.time(),
                "lens": lens_id,
                "z_um": z_um,
                "mode": AcquisitionMode.FOCUS_BRACKETING
            })

        return sequence

    def _simulate_lens_frame(self, lens_id: LensID) -> np.ndarray:
        """Simulate frame capture from specified lens."""
        profile = self.profiles[lens_id]
        z_position = self._focus_positions[lens_id]

        # Simulate different field of view based on magnification
        if lens_id == LensID.LENS_A:
            # Lower magnification = larger FOV
            img_size = (600, 600)
            cell_size_range = (12, 20)  # Larger cells in image
        else:
            # Higher magnification = smaller FOV, more detail
            img_size = (400, 400)
            cell_size_range = (20, 35)  # Larger apparent cell size

        # Create base image
        img = np.random.randint(180, 220, (*img_size, 3), dtype=np.uint8)

        # Add blood cells with lens-appropriate detail
        num_cells = int(30 * (profile.field_of_view_um / 500.0))  # Scale with FOV

        for _ in range(num_cells):
            cx = np.random.randint(20, img_size[0] - 20)
            cy = np.random.randint(20, img_size[1] - 20)
            radius = np.random.randint(*cell_size_range)

            # Create cell mask
            y, x = np.ogrid[:img_size[0], :img_size[1]]
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2

            # RBC color
            img[mask] = [200, 160, 160]

            # Central pallor (more visible with higher magnification)
            pallor_factor = 0.4 if lens_id == LensID.LENS_B else 0.2
            center_mask = (x - cx)**2 + (y - cy)**2 <= (radius * pallor_factor)**2
            img[center_mask] = [180, 140, 140]

            # Add fine detail for high magnification
            if lens_id == LensID.LENS_B and np.random.random() < 0.3:
                # Add membrane details
                for angle in np.linspace(0, 2*np.pi, 8):
                    edge_x = int(cx + radius * 0.8 * np.cos(angle))
                    edge_y = int(cy + radius * 0.8 * np.sin(angle))
                    if 0 <= edge_x < img_size[0] and 0 <= edge_y < img_size[1]:
                        img[edge_y-1:edge_y+2, edge_x-1:edge_x+2] = [190, 150, 150]

        # Simulate focus quality based on distance from optimal
        # Each lens has a different optimal focus for the "specimen"
        optimal_z = profile.get_z_center() + np.random.uniform(-2, 2)  # Specimen variation
        focus_error = abs(z_position - optimal_z)
        focus_quality = np.exp(-(focus_error / profile.depth_of_field_um)**2)

        # Apply blur based on focus quality
        if focus_quality < 0.95:
            blur_amount = int((1.0 - focus_quality) * 3) + 1
            if blur_amount > 1:
                try:
                    import cv2
                    kernel_size = blur_amount * 2 + 1
                    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), blur_amount/2)
                except ImportError:
                    # Fallback: simple smoothing
                    from scipy import ndimage
                    img = ndimage.gaussian_filter(img, sigma=blur_amount/2)
                except ImportError:
                    # No blurring available
                    pass

        # Add lens-specific noise characteristics
        if lens_id == LensID.LENS_A:
            # Scanning lens: lower noise, optimized for speed
            noise_level = 3
        else:
            # Detail lens: slightly higher noise, optimized for resolution
            noise_level = 5

        noise = np.random.normal(0, noise_level, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img

    def get_acquisition_stats(self) -> Dict[str, Any]:
        """Get camera acquisition performance statistics."""
        if not self.acquisition_history:
            return {"status": "no_data"}

        recent_acquisitions = self.acquisition_history[-100:]  # Last 100 acquisitions

        # Count by lens
        lens_a_count = sum(1 for a in recent_acquisitions if a["lens"] == LensID.LENS_A)
        lens_b_count = sum(1 for a in recent_acquisitions if a["lens"] == LensID.LENS_B)

        # Calculate timing statistics
        if len(recent_acquisitions) > 1:
            timestamps = [a["timestamp"] for a in recent_acquisitions]
            intervals = np.diff(timestamps) * 1000  # Convert to milliseconds
            avg_interval_ms = np.mean(intervals)
            throughput_fps = 1000.0 / avg_interval_ms if avg_interval_ms > 0 else 0
        else:
            avg_interval_ms = 0
            throughput_fps = 0

        return {
            "total_acquisitions": len(recent_acquisitions),
            "lens_a_acquisitions": lens_a_count,
            "lens_b_acquisitions": lens_b_count,
            "avg_interval_ms": avg_interval_ms,
            "throughput_fps": throughput_fps,
            "active_lens": self._active_lens.value,
            "focus_positions": {
                "lens_a_um": self._focus_positions[LensID.LENS_A],
                "lens_b_um": self._focus_positions[LensID.LENS_B]
            }
        }

    def calibrate_focus_surface(self,
                              lens_id: LensID,
                              calibration_points: List[Tuple[float, float, float]]) -> Dict[str, Any]:
        """Calibrate focus surface model for specified lens.

        Args:
            lens_id: Target lens
            calibration_points: List of (x_um, y_um, z_optimal_um) points

        Returns:
            Calibration results and model parameters
        """
        if len(calibration_points) < 9:  # Need minimum points for surface fitting
            raise ValueError("Need at least 9 calibration points for surface fitting")

        # Extract coordinates
        points = np.array(calibration_points)
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        z_coords = points[:, 2]

        # Fit polynomial surface: z = a + bx + cy + dx² + exy + fy²
        design_matrix = np.column_stack([
            np.ones(len(points)),  # constant
            x_coords,              # x
            y_coords,              # y
            x_coords**2,           # x²
            x_coords * y_coords,   # xy
            y_coords**2            # y²
        ])

        # Solve for coefficients using least squares
        coefficients, residuals, rank, singular_values = np.linalg.lstsq(
            design_matrix, z_coords, rcond=None
        )

        # Calculate fit quality
        predicted_z = design_matrix @ coefficients
        rms_error = np.sqrt(np.mean((z_coords - predicted_z)**2))
        max_error = np.max(np.abs(z_coords - predicted_z))

        # Store surface model
        surface_model = {
            "coefficients": coefficients.tolist(),
            "rms_error_um": float(rms_error),
            "max_error_um": float(max_error),
            "num_points": len(points),
            "calibration_timestamp": time.time(),
            "x_range": [float(np.min(x_coords)), float(np.max(x_coords))],
            "y_range": [float(np.min(y_coords)), float(np.max(y_coords))],
            "z_range": [float(np.min(z_coords)), float(np.max(z_coords))]
        }

        self._focus_surfaces[lens_id] = surface_model

        return {
            "success": True,
            "lens_id": lens_id.value,
            "model": surface_model,
            "fit_quality": "excellent" if rms_error < 0.2 else "good" if rms_error < 0.5 else "fair"
        }

    def predict_focus_position(self, lens_id: LensID, x_um: float, y_um: float) -> Optional[float]:
        """Predict optimal focus position using calibrated surface model.

        Args:
            lens_id: Target lens
            x_um, y_um: Stage position

        Returns:
            Predicted focus position or None if no model available
        """
        surface_model = self._focus_surfaces.get(lens_id)
        if surface_model is None:
            return None

        coeffs = surface_model["coefficients"]

        # Apply polynomial surface model
        z_predicted = (coeffs[0] +                    # constant
                      coeffs[1] * x_um +             # x term
                      coeffs[2] * y_um +             # y term
                      coeffs[3] * x_um**2 +          # x² term
                      coeffs[4] * x_um * y_um +      # xy term
                      coeffs[5] * y_um**2)           # y² term

        return float(z_predicted)

    def get_focus_surface_info(self, lens_id: LensID) -> Optional[Dict[str, Any]]:
        """Get information about calibrated focus surface."""
        return self._focus_surfaces.get(lens_id)

    def clear_acquisition_history(self) -> None:
        """Clear acquisition history to free memory."""
        self.acquisition_history.clear()

    def close(self) -> None:
        """Clean up camera controller resources."""
        self.clear_acquisition_history()
        self._focus_surfaces.clear()