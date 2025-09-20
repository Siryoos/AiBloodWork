from __future__ import annotations

import time
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .dual_lens import LensID, LensProfile
from .dual_lens_camera import DualLensFrame, AcquisitionMode


@dataclass
class FastLensState:
    """Optimized lens state tracking."""

    lens_id: LensID
    focus_position_um: float
    last_update_time: float
    target_position_um: Optional[float] = None
    is_moving: bool = False
    move_start_time: float = 0.0


class OptimizedDualLensCameraController:
    """Ultra-fast camera controller with predictive operations and caching."""

    def __init__(self,
                 lens_a_profile: LensProfile,
                 lens_b_profile: LensProfile,
                 enable_predictive_focus: bool = True,
                 enable_concurrent_operations: bool = True):
        """Initialize optimized camera controller."""

        self.profiles = {
            LensID.LENS_A: lens_a_profile,
            LensID.LENS_B: lens_b_profile
        }

        # Optimization flags
        self.enable_predictive_focus = enable_predictive_focus
        self.enable_concurrent_operations = enable_concurrent_operations

        # Enhanced state tracking
        self.lens_states = {
            LensID.LENS_A: FastLensState(LensID.LENS_A, 0.0, time.time()),
            LensID.LENS_B: FastLensState(LensID.LENS_B, 0.0, time.time())
        }

        self._active_lens = LensID.LENS_A

        # Performance optimizations
        self._lens_switch_time_ms = 25.0  # Reduced from 50ms
        self._focus_move_speed_um_per_ms = 0.5  # Increased speed
        self._last_switch_time = 0.0

        # Concurrent operation support
        self.executor = ThreadPoolExecutor(max_workers=3)
        self._lock = threading.Lock()

        # Predictive focus caching
        self._focus_prediction_cache = {}
        self._thermal_compensation_cache = {}

        # Performance tracking
        self.operation_times = {
            'lens_switch': [],
            'focus_move': [],
            'frame_capture': []
        }

    def set_active_lens(self, lens_id: LensID) -> None:
        """Optimized lens switching with minimal delays."""
        with self._lock:
            if self._active_lens != lens_id:
                switch_start = time.time()

                # Check if sufficient time has passed since last switch
                time_since_switch = time.time() - self._last_switch_time
                min_switch_interval = self._lens_switch_time_ms / 1000.0

                if time_since_switch < min_switch_interval:
                    remaining_time = min_switch_interval - time_since_switch
                    time.sleep(remaining_time)

                self._active_lens = lens_id
                self._last_switch_time = time.time()

                # Reduced settling time for optimized operation
                optimized_settle = min(0.02, self._lens_switch_time_ms / 2000.0)
                time.sleep(optimized_settle)

                switch_time = (time.time() - switch_start) * 1000
                self.operation_times['lens_switch'].append(switch_time)

                # Keep only recent measurements
                if len(self.operation_times['lens_switch']) > 50:
                    self.operation_times['lens_switch'] = self.operation_times['lens_switch'][-25:]

    def get_active_lens(self) -> LensID:
        """Get currently active lens."""
        return self._active_lens

    def set_focus(self, z_um: float, lens_id: Optional[LensID] = None) -> None:
        """Optimized focus setting with predictive positioning."""
        target_lens = lens_id or self._active_lens

        with self._lock:
            # Validate range
            profile = self.profiles[target_lens]
            if not profile.is_z_in_range(z_um):
                z_um = np.clip(z_um, *profile.z_range_um)

            move_start = time.time()
            state = self.lens_states[target_lens]

            # Calculate move distance and time
            current_pos = state.focus_position_um
            move_distance = abs(z_um - current_pos)

            # Optimized movement calculation
            if move_distance > 0.01:  # Only move if significant
                # Use optimized focus speed
                move_time_s = move_distance / profile.focus_speed_um_per_s

                # Apply speed optimization factor
                speed_factor = 1.5  # 50% faster
                move_time_s /= speed_factor

                # Cap maximum move time for responsiveness
                move_time_s = min(move_time_s, 0.05)  # Max 50ms

                if move_time_s > 0.002:  # Only sleep for significant moves
                    time.sleep(move_time_s)

            # Update state
            state.focus_position_um = z_um
            state.last_update_time = time.time()
            state.is_moving = False

            move_time = (time.time() - move_start) * 1000
            self.operation_times['focus_move'].append(move_time)

            # Keep recent measurements
            if len(self.operation_times['focus_move']) > 50:
                self.operation_times['focus_move'] = self.operation_times['focus_move'][-25:]

    def get_focus(self, lens_id: Optional[LensID] = None) -> float:
        """Get focus position of specified lens."""
        target_lens = lens_id or self._active_lens
        return self.lens_states[target_lens].focus_position_um

    def set_focus_concurrent(self, z_a_um: Optional[float] = None, z_b_um: Optional[float] = None) -> None:
        """Set focus positions for both lenses concurrently (simulation)."""
        if not self.enable_concurrent_operations:
            # Fallback to sequential
            if z_a_um is not None:
                self.set_focus(z_a_um, LensID.LENS_A)
            if z_b_um is not None:
                self.set_focus(z_b_um, LensID.LENS_B)
            return

        # Simulate concurrent focus setting
        futures = []

        def set_focus_threaded(lens_id: LensID, z_um: float):
            state = self.lens_states[lens_id]
            profile = self.profiles[lens_id]

            if not profile.is_z_in_range(z_um):
                z_um = np.clip(z_um, *profile.z_range_um)

            move_distance = abs(z_um - state.focus_position_um)
            if move_distance > 0.01:
                move_time = move_distance / (profile.focus_speed_um_per_s * 1.5)  # 50% faster
                move_time = min(move_time, 0.05)
                if move_time > 0.002:
                    time.sleep(move_time)

            state.focus_position_um = z_um
            state.last_update_time = time.time()

        if z_a_um is not None:
            futures.append(self.executor.submit(set_focus_threaded, LensID.LENS_A, z_a_um))

        if z_b_um is not None:
            futures.append(self.executor.submit(set_focus_threaded, LensID.LENS_B, z_b_um))

        # Wait for completion
        for future in futures:
            future.result()

    def get_frame(self) -> np.ndarray:
        """Optimized frame capture with reduced overhead."""
        capture_start = time.time()

        with self._lock:
            frame = self._simulate_lens_frame_optimized(self._active_lens)

            capture_time = (time.time() - capture_start) * 1000
            self.operation_times['frame_capture'].append(capture_time)

            if len(self.operation_times['frame_capture']) > 50:
                self.operation_times['frame_capture'] = self.operation_times['frame_capture'][-25:]

            return frame

    def get_dual_frame_optimized(self,
                                lens_a_z_um: Optional[float] = None,
                                lens_b_z_um: Optional[float] = None,
                                mode: AcquisitionMode = AcquisitionMode.ALTERNATING) -> DualLensFrame:
        """Ultra-fast dual frame capture with optimization."""

        result = DualLensFrame(
            acquisition_mode=mode,
            timestamp=time.time()
        )

        if mode == AcquisitionMode.ALTERNATING:
            # Optimized alternating capture with minimal switching overhead

            # Pre-configure both lenses concurrently if possible
            if self.enable_concurrent_operations:
                self.set_focus_concurrent(lens_a_z_um, lens_b_z_um)

            # Capture from Lens-A
            if self._active_lens != LensID.LENS_A:
                self.set_active_lens(LensID.LENS_A)

            if lens_a_z_um is not None and not self.enable_concurrent_operations:
                self.set_focus(lens_a_z_um)

            result.lens_a_frame = self.get_frame()
            result.lens_a_z_um = self.get_focus(LensID.LENS_A)

            # Fast switch to Lens-B
            self.set_active_lens(LensID.LENS_B)

            if lens_b_z_um is not None and not self.enable_concurrent_operations:
                self.set_focus(lens_b_z_um)

            result.lens_b_frame = self.get_frame()
            result.lens_b_z_um = self.get_focus(LensID.LENS_B)

        elif mode == AcquisitionMode.SIMULTANEOUS:
            # Simulated simultaneous capture
            if lens_a_z_um is not None or lens_b_z_um is not None:
                self.set_focus_concurrent(lens_a_z_um, lens_b_z_um)

            # Simulate simultaneous capture with minimal delay
            result.lens_a_frame = self._simulate_lens_frame_optimized(LensID.LENS_A)
            result.lens_b_frame = self._simulate_lens_frame_optimized(LensID.LENS_B)
            result.lens_a_z_um = self.get_focus(LensID.LENS_A)
            result.lens_b_z_um = self.get_focus(LensID.LENS_B)

        return result

    def focus_bracketing_optimized(self,
                                  lens_id: LensID,
                                  center_z_um: float,
                                  range_um: float,
                                  num_steps: int = 5) -> List[Tuple[float, np.ndarray]]:
        """Optimized focus bracketing with minimal settling."""

        self.set_active_lens(lens_id)
        profile = self.profiles[lens_id]

        # Generate positions
        half_range = range_um / 2.0
        focus_positions = np.linspace(center_z_um - half_range,
                                    center_z_um + half_range,
                                    num_steps)

        # Filter valid positions
        focus_positions = [z for z in focus_positions if profile.is_z_in_range(z)]

        # Optimized capture sequence
        sequence = []

        for i, z_um in enumerate(focus_positions):
            self.set_focus(z_um)

            # Reduced settling for bracketing (since we're sampling)
            if i > 0:  # Skip settling for first position
                time.sleep(0.002)  # Minimal 2ms settling

            frame = self.get_frame()
            sequence.append((z_um, frame))

        return sequence

    def _simulate_lens_frame_optimized(self, lens_id: LensID) -> np.ndarray:
        """Optimized frame simulation with reduced computation."""
        profile = self.profiles[lens_id]
        z_position = self.lens_states[lens_id].focus_position_um

        # Use smaller image sizes for speed
        if lens_id == LensID.LENS_A:
            img_size = (400, 400)  # Reduced from 600x600
            cell_size_range = (8, 16)  # Smaller cells
        else:
            img_size = (300, 300)  # Reduced from 400x400
            cell_size_range = (15, 25)

        # Create base image with less noise computation
        img = np.random.randint(180, 220, (*img_size, 3), dtype=np.uint8)

        # Reduced number of cells for speed
        num_cells = int(15 * (profile.field_of_view_um / 500.0))  # Reduced from 30

        # Optimized cell generation
        for _ in range(num_cells):
            cx = np.random.randint(15, img_size[0] - 15)
            cy = np.random.randint(15, img_size[1] - 15)
            radius = np.random.randint(*cell_size_range)

            # Simplified cell rendering
            y, x = np.ogrid[:img_size[0], :img_size[1]]
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            img[mask] = [200, 160, 160]

            # Simple central pallor
            if np.random.random() < 0.7:  # 70% chance
                center_radius = max(1, radius // 3)
                center_mask = (x - cx)**2 + (y - cy)**2 <= center_radius**2
                img[center_mask] = [180, 140, 140]

        # Simplified focus simulation
        optimal_z = profile.get_z_center()
        focus_error = abs(z_position - optimal_z)
        focus_quality = np.exp(-(focus_error / profile.depth_of_field_um)**2)

        # Minimal blur simulation for speed
        if focus_quality < 0.9:
            blur_amount = int((1.0 - focus_quality) * 2) + 1
            if blur_amount > 1:
                # Very simple blur approximation
                kernel = np.ones((blur_amount, blur_amount)) / (blur_amount * blur_amount)
                for c in range(3):
                    img[:, :, c] = np.convolve(img[:, :, c].flatten(), kernel.flatten(), mode='same').reshape(img_size)

        # Minimal noise
        noise_level = 2  # Reduced noise
        noise = np.random.normal(0, noise_level, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img

    def predict_focus_with_surface(self, lens_id: LensID, x_um: float, y_um: float) -> Optional[float]:
        """Fast focus prediction using cached surface models."""
        cache_key = (lens_id, round(x_um / 100) * 100, round(y_um / 100) * 100)  # 100μm resolution cache

        if cache_key in self._focus_prediction_cache:
            return self._focus_prediction_cache[cache_key]

        # Simplified surface model prediction
        # In real implementation, this would use actual calibrated surface models
        profile = self.profiles[lens_id]

        # Simulate surface with slight tilt and curvature
        z_base = profile.get_z_center()
        tilt_x = 0.0001 * x_um
        tilt_y = 0.0001 * y_um
        curvature = 0.000001 * (x_um**2 + y_um**2)

        predicted_z = z_base + tilt_x + tilt_y + curvature
        predicted_z = np.clip(predicted_z, *profile.z_range_um)

        # Cache result
        self._focus_prediction_cache[cache_key] = predicted_z

        # Limit cache size
        if len(self._focus_prediction_cache) > 1000:
            # Remove oldest entries
            old_keys = list(self._focus_prediction_cache.keys())[:500]
            for key in old_keys:
                del self._focus_prediction_cache[key]

        return predicted_z

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics."""
        stats = {
            "optimization_enabled": {
                "predictive_focus": self.enable_predictive_focus,
                "concurrent_operations": self.enable_concurrent_operations
            },
            "timing_stats": {},
            "cache_stats": {
                "focus_prediction_cache_size": len(self._focus_prediction_cache),
                "thermal_cache_size": len(self._thermal_compensation_cache)
            },
            "lens_states": {}
        }

        # Timing statistics
        for operation, times in self.operation_times.items():
            if times:
                stats["timing_stats"][operation] = {
                    "avg_ms": np.mean(times),
                    "min_ms": np.min(times),
                    "max_ms": np.max(times),
                    "p95_ms": np.percentile(times, 95),
                    "count": len(times)
                }

        # Lens state information
        for lens_id, state in self.lens_states.items():
            stats["lens_states"][lens_id.value] = {
                "current_focus_um": state.focus_position_um,
                "last_update": state.last_update_time,
                "is_moving": state.is_moving
            }

        return stats

    def clear_caches(self) -> None:
        """Clear performance caches."""
        self._focus_prediction_cache.clear()
        self._thermal_compensation_cache.clear()

    def close(self) -> None:
        """Clean up optimized camera controller."""
        self.executor.shutdown(wait=True)
        self.clear_caches()
        for op_list in self.operation_times.values():
            op_list.clear()