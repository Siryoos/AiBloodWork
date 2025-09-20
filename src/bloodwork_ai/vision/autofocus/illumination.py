from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable, Dict, List, Optional, Sequence
import time
import numpy as np


@runtime_checkable
class IlluminationController(Protocol):
    """Interface for LED illumination control."""

    def set_led_intensity(self, channel: int, intensity: float) -> None:
        """Set LED intensity for a specific channel (0.0-1.0)."""

    def get_led_intensity(self, channel: int) -> float:
        """Get current LED intensity for a channel."""

    def set_all_leds(self, intensities: Sequence[float]) -> None:
        """Set all LED intensities at once."""

    def get_led_count(self) -> int:
        """Get number of LED channels available."""

    def enable_illumination(self, enabled: bool = True) -> None:
        """Enable or disable illumination."""


@dataclass
class IlluminationPattern:
    """Represents an illumination pattern for autofocus."""

    name: str
    intensities: List[float]  # Per-channel intensities (0.0-1.0)
    description: str = ""

    def apply(self, controller: IlluminationController) -> None:
        """Apply this pattern to the illumination controller."""
        controller.set_all_leds(self.intensities)


class IlluminationPatterns:
    """Standard illumination patterns for blood smear autofocus."""

    @staticmethod
    def uniform(intensity: float = 0.5, num_channels: int = 8) -> IlluminationPattern:
        """Uniform illumination from all LEDs."""
        return IlluminationPattern(
            name="uniform",
            intensities=[intensity] * num_channels,
            description="Uniform illumination from all channels"
        )

    @staticmethod
    def darkfield(intensity: float = 0.7, num_channels: int = 8) -> IlluminationPattern:
        """Dark field illumination - side illumination only."""
        pattern = [0.0] * num_channels
        # Assume channels 2,3,6,7 are side illumination
        for i in [2, 3, 6, 7]:
            if i < num_channels:
                pattern[i] = intensity
        return IlluminationPattern(
            name="darkfield",
            intensities=pattern,
            description="Dark field illumination"
        )

    @staticmethod
    def brightfield(intensity: float = 0.6, num_channels: int = 8) -> IlluminationPattern:
        """Bright field illumination - center illumination."""
        pattern = [0.0] * num_channels
        # Assume channels 0,1,4,5 are center illumination
        for i in [0, 1, 4, 5]:
            if i < num_channels:
                pattern[i] = intensity
        return IlluminationPattern(
            name="brightfield",
            intensities=pattern,
            description="Bright field illumination"
        )

    @staticmethod
    def oblique(intensity: float = 0.8, angle_idx: int = 0, num_channels: int = 8) -> IlluminationPattern:
        """Oblique illumination from one direction."""
        pattern = [0.0] * num_channels
        channel = angle_idx % num_channels
        pattern[channel] = intensity
        return IlluminationPattern(
            name=f"oblique_{angle_idx}",
            intensities=pattern,
            description=f"Oblique illumination from angle {angle_idx}"
        )


@dataclass
class IlluminationManager:
    """Manages illumination patterns for autofocus operations."""

    controller: IlluminationController
    settle_time_s: float = 0.02  # Time to wait after changing illumination
    current_pattern: Optional[IlluminationPattern] = None

    def __post_init__(self) -> None:
        # Initialize with uniform illumination
        self.set_pattern(IlluminationPatterns.uniform(num_channels=self.controller.get_led_count()))

    def set_pattern(self, pattern: IlluminationPattern) -> None:
        """Set illumination pattern and wait for settle."""
        pattern.apply(self.controller)
        self.current_pattern = pattern
        time.sleep(self.settle_time_s)

    def get_pattern(self) -> Optional[IlluminationPattern]:
        """Get current illumination pattern."""
        return self.current_pattern

    def set_uniform(self, intensity: float = 0.5) -> None:
        """Quick method to set uniform illumination."""
        pattern = IlluminationPatterns.uniform(intensity, self.controller.get_led_count())
        self.set_pattern(pattern)

    def enable(self, enabled: bool = True) -> None:
        """Enable or disable illumination."""
        self.controller.enable_illumination(enabled)

    def save_pattern(self, pattern: IlluminationPattern) -> None:
        """Store a custom pattern for later use."""
        if not hasattr(self, '_saved_patterns'):
            self._saved_patterns: Dict[str, IlluminationPattern] = {}
        self._saved_patterns[pattern.name] = pattern

    def load_pattern(self, name: str) -> Optional[IlluminationPattern]:
        """Load a previously saved pattern."""
        if not hasattr(self, '_saved_patterns'):
            return None
        return self._saved_patterns.get(name)

    def optimize_for_contrast(self,
                             camera_interface,
                             metric_fn,
                             test_patterns: Optional[List[IlluminationPattern]] = None) -> IlluminationPattern:
        """Find optimal illumination pattern for maximum contrast.

        Args:
            camera_interface: Camera to capture test images
            metric_fn: Focus metric function to optimize
            test_patterns: List of patterns to test, or None for defaults

        Returns:
            Best illumination pattern found
        """
        if test_patterns is None:
            num_channels = self.controller.get_led_count()
            test_patterns = [
                IlluminationPatterns.uniform(0.4, num_channels),
                IlluminationPatterns.uniform(0.6, num_channels),
                IlluminationPatterns.brightfield(0.6, num_channels),
                IlluminationPatterns.darkfield(0.7, num_channels),
                IlluminationPatterns.oblique(0.8, 0, num_channels),
                IlluminationPatterns.oblique(0.8, 2, num_channels),
            ]

        best_pattern = None
        best_score = -np.inf

        original_pattern = self.current_pattern

        for pattern in test_patterns:
            self.set_pattern(pattern)

            # Capture multiple frames and average metric
            scores = []
            for _ in range(3):
                frame = camera_interface.get_frame()
                score = metric_fn(frame)
                scores.append(score)
                time.sleep(0.01)

            avg_score = np.mean(scores)

            if avg_score > best_score:
                best_score = avg_score
                best_pattern = pattern

        # Restore original or set best
        if best_pattern is not None:
            self.set_pattern(best_pattern)
        elif original_pattern is not None:
            self.set_pattern(original_pattern)

        return best_pattern or (original_pattern or IlluminationPatterns.uniform())


@dataclass
class MockIlluminationController:
    """Mock illumination controller for testing."""

    num_channels: int = 8
    _intensities: List[float] = None
    _enabled: bool = True

    def __post_init__(self) -> None:
        if self._intensities is None:
            self._intensities = [0.0] * self.num_channels

    def set_led_intensity(self, channel: int, intensity: float) -> None:
        if 0 <= channel < self.num_channels:
            self._intensities[channel] = max(0.0, min(1.0, intensity))

    def get_led_intensity(self, channel: int) -> float:
        if 0 <= channel < self.num_channels:
            return self._intensities[channel]
        return 0.0

    def set_all_leds(self, intensities: Sequence[float]) -> None:
        for i, intensity in enumerate(intensities):
            if i < self.num_channels:
                self.set_led_intensity(i, intensity)

    def get_led_count(self) -> int:
        return self.num_channels

    def enable_illumination(self, enabled: bool = True) -> None:
        self._enabled = enabled