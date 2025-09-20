from __future__ import annotations

from dataclasses import dataclass

from .camera_interface import CameraInterface
from .strategies.contrast import ContrastMaximizationStrategy


class AutoFocusStrategy:
    """Minimal strategy protocol for type clarity."""

    def find_best_focus(self, *, timeout_s: float | None = None) -> float:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class AutoFocusController:
    """High-level controller that runs an auto-focus strategy.

    Example:
        controller = AutoFocusController(camera, ContrastMaximizationStrategy(camera))
        best_focus = controller.autofocus(timeout_s=2.0)
    """

    camera: CameraInterface
    strategy: AutoFocusStrategy | None = None

    def __post_init__(self) -> None:
        if self.strategy is None:
            # Sensible defaults that work for many devices; tune as needed.
            self.strategy = ContrastMaximizationStrategy(
                camera=self.camera,
                step=5.0,
                settle_time_s=0.02,
                min_step=0.5,
                max_iters=50,
            )

    def autofocus(self, *, timeout_s: float | None = None) -> float:
        if self.strategy is None:  # for mypy
            raise RuntimeError("AutoFocusController not initialized correctly")
        return float(self.strategy.find_best_focus(timeout_s=timeout_s))

