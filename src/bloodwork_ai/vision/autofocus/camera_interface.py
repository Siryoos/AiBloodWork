from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable
import numpy as np


@runtime_checkable
class CameraInterface(Protocol):
    """Typed interface for cameras supporting adjustable focus.

    Concrete implementations should connect to actual hardware. All methods
    must be non-blocking except where noted; small waits for mechanical settle
    may be necessary after `set_focus` in implementations.
    """

    def get_frame(self) -> np.ndarray:
        """Return the latest frame as a HxWxC uint8 BGR/RGB image."""

    def set_focus(self, value: float) -> None:
        """Set focus position within device's supported range."""

    def get_focus(self) -> float:
        """Return the current focus position."""

    def get_focus_range(self) -> Tuple[float, float]:
        """Return `(min_focus, max_focus)` supported by the device."""

