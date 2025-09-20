from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable


@runtime_checkable
class StageInterface(Protocol):
    """XY stage protocol for predictive focus surface building.

    Implementations should provide at least absolute XY moves and the ability to
    query current position in the same units as tile bounding boxes.
    """

    def move_xy(self, x: float, y: float) -> None:
        """Move to absolute XY coordinates (blocking until in-position)."""

    def get_xy(self) -> Tuple[float, float]:
        """Return current XY coordinates."""

