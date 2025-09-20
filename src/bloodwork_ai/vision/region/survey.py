"""Survey acquisition orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np

from .config import SurveyCaptureConfig
from .types import StageCoordinate, SurveyFrame, SurveyStream


class CameraInterface(Protocol):
    """Minimal camera interface for survey acquisition."""

    def capture_roi(self, *, exposure_ms: float) -> np.ndarray:  # pragma: no cover - interface
        ...


class StageInterface(Protocol):
    """Stage control interface."""

    def move_xy(self, x_um: float, y_um: float) -> None:  # pragma: no cover - interface
        ...

    def wait_until_positioned(self) -> None:  # pragma: no cover - interface
        ...

    def current_position(self) -> StageCoordinate:  # pragma: no cover - interface
        ...


@dataclass
class SurveyAcquisitionService:
    """Capture low-magnification survey data for smear detection."""

    config: SurveyCaptureConfig
    camera: CameraInterface
    stage: StageInterface

    def acquire(self, tile_positions: Iterable[StageCoordinate], exposure_ms: float) -> SurveyStream:
        frames: list[SurveyFrame] = []
        for position in tile_positions:
            self.stage.move_xy(position.x_um, position.y_um)
            self.stage.wait_until_positioned()
            image = self.camera.capture_roi(exposure_ms=exposure_ms)
            frames.append(
                SurveyFrame(
                    image=image,
                    stage_position=position,
                    exposure_ms=exposure_ms,
                    timestamp=0.0,
                )
            )
        return SurveyStream(frames=frames, lens_id=self.config.lens_id, slide_id="unknown")
