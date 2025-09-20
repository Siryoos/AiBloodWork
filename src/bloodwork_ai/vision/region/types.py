"""Shared dataclasses and type definitions for region detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PixelCoordinate:
    """Pixel coordinate expressed in the survey image frame."""

    u: float
    v: float
    lens_id: str


@dataclass
class StageCoordinate:
    """Stage coordinate expressed in microns."""

    x_um: float
    y_um: float
    z_um: Optional[float] = None
    lens_id: Optional[str] = None


@dataclass
class CalibrationPack:
    """Calibration payload containing per-lens geometry data."""

    intrinsics: Dict[str, np.ndarray]
    distortion: Dict[str, np.ndarray]
    extrinsics: Dict[str, Tuple[np.ndarray, np.ndarray]]  # R, t
    homographies: Dict[str, np.ndarray] = field(default_factory=dict)
    stage_scale_um_per_count: Tuple[float, float] = (1.0, 1.0)
    cross_lens_transform: Optional[np.ndarray] = None  # 3x3 similarity matrix
    metadata: Dict[str, str] = field(default_factory=dict)

    def get_intrinsics(self, lens_id: str) -> np.ndarray:
        return self.intrinsics[lens_id]

    def get_distortion(self, lens_id: str) -> np.ndarray:
        return self.distortion[lens_id]

    def get_extrinsics(self, lens_id: str) -> Tuple[np.ndarray, np.ndarray]:
        return self.extrinsics[lens_id]

    def get_homography(self, lens_id: str) -> np.ndarray:
        if lens_id not in self.homographies:
            raise KeyError(f"No homography stored for lens '{lens_id}'")
        return self.homographies[lens_id]


@dataclass
class RegionDetectionArtifacts:
    """Intermediate artifacts produced by the pipeline."""

    smear_mask: Optional[np.ndarray] = None
    smear_polygon: Optional[np.ndarray] = None
    feathered_edge: Optional[np.ndarray] = None
    quality_map: Optional[np.ndarray] = None
    survey_mosaic_path: Optional[Path] = None
    mask_pyramid_path: Optional[Path] = None


@dataclass
class RegionDetectionResult:
    """Final result returned by the detection service."""

    slide_id: str
    calibration: CalibrationPack
    mapper: "CoordinateMapper"
    artifacts: RegionDetectionArtifacts
    scan_plan: Optional[List[Dict[str, object]]] = None


class CoordinateMapper:
    """Pixel to stage coordinate transformer."""

    def to_stage(self, coord: PixelCoordinate) -> StageCoordinate:  # pragma: no cover - interface
        raise NotImplementedError

    def to_pixel(self, coord: StageCoordinate) -> PixelCoordinate:  # pragma: no cover - interface
        raise NotImplementedError

    def batch_to_stage(self, coords: Sequence[PixelCoordinate]) -> List[StageCoordinate]:
        return [self.to_stage(c) for c in coords]

    def batch_to_pixel(self, coords: Sequence[StageCoordinate]) -> List[PixelCoordinate]:
        return [self.to_pixel(c) for c in coords]


@dataclass
class SurveyFrame:
    """Survey acquisition frame and metadata."""

    image: np.ndarray
    stage_position: StageCoordinate
    exposure_ms: float
    timestamp: float


@dataclass
class SurveyStream:
    """Iterable container for survey frames."""

    frames: Iterable[SurveyFrame]
    lens_id: str
    slide_id: str
