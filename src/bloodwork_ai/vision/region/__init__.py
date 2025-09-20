"""Blood smear region detection and coordinate mapping module.

Provides calibration utilities, smear detection services, scan planning, and
coordinate mapping helpers used by the hematology autofocus stack.
"""

from .config import RegionModuleConfig
from .geometry import CalibratedCoordinateMapper, CalibrationEstimator, HomographyCoordinateMapper
from .service import RegionDetectionService
from .types import (
    CalibrationPack,
    CoordinateMapper,
    PixelCoordinate,
    RegionDetectionArtifacts,
    RegionDetectionResult,
    StageCoordinate,
)

__all__ = [
    "RegionModuleConfig",
    "RegionDetectionService",
    "RegionDetectionResult",
    "RegionDetectionArtifacts",
    "CoordinateMapper",
    "CalibrationPack",
    "PixelCoordinate",
    "StageCoordinate",
    "CalibrationEstimator",
    "CalibratedCoordinateMapper",
    "HomographyCoordinateMapper",
]
