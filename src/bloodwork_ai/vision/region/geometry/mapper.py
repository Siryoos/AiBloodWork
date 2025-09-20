"""Coordinate mapper implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from ..types import CalibrationPack, CoordinateMapper, PixelCoordinate, StageCoordinate


@dataclass
class CalibratedCoordinateMapper(CoordinateMapper):
    """Full calibration-based mapper using intrinsics/extrinsics and distortion."""

    calibration: CalibrationPack
    default_lens: Optional[str] = None

    def to_stage(self, coord: PixelCoordinate) -> StageCoordinate:
        lens_id = coord.lens_id or self.default_lens
        if lens_id is None:
            raise ValueError("Lens ID required for mapping")

        K = self.calibration.get_intrinsics(lens_id)
        dist = self.calibration.get_distortion(lens_id)
        R, t = self.calibration.get_extrinsics(lens_id)

        points = np.array([[coord.u, coord.v]], dtype=np.float32)
        undistorted = cv2.undistortPoints(points, K, dist, P=K).reshape(-1, 2)

        camera_ray = np.array([
            (undistorted[0, 0] - K[0, 2]) / K[0, 0],
            (undistorted[0, 1] - K[1, 2]) / K[1, 1],
            1.0
        ])

        stage_point = self._intersect_with_stage_plane(camera_ray, R, t)

        scale_x, scale_y = self.calibration.stage_scale_um_per_count
        stage_x = stage_point[0] * scale_x
        stage_y = stage_point[1] * scale_y

        return StageCoordinate(x_um=float(stage_x), y_um=float(stage_y), lens_id=lens_id)

    def to_pixel(self, coord: StageCoordinate) -> PixelCoordinate:
        lens_id = coord.lens_id or self.default_lens
        if lens_id is None:
            raise ValueError("Lens ID required for inverse mapping")

        K = self.calibration.get_intrinsics(lens_id)
        dist = self.calibration.get_distortion(lens_id)
        R, t = self.calibration.get_extrinsics(lens_id)

        scale_x, scale_y = self.calibration.stage_scale_um_per_count
        stage_point = np.array([
            coord.x_um / scale_x,
            coord.y_um / scale_y,
            0.0
        ])

        camera_point = R @ stage_point + t

        rvec = cv2.Rodrigues(R)[0]
        tvec = t.reshape(3, 1)

        projected, _ = cv2.projectPoints(
            stage_point.reshape(1, 1, 3), rvec, tvec, K, dist
        )

        u, v = projected[0, 0]
        return PixelCoordinate(u=float(u), v=float(v), lens_id=lens_id)

    def _intersect_with_stage_plane(self, ray: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Intersect camera ray with stage plane (Z=0)."""
        ray_world = R.T @ ray - R.T @ t

        if abs(ray_world[2]) < 1e-8:
            raise ValueError("Ray is parallel to stage plane")

        scale = -ray_world[2] / ray_world[2]
        intersection = ray_world * scale

        return intersection[:2]

    def transform_between_lenses(self, coord: PixelCoordinate, target_lens_id: str) -> PixelCoordinate:
        """Transform pixel coordinate between different lenses using cross-lens transform."""
        if coord.lens_id == target_lens_id:
            return coord

        if self.calibration.cross_lens_transform is None:
            raise ValueError("Cross-lens transform not available")

        stage_coord = self.to_stage(coord)
        target_coord = StageCoordinate(
            x_um=stage_coord.x_um,
            y_um=stage_coord.y_um,
            lens_id=target_lens_id
        )

        return self.to_pixel(target_coord)


@dataclass
class HomographyCoordinateMapper(CoordinateMapper):
    """Legacy homography-based mapper for pixel↔stage conversions."""

    homographies: Dict[str, np.ndarray]
    inverse_homographies: Dict[str, np.ndarray]
    default_lens: Optional[str] = None

    def to_stage(self, coord: PixelCoordinate) -> StageCoordinate:
        H = self._select_homography(coord.lens_id)
        vec = np.array([coord.u, coord.v, 1.0], dtype=np.float64)
        mapped = H @ vec
        mapped /= mapped[2]
        return StageCoordinate(x_um=float(mapped[0]), y_um=float(mapped[1]), lens_id=coord.lens_id)

    def to_pixel(self, coord: StageCoordinate) -> PixelCoordinate:
        lens_id = coord.lens_id or self.default_lens
        if lens_id is None:
            raise ValueError("Lens ID required for inverse mapping")
        H_inv = self._select_inverse_homography(lens_id)
        vec = np.array([coord.x_um, coord.y_um, 1.0], dtype=np.float64)
        mapped = H_inv @ vec
        mapped /= mapped[2]
        return PixelCoordinate(u=float(mapped[0]), v=float(mapped[1]), lens_id=lens_id)

    def _select_homography(self, lens_id: Optional[str]) -> np.ndarray:
        if lens_id is None:
            if self.default_lens is None:
                raise ValueError("Lens ID not provided and no default set")
            lens_id = self.default_lens
        try:
            return self.homographies[lens_id]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unknown lens id: {lens_id}") from exc

    def _select_inverse_homography(self, lens_id: str) -> np.ndarray:
        try:
            return self.inverse_homographies[lens_id]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unknown lens id: {lens_id}") from exc

    @classmethod
    def from_homographies(cls, homographies: Dict[str, np.ndarray], default_lens: Optional[str] = None) -> "HomographyCoordinateMapper":
        inv = {lid: np.linalg.inv(H) for lid, H in homographies.items()}
        return cls(homographies=homographies, inverse_homographies=inv, default_lens=default_lens)

    @classmethod
    def from_calibration(
        cls,
        calibration: "CalibrationPack",
        default_lens: Optional[str] = None,
    ) -> "HomographyCoordinateMapper":
        if not calibration.homographies:
            raise ValueError("Calibration pack does not contain homographies")
        return cls.from_homographies(calibration.homographies, default_lens)
