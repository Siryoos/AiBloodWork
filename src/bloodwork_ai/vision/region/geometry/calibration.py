"""Estimation utilities for per-lens geometry calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

from ..types import CalibrationPack, PixelCoordinate, StageCoordinate


@dataclass
class FiducialObservation:
    """Single fiducial correspondence between pixel and stage space."""

    lens_id: str
    pixel: PixelCoordinate
    stage: StageCoordinate


@dataclass
class GridObservation:
    """Grid-based calibration observation with world and image coordinates."""

    lens_id: str
    world_points: np.ndarray  # Nx3 world coordinates
    image_points: np.ndarray  # Nx2 image coordinates
    stage_pose: StageCoordinate  # Stage position during capture


@dataclass
class CalibrationMetrics:
    """Calibration quality metrics."""

    reprojection_error_um: Dict[str, float]
    cross_lens_residual_um: float
    stage_scale_residual_um: Tuple[float, float]


class CalibrationEstimator:
    """Estimate camera intrinsics/extrinsics and stage scale using OpenCV."""

    def __init__(
        self,
        stage_scale_prior_um_per_count: Optional[Tuple[float, float]] = None,
        max_reprojection_error_um: float = 2.0,
        grid_size: Tuple[int, int] = (9, 6),
        grid_spacing_um: float = 100.0,
    ):
        self.stage_scale_prior = stage_scale_prior_um_per_count
        self.max_reprojection_error = max_reprojection_error_um
        self.grid_size = grid_size
        self.grid_spacing = grid_spacing_um

    def estimate_from_grid(self, observations: List[GridObservation]) -> Tuple[CalibrationPack, CalibrationMetrics]:
        """Estimate calibration from fiducial grid observations."""
        per_lens_obs = self._group_observations_by_lens(observations)

        intrinsics: Dict[str, np.ndarray] = {}
        distortion: Dict[str, np.ndarray] = {}
        extrinsics: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        homographies: Dict[str, np.ndarray] = {}
        reprojection_errors: Dict[str, float] = {}

        for lens_id, lens_obs in per_lens_obs.items():
            K, dist, R_list, t_list, reproj_error = self._calibrate_single_lens(lens_obs)
            intrinsics[lens_id] = K
            distortion[lens_id] = dist

            R_avg, t_avg = self._average_extrinsics(R_list, t_list)
            extrinsics[lens_id] = (R_avg, t_avg)
            reprojection_errors[lens_id] = reproj_error

            homographies[lens_id] = self._compute_lens_homography(lens_obs)

        stage_scale, scale_residual = self._estimate_stage_scale(observations, intrinsics, distortion, extrinsics)

        cross_lens_transform = None
        cross_lens_residual = 0.0
        if len(per_lens_obs) >= 2:
            cross_lens_transform, cross_lens_residual = self._estimate_cross_lens_transform(
                observations, intrinsics, distortion, extrinsics
            )

        calibration = CalibrationPack(
            intrinsics=intrinsics,
            distortion=distortion,
            extrinsics=extrinsics,
            homographies=homographies,
            stage_scale_um_per_count=stage_scale,
            cross_lens_transform=cross_lens_transform,
        )

        metrics = CalibrationMetrics(
            reprojection_error_um=reprojection_errors,
            cross_lens_residual_um=cross_lens_residual,
            stage_scale_residual_um=scale_residual,
        )

        return calibration, metrics

    def _group_observations_by_lens(self, observations: List[GridObservation]) -> Dict[str, List[GridObservation]]:
        """Group observations by lens ID."""
        grouped: Dict[str, List[GridObservation]] = {}
        for obs in observations:
            grouped.setdefault(obs.lens_id, []).append(obs)
        return grouped

    def _calibrate_single_lens(self, observations: List[GridObservation]) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], float]:
        """Calibrate a single lens using OpenCV."""
        if len(observations) < 5:
            raise ValueError(f"Need at least 5 observations for calibration, got {len(observations)}")

        object_points = []
        image_points = []

        for obs in observations:
            object_points.append(obs.world_points.astype(np.float32))
            image_points.append(obs.image_points.astype(np.float32))

        image_size = (int(max(pt[:, 0].max() for pt in image_points) + 1),
                     int(max(pt[:, 1].max() for pt in image_points) + 1))

        calibration_flags = (
            cv2.CALIB_RATIONAL_MODEL |
            cv2.CALIB_THIN_PRISM_MODEL |
            cv2.CALIB_TILTED_MODEL
        )

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, image_size, None, None, flags=calibration_flags
        )

        if not ret:
            raise RuntimeError("Camera calibration failed")

        R_list = [cv2.Rodrigues(rvec)[0] for rvec in rvecs]
        t_list = [tvec.flatten() for tvec in tvecs]

        mean_error = 0.0
        for i in range(len(object_points)):
            projected, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], K, dist)
            projected = projected.reshape(-1, 2)
            error = cv2.norm(image_points[i], projected, cv2.NORM_L2) / len(projected)
            mean_error += error

        mean_error /= len(object_points)

        return K, dist, R_list, t_list, mean_error

    def _average_extrinsics(self, R_list: List[np.ndarray], t_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Average rotation matrices and translation vectors."""
        if not R_list:
            return np.eye(3), np.zeros(3)

        R_avg = np.mean(R_list, axis=0)
        U, _, Vt = np.linalg.svd(R_avg)
        R_avg = U @ Vt

        t_avg = np.mean(t_list, axis=0)

        return R_avg, t_avg

    def _compute_lens_homography(self, observations: List[GridObservation]) -> np.ndarray:
        """Compute planar homography mapping image pixels to stage coordinates."""
        image_points = []
        world_points = []

        for obs in observations:
            image_points.append(obs.image_points.astype(np.float64))
            world_points.append(obs.world_points[:, :2].astype(np.float64))

        if not image_points:
            raise ValueError("No observations available for homography computation")

        image_concat = np.vstack(image_points)
        world_concat = np.vstack(world_points)

        H, mask = cv2.findHomography(image_concat, world_concat, method=cv2.RANSAC)
        if H is None:
            raise RuntimeError("Failed to compute homography for lens")

        if mask is not None:
            valid = mask.flatten().astype(bool)
            if valid.sum() < 4:
                raise RuntimeError("Insufficient inliers for homography estimation")

        H /= H[2, 2]
        return H

    def _estimate_stage_scale(
        self,
        observations: List[GridObservation],
        intrinsics: Dict[str, np.ndarray],
        distortion: Dict[str, np.ndarray],
        extrinsics: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Estimate stage encoder scale using grid observations."""
        if self.stage_scale_prior:
            return self.stage_scale_prior, (0.0, 0.0)

        scale_estimates_x = []
        scale_estimates_y = []

        for obs in observations:
            if obs.lens_id not in intrinsics:
                continue

            K = intrinsics[obs.lens_id]
            dist = distortion[obs.lens_id]
            R, t = extrinsics[obs.lens_id]

            undistorted_points = cv2.undistortPoints(
                obs.image_points.reshape(-1, 1, 2), K, dist, P=K
            ).reshape(-1, 2)

            for i in range(len(undistorted_points) - 1):
                pixel_delta = undistorted_points[i + 1] - undistorted_points[i]
                world_delta = obs.world_points[i + 1] - obs.world_points[i]

                if abs(world_delta[0]) > 1e-6:
                    scale_estimates_x.append(abs(world_delta[0] / pixel_delta[0]))
                if abs(world_delta[1]) > 1e-6:
                    scale_estimates_y.append(abs(world_delta[1] / pixel_delta[1]))

        scale_x = np.median(scale_estimates_x) if scale_estimates_x else 1.0
        scale_y = np.median(scale_estimates_y) if scale_estimates_y else 1.0

        residual_x = np.median(np.abs(np.array(scale_estimates_x) - scale_x)) if scale_estimates_x else 0.0
        residual_y = np.median(np.abs(np.array(scale_estimates_y) - scale_y)) if scale_estimates_y else 0.0

        return (float(scale_x), float(scale_y)), (float(residual_x), float(residual_y))

    def _estimate_cross_lens_transform(
        self,
        observations: List[GridObservation],
        intrinsics: Dict[str, np.ndarray],
        distortion: Dict[str, np.ndarray],
        extrinsics: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[np.ndarray, float]:
        """Estimate 2D similarity transform between lens coordinate systems."""
        lens_ids = list(intrinsics.keys())
        if len(lens_ids) < 2:
            return np.eye(3), 0.0

        lens_a, lens_b = lens_ids[0], lens_ids[1]

        points_a = []
        points_b = []

        for obs in observations:
            if obs.lens_id == lens_a:
                K_a = intrinsics[lens_a]
                dist_a = distortion[lens_a]
                undistorted_a = cv2.undistortPoints(
                    obs.image_points.reshape(-1, 1, 2), K_a, dist_a, P=K_a
                ).reshape(-1, 2)
                points_a.extend(undistorted_a)

        for obs in observations:
            if obs.lens_id == lens_b:
                K_b = intrinsics[lens_b]
                dist_b = distortion[lens_b]
                undistorted_b = cv2.undistortPoints(
                    obs.image_points.reshape(-1, 1, 2), K_b, dist_b, P=K_b
                ).reshape(-1, 2)
                points_b.extend(undistorted_b)

        if len(points_a) < 4 or len(points_b) < 4:
            return np.eye(3), 0.0

        points_a = np.array(points_a[:min(len(points_a), len(points_b))])
        points_b = np.array(points_b[:min(len(points_a), len(points_b))])

        transform, mask = cv2.estimateAffinePartial2D(points_a, points_b, method=cv2.RANSAC)

        if transform is None:
            return np.eye(3), float('inf')

        transform_3x3 = np.eye(3)
        transform_3x3[:2, :] = transform

        residuals = []
        for i, inlier in enumerate(mask.flatten()):
            if inlier:
                p_a_hom = np.array([points_a[i][0], points_a[i][1], 1.0])
                p_b_pred = transform_3x3 @ p_a_hom
                residual = np.linalg.norm(p_b_pred[:2] - points_b[i])
                residuals.append(residual)

        mean_residual = np.mean(residuals) if residuals else 0.0

        return transform_3x3, float(mean_residual)

    def estimate_legacy(self, observations: Iterable[FiducialObservation]) -> CalibrationPack:
        """Legacy estimation method for backward compatibility."""
        per_lens_pixels: Dict[str, List[np.ndarray]] = {}
        per_lens_stage: Dict[str, List[np.ndarray]] = {}

        for obs in observations:
            per_lens_pixels.setdefault(obs.lens_id, []).append(
                np.array([obs.pixel.u, obs.pixel.v, 1.0], dtype=np.float64)
            )
            per_lens_stage.setdefault(obs.lens_id, []).append(
                np.array([obs.stage.x_um, obs.stage.y_um, 1.0], dtype=np.float64)
            )

        intrinsics: Dict[str, np.ndarray] = {}
        distortion: Dict[str, np.ndarray] = {}
        extrinsics: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        homographies: Dict[str, np.ndarray] = {}

        for lens_id, pixels in per_lens_pixels.items():
            stage_pts = per_lens_stage[lens_id]
            if len(pixels) < 4:
                raise ValueError(f"Need at least 4 correspondences for lens {lens_id}")

            H = self._direct_linear_transform(np.stack(pixels), np.stack(stage_pts))
            intrinsics[lens_id] = np.eye(3, dtype=np.float64)
            distortion[lens_id] = np.zeros(5, dtype=np.float64)
            R = np.eye(3, dtype=np.float64)
            R[:2, :2] = H[:2, :2]
            t = np.zeros(3, dtype=np.float64)
            t[:2] = H[:2, 2]
            extrinsics[lens_id] = (R, t)
            homographies[lens_id] = H

        scale = self.stage_scale_prior or (1.0, 1.0)

        return CalibrationPack(
            intrinsics=intrinsics,
            distortion=distortion,
            extrinsics=extrinsics,
            homographies=homographies,
            stage_scale_um_per_count=scale,
        )

    @staticmethod
    def _direct_linear_transform(pixels: np.ndarray, stage_pts: np.ndarray) -> np.ndarray:
        """Compute homography using DLT."""
        if pixels.shape != stage_pts.shape:
            raise ValueError("Pixel and stage arrays must have matching shapes")

        n = pixels.shape[0]
        A = []
        for i in range(n):
            x, y, _ = pixels[i]
            X, Y, _ = stage_pts[i]
            A.append([0, 0, 0, -x, -y, -1, Y * x, Y * y, Y])
            A.append([x, y, 1, 0, 0, 0, -X * x, -X * y, -X])
        A = np.asarray(A, dtype=np.float64)
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        H /= H[2, 2]
        return H


class FiducialGridGenerator:
    """Generate world coordinates for calibration grids."""

    def __init__(self, grid_size: Tuple[int, int], spacing_um: float):
        self.grid_size = grid_size
        self.spacing_um = spacing_um

    def generate_world_points(self) -> np.ndarray:
        """Generate 3D world coordinates for calibration grid."""
        cols, rows = self.grid_size
        points = []

        for row in range(rows):
            for col in range(cols):
                x = col * self.spacing_um
                y = row * self.spacing_um
                z = 0.0
                points.append([x, y, z])

        return np.array(points, dtype=np.float32)
