"""Feathered edge analysis and quality mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage, optimize
from skimage import filters, measure, morphology, segmentation

from .config import EdgeAnalysisConfig


@dataclass
class EdgeAnalyzer:
    """Compute smear axis, feathered edge, and comprehensive quality map."""

    config: EdgeAnalysisConfig

    def analyze(self, mask: np.ndarray, image: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Analyze mask to extract feathered edge and quality metrics."""
        if mask.sum() == 0:
            empty_edge = np.zeros((0, 2), dtype=np.float32)
            return empty_edge, np.zeros_like(mask, dtype=np.float32)

        # 1. Compute principal axis
        principal_axis_vector, centroid = self._compute_principal_axis(mask)

        # 2. Find feathered edge along the axis
        feathered_edge = self._find_feathered_edge(mask, principal_axis_vector, centroid)

        # 3. Compute comprehensive quality map
        quality_map = self._compute_quality_map(mask, image, principal_axis_vector)

        return feathered_edge, quality_map

    def _compute_principal_axis(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute principal axis using PCA on mask pixels."""
        coords = np.column_stack(np.nonzero(mask > 0))

        if coords.size == 0:
            return np.array([1.0, 0.0]), np.array([0.0, 0.0])

        # Compute centroid
        centroid = coords.mean(axis=0)

        # Center coordinates
        centered = coords - centroid

        # Compute covariance matrix
        cov_matrix = np.cov(centered, rowvar=False)

        # Eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)

        # Principal axis is the eigenvector with largest eigenvalue
        principal_idx = np.argmax(eigenvals)
        principal_axis = eigenvecs[:, principal_idx]

        # Ensure consistent orientation
        if principal_axis[1] < 0:
            principal_axis = -principal_axis

        return principal_axis, centroid

    def _find_feathered_edge(self, mask: np.ndarray, axis_vector: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        """Find feathered edge by detecting maximum gradient region along axis."""
        h, w = mask.shape
        feathered_points = []

        # Create line samples perpendicular to axis
        perpendicular = np.array([-axis_vector[1], axis_vector[0]])

        # Sample along the principal axis
        axis_length = np.sqrt(mask.sum()) * 0.5
        num_samples = max(50, int(axis_length / 10))

        for i in range(num_samples):
            # Position along axis
            t = (i / (num_samples - 1) - 0.5) * 2 * axis_length
            center_point = centroid + t * axis_vector

            # Check if center point is within image bounds
            if (center_point[0] < 0 or center_point[0] >= h or
                center_point[1] < 0 or center_point[1] >= w):
                continue

            # Sample perpendicular to axis
            edge_point = self._find_edge_along_line(mask, center_point, perpendicular)

            if edge_point is not None:
                feathered_points.append(edge_point)

        if len(feathered_points) < 2:
            # Fallback: return boundary points
            boundary = self._extract_boundary_points(mask)
            if len(boundary) > 0:
                return boundary[:min(len(boundary), 10)]
            else:
                return np.zeros((0, 2), dtype=np.float32)

        return np.array(feathered_points, dtype=np.float32)

    def _find_edge_along_line(self, mask: np.ndarray, center: np.ndarray, direction: np.ndarray) -> Optional[np.ndarray]:
        """Find edge point along a line by maximum gradient detection."""
        h, w = mask.shape

        # Sample distance
        max_distance = min(h, w) * 0.3
        num_samples = int(max_distance * 2)

        if num_samples < 5:
            return None

        # Sample points along line
        sample_points = []
        sample_values = []

        for i in range(num_samples):
            t = (i / (num_samples - 1) - 0.5) * 2 * max_distance
            point = center + t * direction

            # Check bounds
            y, x = int(round(point[0])), int(round(point[1]))
            if 0 <= y < h and 0 <= x < w:
                sample_points.append(point)
                sample_values.append(mask[y, x])

        if len(sample_values) < 5:
            return None

        sample_values = np.array(sample_values)

        # Find maximum gradient (edge)
        gradient = np.abs(np.gradient(sample_values))

        if gradient.max() < 0.1:  # No significant edge
            return None

        edge_idx = np.argmax(gradient)
        edge_point = sample_points[edge_idx]

        return edge_point

    def _extract_boundary_points(self, mask: np.ndarray) -> np.ndarray:
        """Extract boundary points using contour detection."""
        mask_uint = (mask > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return np.zeros((0, 2), dtype=np.float32)

        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Simplify contour
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        simplified = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Convert to (y, x) format
        points = simplified.reshape(-1, 2)
        points_yx = points[:, [1, 0]]  # Convert from (x, y) to (y, x)

        return points_yx.astype(np.float32)

    def _compute_quality_map(self, mask: np.ndarray, image: Optional[np.ndarray], axis_vector: np.ndarray) -> np.ndarray:
        """Compute comprehensive quality map including monolayer probability."""
        quality_map = np.zeros_like(mask, dtype=np.float32)

        if mask.sum() == 0:
            return quality_map

        # Base quality from distance transform (thickness proxy)
        mask_uint = (mask > 0.5).astype(np.uint8)
        distance_transform = cv2.distanceTransform(mask_uint, cv2.DIST_L2, 3)

        if distance_transform.max() > 0:
            thickness_quality = distance_transform / distance_transform.max()
        else:
            thickness_quality = np.zeros_like(mask, dtype=np.float32)

        # Monolayer quality assessment
        if image is not None:
            monolayer_quality = self._assess_monolayer_quality(image, mask)
        else:
            monolayer_quality = thickness_quality

        # Density quality (based on local variance)
        density_quality = self._compute_density_quality(mask)

        # Edge distance quality (prefer areas away from edges)
        edge_quality = self._compute_edge_distance_quality(mask)

        # Combine quality metrics
        weights = [0.3, 0.4, 0.2, 0.1]  # thickness, monolayer, density, edge
        quality_map = (
            weights[0] * thickness_quality +
            weights[1] * monolayer_quality +
            weights[2] * density_quality +
            weights[3] * edge_quality
        )

        # Apply mask constraint
        quality_map *= mask

        # Normalize to [0, 1]
        if quality_map.max() > 0:
            quality_map = quality_map / quality_map.max()

        return quality_map

    def _assess_monolayer_quality(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Assess monolayer quality based on cell density and size distribution."""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        gray = gray.astype(np.float32) / 255.0
        quality = np.zeros_like(mask, dtype=np.float32)

        # Apply mask
        masked_gray = gray * mask

        # Local standard deviation (proxy for texture/cell density)
        kernel_size = 15
        local_mean = cv2.blur(masked_gray, (kernel_size, kernel_size))
        local_var = cv2.blur(masked_gray**2, (kernel_size, kernel_size)) - local_mean**2
        local_std = np.sqrt(np.maximum(local_var, 0))

        # Ideal monolayer has moderate texture
        ideal_std = 0.1
        texture_quality = np.exp(-((local_std - ideal_std) ** 2) / (2 * (ideal_std * 0.5) ** 2))

        # Local contrast quality
        contrast = self._compute_local_contrast(masked_gray, kernel_size=9)

        # Ideal contrast range for monolayer
        contrast_quality = np.where(
            (contrast > 0.05) & (contrast < 0.3),
            1.0,
            np.exp(-((contrast - 0.15) ** 2) / (2 * 0.1 ** 2))
        )

        # Combine texture and contrast
        quality = 0.6 * texture_quality + 0.4 * contrast_quality

        return quality * mask

    def _compute_local_contrast(self, image: np.ndarray, kernel_size: int) -> np.ndarray:
        """Compute local contrast using local min/max."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        local_min = cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel)
        local_max = cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel)

        contrast = (local_max - local_min) / (local_max + local_min + 1e-8)

        return contrast

    def _compute_density_quality(self, mask: np.ndarray) -> np.ndarray:
        """Compute density quality based on local mask density."""
        kernel_size = 21
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)

        # Local density
        local_density = cv2.filter2D(mask.astype(np.float32), -1, kernel)

        # Ideal density is moderate (not too sparse, not too dense)
        ideal_density = 0.7
        density_quality = np.exp(-((local_density - ideal_density) ** 2) / (2 * (0.2 ** 2)))

        return density_quality * mask

    def _compute_edge_distance_quality(self, mask: np.ndarray) -> np.ndarray:
        """Compute quality based on distance from edges (prefer interior regions)."""
        mask_uint = (mask > 0.5).astype(np.uint8)
        distance = cv2.distanceTransform(mask_uint, cv2.DIST_L2, 3)

        # Normalize and apply sigmoid to prefer interior
        if distance.max() > 0:
            normalized_distance = distance / distance.max()
            edge_quality = 1 / (1 + np.exp(-10 * (normalized_distance - 0.3)))
        else:
            edge_quality = np.zeros_like(mask, dtype=np.float32)

        return edge_quality * mask

    def compute_axis_profile(self, mask: np.ndarray, image: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute density profile along the principal axis."""
        if mask.sum() == 0:
            return np.array([]), np.array([])

        axis_vector, centroid = self._compute_principal_axis(mask)

        # Sample along axis
        axis_length = np.sqrt(mask.sum()) * 0.8
        num_samples = max(50, int(axis_length / 5))

        positions = []
        densities = []

        for i in range(num_samples):
            t = (i / (num_samples - 1) - 0.5) * 2 * axis_length
            point = centroid + t * axis_vector

            y, x = int(round(point[0])), int(round(point[1]))

            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                positions.append(t)

                # Sample local density
                local_density = self._sample_local_density(mask, point, radius=10)
                densities.append(local_density)

        return np.array(positions), np.array(densities)

    def _sample_local_density(self, mask: np.ndarray, center: np.ndarray, radius: int) -> float:
        """Sample local density around a point."""
        y, x = int(round(center[0])), int(round(center[1]))
        h, w = mask.shape

        y_min = max(0, y - radius)
        y_max = min(h, y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(w, x + radius + 1)

        if y_min >= y_max or x_min >= x_max:
            return 0.0

        local_region = mask[y_min:y_max, x_min:x_max]

        return float(local_region.mean())

    def detect_feathered_edge_gradient(self, density_profile: np.ndarray, positions: np.ndarray) -> Optional[float]:
        """Detect feathered edge position by finding maximum gradient."""
        if len(density_profile) < 5:
            return None

        # Smooth profile
        smoothed = ndimage.gaussian_filter1d(density_profile, sigma=2)

        # Compute gradient
        gradient = np.gradient(smoothed)

        # Find position of maximum negative gradient (density drop)
        max_grad_idx = np.argmin(gradient)

        # Verify it's a significant gradient
        if abs(gradient[max_grad_idx]) < 0.05:
            return None

        return positions[max_grad_idx]


class AxisEstimator:
    """Specialized class for robust axis estimation."""

    @staticmethod
    def fit_ellipse_axis(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Fit ellipse to mask and extract major axis."""
        coords = np.column_stack(np.nonzero(mask > 0))

        if len(coords) < 5:
            centroid = np.array([0.0, 0.0])
            axis = np.array([1.0, 0.0])
            return axis, centroid, 0.0

        # Fit ellipse using least squares
        try:
            ellipse = cv2.fitEllipse(coords.astype(np.float32)[:, [1, 0]])  # Convert to (x,y)
            center, (width, height), angle = ellipse

            # Convert back to (y, x) and compute axis
            centroid = np.array([center[1], center[0]])

            # Major axis direction
            angle_rad = np.radians(angle)
            axis = np.array([np.sin(angle_rad), np.cos(angle_rad)])

            # Aspect ratio confidence
            aspect_ratio = max(width, height) / (min(width, height) + 1e-8)
            confidence = min(1.0, aspect_ratio / 3.0)

            return axis, centroid, confidence

        except cv2.error:
            # Fallback to PCA
            centroid = coords.mean(axis=0)
            centered = coords - centroid
            cov = np.cov(centered, rowvar=False)
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            axis = eigenvecs[:, np.argmax(eigenvals)]

            confidence = eigenvals.max() / (eigenvals.sum() + 1e-8)

            return axis, centroid, confidence

    @staticmethod
    def refine_axis_with_moments(mask: np.ndarray, initial_axis: np.ndarray) -> np.ndarray:
        """Refine axis estimation using image moments."""
        moments = cv2.moments((mask > 0.5).astype(np.uint8))

        if moments['m00'] == 0:
            return initial_axis

        # Central moments
        mu20 = moments['mu20'] / moments['m00']
        mu02 = moments['mu02'] / moments['m00']
        mu11 = moments['mu11'] / moments['m00']

        # Orientation angle from moments
        angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)

        refined_axis = np.array([np.sin(angle), np.cos(angle)])

        return refined_axis
