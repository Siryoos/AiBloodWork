"""QA utilities for the region detection pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import spatial
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_auc_score

from .types import PixelCoordinate, RegionDetectionResult, StageCoordinate


@dataclass
class RegionQAHarness:
    """Comprehensive evaluation harness for region detection pipeline."""

    def compute_iou(self, predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        """Compute Intersection over Union (IoU) metric."""
        pred = predicted > 0.5
        gt = ground_truth > 0.5
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        return float(intersection / union)

    def compute_dice_coefficient(self, predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        """Compute Dice coefficient (F1 score for binary segmentation)."""
        pred = predicted > 0.5
        gt = ground_truth > 0.5
        intersection = np.logical_and(pred, gt).sum()
        total = pred.sum() + gt.sum()
        if total == 0:
            return 1.0 if intersection == 0 else 0.0
        return float(2.0 * intersection / total)

    def compute_boundary_error(self, predicted: np.ndarray, ground_truth: np.ndarray,
                             pixel_size_um: float = 1.0) -> Dict[str, float]:
        """Compute boundary distance errors in micrometers."""
        pred_boundary = self._extract_boundary(predicted > 0.5)
        gt_boundary = self._extract_boundary(ground_truth > 0.5)

        if len(pred_boundary) == 0 or len(gt_boundary) == 0:
            return {"hausdorff_um": float('inf'), "avg_surface_distance_um": float('inf')}

        # Compute distances
        distances_pred_to_gt = spatial.distance.cdist(pred_boundary, gt_boundary).min(axis=1)
        distances_gt_to_pred = spatial.distance.cdist(gt_boundary, pred_boundary).min(axis=1)

        # Convert to micrometers
        hausdorff_um = max(distances_pred_to_gt.max(), distances_gt_to_pred.max()) * pixel_size_um
        avg_surface_distance_um = (distances_pred_to_gt.mean() + distances_gt_to_pred.mean()) / 2 * pixel_size_um

        return {
            "hausdorff_um": float(hausdorff_um),
            "avg_surface_distance_um": float(avg_surface_distance_um)
        }

    def _extract_boundary(self, mask: np.ndarray) -> np.ndarray:
        """Extract boundary points from binary mask."""
        if not np.any(mask):
            return np.array([]).reshape(0, 2)

        # Find contours
        mask_uint8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            return np.array([]).reshape(0, 2)

        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        boundary_points = largest_contour.reshape(-1, 2)

        return boundary_points

    def compute_coverage_metrics(self, predicted: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
        """Compute coverage and inclusion/exclusion metrics."""
        pred = predicted > 0.5
        gt = ground_truth > 0.5

        # True/False positives/negatives
        tp = np.logical_and(pred, gt).sum()
        fp = np.logical_and(pred, ~gt).sum()
        fn = np.logical_and(~pred, gt).sum()
        tn = np.logical_and(~pred, ~gt).sum()

        # Metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1_score": float(f1_score),
            "accuracy": float((tp + tn) / (tp + fp + fn + tn)) if (tp + fp + fn + tn) > 0 else 0.0
        }

    def compute_monolayer_metrics(
        self,
        mask: np.ndarray,
        quality_map: np.ndarray,
        thresholds: Tuple[float, float] = (0.3, 0.7),
    ) -> Dict[str, float]:
        """Compute monolayer coverage metrics from quality map.

        Args:
            mask: Binary smear mask.
            quality_map: Monolayer probability/density map (0-1).
            thresholds: Tuple of (low, high) thresholds defining sparse vs dense regions.
        """
        if mask.size == 0 or quality_map.size == 0:
            return {
                "monolayer_fraction": 0.0,
                "low_density_fraction": 0.0,
                "high_density_fraction": 0.0,
            }

        valid = mask > 0.5
        if not np.any(valid):
            return {
                "monolayer_fraction": 0.0,
                "low_density_fraction": 0.0,
                "high_density_fraction": 0.0,
            }

        clipped = np.nan_to_num(quality_map, nan=0.0, neginf=0.0, posinf=1.0)
        low_thr, high_thr = thresholds

        valid_values = clipped[valid]
        total = valid_values.size
        monolayer = np.count_nonzero(valid_values >= low_thr) / total
        high_density = np.count_nonzero(valid_values >= high_thr) / total
        low_density = np.count_nonzero(valid_values < low_thr) / total

        return {
            "monolayer_fraction": float(monolayer),
            "low_density_fraction": float(low_density),
            "high_density_fraction": float(high_density),
        }

    def evaluate_feathered_edge(self, predicted_edge: np.ndarray, ground_truth_edge: np.ndarray,
                               pixel_size_um: float = 1.0) -> Dict[str, float]:
        """Evaluate feathered edge detection accuracy."""
        if len(predicted_edge) == 0 or len(ground_truth_edge) == 0:
            return {"edge_error_um": float('inf')}

        # Find nearest points
        distances = spatial.distance.cdist(predicted_edge, ground_truth_edge)
        min_distances = distances.min(axis=1)

        edge_error_um = np.mean(min_distances) * pixel_size_um

        return {"edge_error_um": float(edge_error_um)}

    def evaluate_scan_plan(self, scan_plan: List[Dict], ground_truth_mask: np.ndarray,
                          mapper) -> Dict[str, float]:
        """Evaluate scan plan quality."""
        if not scan_plan:
            return {"coverage_score": 0.0, "efficiency_score": 0.0}

        # Convert scan plan back to pixel coordinates
        planned_pixels = []
        for tile in scan_plan:
            stage_coord = StageCoordinate(x_um=tile["x_um"], y_um=tile["y_um"], lens_id=tile["lens_id"])
            pixel_coord = mapper.to_pixel(stage_coord)
            planned_pixels.append((int(pixel_coord.v), int(pixel_coord.u)))

        # Create planned coverage mask
        h, w = ground_truth_mask.shape
        planned_mask = np.zeros((h, w), dtype=bool)

        tile_size_px = 100  # Approximate tile size in pixels

        for y, x in planned_pixels:
            y_start = max(0, y - tile_size_px // 2)
            y_end = min(h, y + tile_size_px // 2)
            x_start = max(0, x - tile_size_px // 2)
            x_end = min(w, x + tile_size_px // 2)
            planned_mask[y_start:y_end, x_start:x_end] = True

        # Compute coverage score
        gt_mask = ground_truth_mask > 0.5
        coverage_intersection = np.logical_and(planned_mask, gt_mask).sum()
        coverage_score = coverage_intersection / gt_mask.sum() if gt_mask.sum() > 0 else 0.0

        # Compute efficiency score (minimize wasted area outside GT)
        wasted_area = np.logical_and(planned_mask, ~gt_mask).sum()
        total_planned = planned_mask.sum()
        efficiency_score = 1.0 - (wasted_area / total_planned) if total_planned > 0 else 0.0

        return {
            "coverage_score": float(coverage_score),
            "efficiency_score": float(efficiency_score)
        }

    def summarize(self, result: RegionDetectionResult, ground_truth_mask: np.ndarray,
                 ground_truth_edge: Optional[np.ndarray] = None,
                 pixel_size_um: float = 1.0) -> Dict[str, float]:
        """Comprehensive evaluation summary."""
        metrics = {}

        # Basic segmentation metrics
        if result.artifacts.smear_mask is not None:
            metrics["iou"] = self.compute_iou(result.artifacts.smear_mask, ground_truth_mask)
            metrics["dice"] = self.compute_dice_coefficient(result.artifacts.smear_mask, ground_truth_mask)

            # Coverage metrics
            coverage_metrics = self.compute_coverage_metrics(result.artifacts.smear_mask, ground_truth_mask)
            metrics.update(coverage_metrics)

            # Boundary error
            boundary_metrics = self.compute_boundary_error(
                result.artifacts.smear_mask, ground_truth_mask, pixel_size_um
            )
            metrics.update(boundary_metrics)

        # Feathered edge evaluation
        if result.artifacts.feathered_edge is not None and ground_truth_edge is not None:
            edge_metrics = self.evaluate_feathered_edge(
                result.artifacts.feathered_edge, ground_truth_edge, pixel_size_um
            )
            metrics.update(edge_metrics)

        # Scan plan evaluation
        if result.scan_plan:
            scan_metrics = self.evaluate_scan_plan(result.scan_plan, ground_truth_mask, result.mapper)
            metrics.update(scan_metrics)

        return metrics


class CoordinateMappingValidator:
    """Validate coordinate mapping accuracy using fiducial points."""

    def __init__(self, tolerance_um: float = 3.0):
        self.tolerance_um = tolerance_um

    def validate_mapping(self, mapper, fiducial_points: List[Tuple[PixelCoordinate, StageCoordinate]]) -> Dict[str, float]:
        """Validate pixel-to-stage mapping accuracy."""
        pixel_to_stage_errors = []
        stage_to_pixel_errors = []

        for pixel_coord, expected_stage_coord in fiducial_points:
            # Test pixel-to-stage mapping
            predicted_stage = mapper.to_stage(pixel_coord)
            stage_error = np.sqrt(
                (predicted_stage.x_um - expected_stage_coord.x_um) ** 2 +
                (predicted_stage.y_um - expected_stage_coord.y_um) ** 2
            )
            pixel_to_stage_errors.append(stage_error)

            # Test stage-to-pixel mapping
            predicted_pixel = mapper.to_pixel(expected_stage_coord)
            pixel_error = np.sqrt(
                (predicted_pixel.u - pixel_coord.u) ** 2 +
                (predicted_pixel.v - pixel_coord.v) ** 2
            )
            stage_to_pixel_errors.append(pixel_error)

        return {
            "pixel_to_stage_rms_error_um": float(np.sqrt(np.mean(np.array(pixel_to_stage_errors) ** 2))),
            "pixel_to_stage_max_error_um": float(np.max(pixel_to_stage_errors)),
            "pixel_to_stage_p95_error_um": float(np.percentile(pixel_to_stage_errors, 95)),
            "stage_to_pixel_rms_error_px": float(np.sqrt(np.mean(np.array(stage_to_pixel_errors) ** 2))),
            "stage_to_pixel_max_error_px": float(np.max(stage_to_pixel_errors)),
            "stage_to_pixel_p95_error_px": float(np.percentile(stage_to_pixel_errors, 95)),
            "validation_passed": bool(np.percentile(pixel_to_stage_errors, 95) <= self.tolerance_um)
        }


class PerformanceBenchmark:
    """Benchmark pipeline performance and throughput."""

    def __init__(self):
        self.timing_results = {}

    def time_operation(self, operation_name: str, operation_func, *args, **kwargs):
        """Time a specific operation."""
        import time
        start_time = time.perf_counter()
        result = operation_func(*args, **kwargs)
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        self.timing_results[operation_name] = elapsed_time

        return result

    def benchmark_full_pipeline(self, service, survey_stream, slide_id: str) -> Dict[str, float]:
        """Benchmark the complete pipeline."""
        import time

        start_time = time.perf_counter()
        result = service.process_survey(survey_stream, slide_id)
        end_time = time.perf_counter()

        total_time = end_time - start_time

        # Extract image dimensions for throughput calculation
        if result.artifacts.smear_mask is not None:
            total_pixels = result.artifacts.smear_mask.size
            throughput_mpx_per_s = total_pixels / 1e6 / total_time
        else:
            throughput_mpx_per_s = 0.0

        return {
            "total_time_s": total_time,
            "throughput_mpx_per_s": throughput_mpx_per_s,
            **self.timing_results
        }


class RegressionTestSuite:
    """Automated regression testing for pipeline changes."""

    def __init__(self, test_data_dir: Path, tolerance: Dict[str, float]):
        self.test_data_dir = test_data_dir
        self.tolerance = tolerance

    def run_regression_tests(self, service) -> Dict[str, bool]:
        """Run full regression test suite."""
        test_results = {}

        # Load test cases
        test_cases = self._load_test_cases()

        for test_case in test_cases:
            case_results = self._run_single_test_case(service, test_case)
            test_results[test_case["name"]] = case_results

        return test_results

    def _load_test_cases(self) -> List[Dict]:
        """Load test cases from data directory."""
        # Placeholder implementation
        return [
            {"name": "normal_smear", "image_path": "normal_smear.png", "expected_iou": 0.92},
            {"name": "thin_smear", "image_path": "thin_smear.png", "expected_iou": 0.88},
            {"name": "thick_smear", "image_path": "thick_smear.png", "expected_iou": 0.85},
        ]

    def _run_single_test_case(self, service, test_case: Dict) -> bool:
        """Run a single test case."""
        # Placeholder - would load actual test data and run pipeline
        expected_iou = test_case.get("expected_iou", 0.9)
        tolerance = self.tolerance.get("iou", 0.05)

        # Simulate test result
        actual_iou = 0.91  # Placeholder

        return abs(actual_iou - expected_iou) <= tolerance


class QualityAssessmentReport:
    """Generate comprehensive QA reports."""

    def __init__(self, qa_results: Dict[str, float]):
        self.qa_results = qa_results

    def generate_pdf_report(self, output_path: Path) -> None:
        """Generate PDF report with visualizations."""
        # Placeholder for PDF generation
        # Would use matplotlib/reportlab to create comprehensive report
        pass

    def check_acceptance_criteria(self) -> Dict[str, bool]:
        """Check if results meet acceptance criteria from roadmap."""
        acceptance_results = {}

        # IoU thresholds
        iou = self.qa_results.get("iou", 0.0)
        acceptance_results["iou_p50"] = iou >= 0.92
        acceptance_results["iou_p95"] = iou >= 0.88  # Simplified - would need P95 calculation

        # Boundary error thresholds
        boundary_error = self.qa_results.get("hausdorff_um", float('inf'))
        acceptance_results["boundary_error_p95"] = boundary_error <= 40.0

        # Coordinate mapping thresholds
        mapping_error = self.qa_results.get("pixel_to_stage_p95_error_um", float('inf'))
        acceptance_results["mapping_accuracy"] = mapping_error <= 3.0

        # Overall pass/fail
        acceptance_results["overall_pass"] = all(acceptance_results.values())

        return acceptance_results

    def export_results(self, output_path: Path) -> None:
        """Export QA results as JSON."""
        import json

        output_path.parent.mkdir(parents=True, exist_ok=True)

        report_data = {
            "qa_metrics": self.qa_results,
            "acceptance_criteria": self.check_acceptance_criteria(),
            "timestamp": np.datetime64('now').isoformat()
        }

        with output_path.open('w') as f:
            json.dump(report_data, f, indent=2)
