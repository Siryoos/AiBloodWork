from __future__ import annotations

import time
import csv
import json
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import numpy as np

from .dual_lens import LensID, DualLensAutofocusManager, DualLensHandoffResult
from .dual_lens_camera import DualLensCameraController


@dataclass
class QATestResult:
    """Result from a QA test."""
    test_name: str
    passed: bool
    duration_s: float
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class QATest:
    """Base class for QA tests."""
    name: str
    description: str

    def run(self, system: Any, **kwargs) -> QATestResult:
        """Run the test and return results."""
        raise NotImplementedError


@dataclass
class DualLensQAConfig:
    """Configuration for dual-lens QA testing."""

    # Test execution
    num_handoff_tests: int = 50
    num_surface_calibration_tests: int = 20
    num_parfocal_validation_tests: int = 30

    # Performance targets
    handoff_time_target_ms: float = 300.0
    mapping_accuracy_target_um: float = 1.0
    surface_prediction_target_um: float = 0.5
    cross_lens_repeatability_target_um: float = 0.3

    # Test ranges
    x_test_range_um: Tuple[float, float] = (-5000.0, 5000.0)
    y_test_range_um: Tuple[float, float] = (-5000.0, 5000.0)
    z_test_range_um: Tuple[float, float] = (-10.0, 10.0)

    # Temperature testing
    enable_temperature_tests: bool = True
    temperature_range_c: Tuple[float, float] = (20.0, 30.0)

    # Output configuration
    output_dir: str = "dual_lens_qa_results"
    enable_detailed_logging: bool = True
    enable_performance_plots: bool = False


@dataclass
class HandoffPerformanceTest(QATest):
    """Test handoff performance between lenses."""

    name: str = "Dual-Lens Handoff Performance"
    description: str = "Validate A↔B handoff speed and accuracy"

    def run(self, system: DualLensAutofocusManager, **kwargs) -> QATestResult:
        """Run handoff performance test."""
        start_time = time.time()
        results = []
        errors = []

        config = kwargs.get('config', DualLensQAConfig())

        try:
            # Test A→B handoffs
            for i in range(config.num_handoff_tests // 2):
                # Random source position
                z_source = np.random.uniform(*config.z_test_range_um)

                # Ensure we start from Lens-A
                system.camera.set_active_lens(LensID.LENS_A)
                system.camera.set_focus(z_source)

                # Perform handoff
                handoff_result = system.handoff_a_to_b(z_source)
                results.append({
                    "direction": "A_to_B",
                    "source_z": z_source,
                    "handoff_result": handoff_result
                })

                if not handoff_result.success:
                    errors.append(f"A→B handoff failed at z={z_source:.2f}")

            # Test B→A handoffs
            for i in range(config.num_handoff_tests // 2):
                z_source = np.random.uniform(*config.z_test_range_um)

                # Ensure we start from Lens-B
                system.camera.set_active_lens(LensID.LENS_B)
                system.camera.set_focus(z_source)

                handoff_result = system.handoff_b_to_a(z_source)
                results.append({
                    "direction": "B_to_A",
                    "source_z": z_source,
                    "handoff_result": handoff_result
                })

                if not handoff_result.success:
                    errors.append(f"B→A handoff failed at z={z_source:.2f}")

            # Analyze results
            successful_handoffs = [r["handoff_result"] for r in results if r["handoff_result"].success]

            if not successful_handoffs:
                return QATestResult(
                    test_name=self.name,
                    passed=False,
                    duration_s=time.time() - start_time,
                    error_message="No successful handoffs",
                    details={"total_attempts": len(results), "errors": errors}
                )

            # Performance statistics
            handoff_times = [h.elapsed_ms for h in successful_handoffs]
            mapping_errors = [h.mapping_error_um for h in successful_handoffs]

            avg_time = np.mean(handoff_times)
            p95_time = np.percentile(handoff_times, 95)
            max_time = np.max(handoff_times)

            avg_error = np.mean(mapping_errors)
            p95_error = np.percentile(mapping_errors, 95)
            max_error = np.max(mapping_errors)

            # Check targets
            time_target_met = p95_time <= config.handoff_time_target_ms
            accuracy_target_met = p95_error <= config.mapping_accuracy_target_um

            passed = time_target_met and accuracy_target_met and len(errors) == 0

            details = {
                "total_tests": len(results),
                "successful_handoffs": len(successful_handoffs),
                "success_rate": len(successful_handoffs) / len(results),
                "performance_stats": {
                    "avg_handoff_time_ms": avg_time,
                    "p95_handoff_time_ms": p95_time,
                    "max_handoff_time_ms": max_time,
                    "time_target_met": time_target_met,
                    "avg_mapping_error_um": avg_error,
                    "p95_mapping_error_um": p95_error,
                    "max_mapping_error_um": max_error,
                    "accuracy_target_met": accuracy_target_met
                },
                "target_performance": {
                    "handoff_time_target_ms": config.handoff_time_target_ms,
                    "mapping_accuracy_target_um": config.mapping_accuracy_target_um
                },
                "errors": errors
            }

            return QATestResult(
                test_name=self.name,
                passed=passed,
                duration_s=time.time() - start_time,
                details=details
            )

        except Exception as e:
            return QATestResult(
                test_name=self.name,
                passed=False,
                duration_s=time.time() - start_time,
                error_message=str(e),
                details={"errors": errors}
            )


@dataclass
class SurfaceCalibrationTest(QATest):
    """Test focus surface calibration and prediction accuracy."""

    name: str = "Focus Surface Calibration"
    description: str = "Validate per-lens focus surface models"

    def run(self, system: DualLensAutofocusManager, **kwargs) -> QATestResult:
        """Run surface calibration test."""
        start_time = time.time()
        config = kwargs.get('config', DualLensQAConfig())

        try:
            results = {}

            for lens_id in [LensID.LENS_A, LensID.LENS_B]:
                # Generate calibration points
                num_points = config.num_surface_calibration_tests
                calibration_points = self._generate_calibration_points(
                    num_points, config.x_test_range_um, config.y_test_range_um, config.z_test_range_um
                )

                # Perform calibration
                calibration_result = system.camera.calibrate_focus_surface(lens_id, calibration_points)

                # Test prediction accuracy
                test_points = self._generate_calibration_points(
                    10, config.x_test_range_um, config.y_test_range_um, config.z_test_range_um
                )

                prediction_errors = []
                for x, y, actual_z in test_points:
                    predicted_z = system.camera.predict_focus_position(lens_id, x, y)
                    if predicted_z is not None:
                        error = abs(predicted_z - actual_z)
                        prediction_errors.append(error)

                # Analyze prediction accuracy
                if prediction_errors:
                    avg_prediction_error = np.mean(prediction_errors)
                    max_prediction_error = np.max(prediction_errors)
                    rms_prediction_error = np.sqrt(np.mean(np.array(prediction_errors)**2))
                else:
                    avg_prediction_error = max_prediction_error = rms_prediction_error = 999.0

                results[lens_id.value] = {
                    "calibration": calibration_result,
                    "prediction_accuracy": {
                        "avg_error_um": avg_prediction_error,
                        "max_error_um": max_prediction_error,
                        "rms_error_um": rms_prediction_error,
                        "num_test_points": len(prediction_errors),
                        "target_met": rms_prediction_error <= config.surface_prediction_target_um
                    }
                }

            # Overall pass/fail
            all_targets_met = all(
                results[lens_id.value]["prediction_accuracy"]["target_met"]
                for lens_id in [LensID.LENS_A, LensID.LENS_B]
            )

            return QATestResult(
                test_name=self.name,
                passed=all_targets_met,
                duration_s=time.time() - start_time,
                details={
                    "lens_results": results,
                    "prediction_target_um": config.surface_prediction_target_um
                }
            )

        except Exception as e:
            return QATestResult(
                test_name=self.name,
                passed=False,
                duration_s=time.time() - start_time,
                error_message=str(e)
            )

    def _generate_calibration_points(self,
                                   num_points: int,
                                   x_range: Tuple[float, float],
                                   y_range: Tuple[float, float],
                                   z_range: Tuple[float, float]) -> List[Tuple[float, float, float]]:
        """Generate calibration points with realistic focus surface."""
        points = []

        for _ in range(num_points):
            x = np.random.uniform(*x_range)
            y = np.random.uniform(*y_range)

            # Simulate realistic focus surface with slight tilt and curvature
            z_base = np.random.uniform(*z_range)
            tilt_x = 0.0001 * x  # Slight X tilt
            tilt_y = 0.0001 * y  # Slight Y tilt
            curvature = 0.000001 * (x**2 + y**2)  # Slight field curvature

            z = z_base + tilt_x + tilt_y + curvature
            z = np.clip(z, *z_range)  # Keep within range

            points.append((x, y, z))

        return points


@dataclass
class ParfocalValidationTest(QATest):
    """Test parfocal mapping accuracy and consistency."""

    name: str = "Parfocal Mapping Validation"
    description: str = "Validate cross-lens parfocal mapping accuracy"

    def run(self, system: DualLensAutofocusManager, **kwargs) -> QATestResult:
        """Run parfocal validation test."""
        start_time = time.time()
        config = kwargs.get('config', DualLensQAConfig())

        try:
            mapping_errors = []
            consistency_errors = []

            for i in range(config.num_parfocal_validation_tests):
                # Test A→B→A round-trip consistency
                z_start = np.random.uniform(*config.z_test_range_um)

                # Map A→B
                z_b_predicted = system.parfocal_mapping.map_lens_a_to_b(z_start)

                # Map B→A
                z_a_recovered = system.parfocal_mapping.map_lens_b_to_a(z_b_predicted)

                # Calculate round-trip error
                round_trip_error = abs(z_a_recovered - z_start)
                consistency_errors.append(round_trip_error)

                # Test actual handoff accuracy
                # Set Lens-A to start position
                system.camera.set_active_lens(LensID.LENS_A)
                system.camera.set_focus(z_start)

                # Perform handoff
                handoff_result = system.handoff_a_to_b(z_start)
                if handoff_result.success:
                    mapping_errors.append(handoff_result.mapping_error_um)

            # Analyze results
            if mapping_errors:
                avg_mapping_error = np.mean(mapping_errors)
                p95_mapping_error = np.percentile(mapping_errors, 95)
                max_mapping_error = np.max(mapping_errors)
            else:
                avg_mapping_error = p95_mapping_error = max_mapping_error = 999.0

            avg_consistency_error = np.mean(consistency_errors)
            max_consistency_error = np.max(consistency_errors)

            # Check targets
            mapping_target_met = p95_mapping_error <= config.mapping_accuracy_target_um
            consistency_target_met = avg_consistency_error <= config.cross_lens_repeatability_target_um

            passed = mapping_target_met and consistency_target_met

            return QATestResult(
                test_name=self.name,
                passed=passed,
                duration_s=time.time() - start_time,
                details={
                    "mapping_accuracy": {
                        "avg_error_um": avg_mapping_error,
                        "p95_error_um": p95_mapping_error,
                        "max_error_um": max_mapping_error,
                        "target_met": mapping_target_met,
                        "num_tests": len(mapping_errors)
                    },
                    "round_trip_consistency": {
                        "avg_error_um": avg_consistency_error,
                        "max_error_um": max_consistency_error,
                        "target_met": consistency_target_met,
                        "num_tests": len(consistency_errors)
                    },
                    "targets": {
                        "mapping_accuracy_target_um": config.mapping_accuracy_target_um,
                        "consistency_target_um": config.cross_lens_repeatability_target_um
                    }
                }
            )

        except Exception as e:
            return QATestResult(
                test_name=self.name,
                passed=False,
                duration_s=time.time() - start_time,
                error_message=str(e)
            )


@dataclass
class TemperatureCompensationTest(QATest):
    """Test temperature compensation for parfocal mapping."""

    name: str = "Temperature Compensation"
    description: str = "Validate thermal drift compensation"

    def run(self, system: DualLensAutofocusManager, **kwargs) -> QATestResult:
        """Run temperature compensation test."""
        start_time = time.time()
        config = kwargs.get('config', DualLensQAConfig())

        if not config.enable_temperature_tests:
            return QATestResult(
                test_name=self.name,
                passed=True,
                duration_s=0.0,
                details={"status": "skipped", "reason": "temperature tests disabled"}
            )

        try:
            temp_range = config.temperature_range_c
            base_temp = 23.0  # Reference temperature

            temperature_errors = []

            # Test at different temperatures
            test_temperatures = np.linspace(temp_range[0], temp_range[1], 5)

            for temp in test_temperatures:
                # Update system temperature
                system.update_temperature(temp)

                # Test mapping at this temperature
                z_test = 0.0  # Test at center position
                z_mapped_temp = system.parfocal_mapping.map_lens_a_to_b(z_test, temp)
                z_mapped_base = system.parfocal_mapping.map_lens_a_to_b(z_test, base_temp)

                # Calculate temperature-induced error
                temp_error = abs(z_mapped_temp - z_mapped_base)
                temperature_errors.append({
                    "temperature_c": temp,
                    "temp_delta_c": temp - base_temp,
                    "mapping_shift_um": temp_error
                })

            # Analyze temperature sensitivity
            temp_deltas = [e["temp_delta_c"] for e in temperature_errors]
            mapping_shifts = [e["mapping_shift_um"] for e in temperature_errors]

            if len(temp_deltas) > 1:
                # Calculate temperature coefficient
                temp_coeff = np.polyfit(temp_deltas, mapping_shifts, 1)[0]  # um/°C
            else:
                temp_coeff = 0.0

            max_temp_error = max(mapping_shifts) if mapping_shifts else 0.0

            # Target: <0.1 μm shift per °C
            temp_target_met = abs(temp_coeff) <= 0.1

            return QATestResult(
                test_name=self.name,
                passed=temp_target_met,
                duration_s=time.time() - start_time,
                details={
                    "temperature_sensitivity": {
                        "coefficient_um_per_c": temp_coeff,
                        "max_shift_um": max_temp_error,
                        "target_met": temp_target_met,
                        "target_um_per_c": 0.1
                    },
                    "test_data": temperature_errors,
                    "temperature_range_c": temp_range
                }
            )

        except Exception as e:
            return QATestResult(
                test_name=self.name,
                passed=False,
                duration_s=time.time() - start_time,
                error_message=str(e)
            )


class DualLensQAHarness:
    """QA harness for dual-lens autofocus system validation."""

    def __init__(self, config: DualLensQAConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Define test suite
        self.tests = [
            HandoffPerformanceTest(),
            SurfaceCalibrationTest(),
            ParfocalValidationTest(),
            TemperatureCompensationTest()
        ]

        # Results storage
        self.test_results: List[QATestResult] = []
        self._lock = threading.Lock()

    def run_full_validation(self, system: DualLensAutofocusManager) -> Dict[str, Any]:
        """Run complete dual-lens validation suite."""
        start_time = time.time()

        print("Starting dual-lens autofocus QA validation...")
        print(f"Output directory: {self.output_dir}")

        results = {}
        all_passed = True

        for test in self.tests:
            print(f"\nRunning {test.name}...")

            test_result = test.run(system, config=self.config)
            self.test_results.append(test_result)

            status = "PASS" if test_result.passed else "FAIL"
            print(f"  {status} ({test_result.duration_s:.1f}s)")

            if not test_result.passed:
                all_passed = False
                if test_result.error_message:
                    print(f"  Error: {test_result.error_message}")

            results[test.name] = test_result

        total_duration = time.time() - start_time

        # Generate summary report
        summary = self._generate_summary_report(results, total_duration, all_passed)

        # Save detailed results
        self._save_results(results, summary)

        print(f"\nQA Validation {'PASSED' if all_passed else 'FAILED'}")
        print(f"Total duration: {total_duration:.1f}s")
        print(f"Results saved to: {self.output_dir}")

        return summary

    def _generate_summary_report(self,
                                results: Dict[str, QATestResult],
                                total_duration: float,
                                all_passed: bool) -> Dict[str, Any]:
        """Generate summary report."""
        summary = {
            "overall_status": "PASS" if all_passed else "FAIL",
            "total_duration_s": total_duration,
            "timestamp": time.time(),
            "config": {
                "handoff_time_target_ms": self.config.handoff_time_target_ms,
                "mapping_accuracy_target_um": self.config.mapping_accuracy_target_um,
                "surface_prediction_target_um": self.config.surface_prediction_target_um,
                "cross_lens_repeatability_target_um": self.config.cross_lens_repeatability_target_um
            },
            "test_summary": {}
        }

        for test_name, result in results.items():
            summary["test_summary"][test_name] = {
                "status": "PASS" if result.passed else "FAIL",
                "duration_s": result.duration_s,
                "error": result.error_message
            }

        # Extract key performance metrics
        handoff_test = results.get("Dual-Lens Handoff Performance")
        if handoff_test and handoff_test.passed and handoff_test.details:
            perf_stats = handoff_test.details.get("performance_stats", {})
            summary["key_metrics"] = {
                "avg_handoff_time_ms": perf_stats.get("avg_handoff_time_ms"),
                "p95_handoff_time_ms": perf_stats.get("p95_handoff_time_ms"),
                "avg_mapping_error_um": perf_stats.get("avg_mapping_error_um"),
                "p95_mapping_error_um": perf_stats.get("p95_mapping_error_um"),
                "handoff_success_rate": handoff_test.details.get("success_rate")
            }

        return summary

    def _save_results(self, results: Dict[str, QATestResult], summary: Dict[str, Any]) -> None:
        """Save test results to files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save JSON summary
        summary_file = self.output_dir / f"dual_lens_qa_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Save detailed results
        if self.config.enable_detailed_logging:
            details_file = self.output_dir / f"dual_lens_qa_details_{timestamp}.json"
            detailed_results = {}

            for test_name, result in results.items():
                detailed_results[test_name] = {
                    "passed": result.passed,
                    "duration_s": result.duration_s,
                    "error_message": result.error_message,
                    "details": result.details
                }

            with open(details_file, 'w') as f:
                json.dump(detailed_results, f, indent=2, default=str)

        # Save CSV performance data
        self._save_performance_csv(results, timestamp)

    def _save_performance_csv(self, results: Dict[str, QATestResult], timestamp: str) -> None:
        """Save performance data to CSV for analysis."""
        csv_file = self.output_dir / f"dual_lens_performance_{timestamp}.csv"

        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "test_name", "metric_name", "value", "unit", "target", "target_met"
            ])

            # Extract metrics from each test
            for test_name, result in results.items():
                if not result.details:
                    continue

                if test_name == "Dual-Lens Handoff Performance" and "performance_stats" in result.details:
                    stats = result.details["performance_stats"]
                    targets = result.details["target_performance"]

                    writer.writerow(["Handoff", "avg_time", stats.get("avg_handoff_time_ms"), "ms",
                                   targets.get("handoff_time_target_ms"), ""])
                    writer.writerow(["Handoff", "p95_time", stats.get("p95_handoff_time_ms"), "ms",
                                   targets.get("handoff_time_target_ms"), stats.get("time_target_met")])
                    writer.writerow(["Handoff", "avg_error", stats.get("avg_mapping_error_um"), "um",
                                   targets.get("mapping_accuracy_target_um"), ""])
                    writer.writerow(["Handoff", "p95_error", stats.get("p95_mapping_error_um"), "um",
                                   targets.get("mapping_accuracy_target_um"), stats.get("accuracy_target_met")])

                elif test_name == "Parfocal Mapping Validation":
                    if "mapping_accuracy" in result.details:
                        acc = result.details["mapping_accuracy"]
                        writer.writerow(["Mapping", "avg_error", acc.get("avg_error_um"), "um",
                                       result.details["targets"]["mapping_accuracy_target_um"], ""])
                        writer.writerow(["Mapping", "p95_error", acc.get("p95_error_um"), "um",
                                       result.details["targets"]["mapping_accuracy_target_um"], acc.get("target_met")])

    def get_test_results(self) -> List[QATestResult]:
        """Get all test results."""
        return self.test_results.copy()

    def clear_results(self) -> None:
        """Clear stored test results."""
        with self._lock:
            self.test_results.clear()