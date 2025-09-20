from __future__ import annotations

import json
import time
import csv
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from pathlib import Path
from datetime import datetime
import numpy as np

from .validation import FocusAccuracyTest, MTFMeasurement, AutofocusTestSuite
from .outlier_detection import OutlierDetector, OutlierDetectionConfig
from .telemetry import ProductionTelemetryLogger, RegulatoryLogger


@dataclass
class KPIThresholds:
    """Key Performance Indicator thresholds for autofocus validation."""

    # Accuracy requirements
    max_focus_error_um: float = 1.0  # PBS monolayer
    max_focus_error_edge_um: float = 1.5  # Feathered edge
    max_repeatability_um: float = 0.5

    # Performance requirements
    max_elapsed_p95_ms: float = 150.0  # 95th percentile
    min_success_rate: float = 0.95

    # Image quality requirements
    min_mtf50_ratio: float = 0.95  # Relative to ground truth
    min_hf_ratio: float = 0.95

    # Robustness requirements
    max_temperature_sensitivity_um_per_c: float = 1.0
    max_illumination_sensitivity_um: float = 0.5
    max_tilt_sensitivity_um: float = 2.0  # Per 50um tile height difference


@dataclass
class TestConfiguration:
    """Configuration for automated testing."""

    # Test slide information
    slide_id: str
    slide_type: str = "PBS"  # PBS, bone_marrow, calibration_target
    stain_type: str = "Wright_Giemsa"

    # Test positions
    test_positions: List[Tuple[float, float, float]] = field(default_factory=list)  # (x, y, z_expected)
    edge_positions: List[Tuple[float, float, float]] = field(default_factory=list)  # Feathered edge

    # Environmental test conditions
    temperature_range_c: Tuple[float, float] = (18.0, 28.0)
    illumination_variations: List[str] = field(default_factory=lambda: ["NORMAL", "+10%", "-10%"])
    tilt_test_positions: List[Tuple[float, float, float]] = field(default_factory=list)  # Shimmed positions

    # Test parameters
    repeatability_count: int = 20
    timeout_per_test_s: float = 30.0


@dataclass
class TestResult:
    """Result of a single autofocus test."""

    test_id: str
    timestamp: float
    position: Tuple[float, float, float]  # x, y, z_expected
    z_measured: Optional[float] = None
    elapsed_ms: Optional[float] = None
    metric_values: Optional[Dict[str, float]] = None
    mtf50_ratio: Optional[float] = None
    status: str = "unknown"  # success, failed, timeout, error
    error_message: Optional[str] = None
    environmental_conditions: Optional[Dict[str, Any]] = None


@dataclass
class TestReport:
    """Comprehensive test report."""

    test_suite_id: str
    timestamp: float
    configuration: TestConfiguration
    kpi_thresholds: KPIThresholds

    # Test results
    test_results: List[TestResult] = field(default_factory=list)
    kpi_results: Dict[str, Any] = field(default_factory=dict)

    # Summary statistics
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    overall_status: str = "unknown"  # passed, failed, partial

    # Detailed analysis
    accuracy_analysis: Optional[Dict[str, Any]] = None
    performance_analysis: Optional[Dict[str, Any]] = None
    robustness_analysis: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def save_json(self, filepath: Union[str, Path]) -> None:
        """Save report as JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_csv_summary(self, filepath: Union[str, Path]) -> None:
        """Save summary results as CSV."""
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'test_id', 'x_um', 'y_um', 'z_expected_um', 'z_measured_um',
                'focus_error_um', 'elapsed_ms', 'status'
            ])
            writer.writeheader()

            for result in self.test_results:
                if result.z_measured is not None:
                    focus_error = abs(result.z_measured - result.position[2])
                else:
                    focus_error = None

                writer.writerow({
                    'test_id': result.test_id,
                    'x_um': result.position[0],
                    'y_um': result.position[1],
                    'z_expected_um': result.position[2],
                    'z_measured_um': result.z_measured,
                    'focus_error_um': focus_error,
                    'elapsed_ms': result.elapsed_ms,
                    'status': result.status
                })


class AutofocusQAHarness:
    """Production QA harness for autofocus validation."""

    def __init__(self,
                 autofocus_system,
                 kpi_thresholds: Optional[KPIThresholds] = None,
                 output_dir: Union[str, Path] = "./qa_results"):
        """Initialize QA harness.

        Args:
            autofocus_system: BloodSmearAutofocus instance to test
            kpi_thresholds: KPI acceptance thresholds
            output_dir: Directory for test outputs
        """
        self.autofocus_system = autofocus_system
        self.kpi_thresholds = kpi_thresholds or KPIThresholds()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.outlier_detector = OutlierDetector(OutlierDetectionConfig())
        self.telemetry_logger = ProductionTelemetryLogger(
            self.output_dir / "telemetry",
            enable_csv=True,
            enable_sqlite=False  # Disable SQLite to avoid schema issues
        )
        self.regulatory_logger = RegulatoryLogger(self.output_dir / "regulatory")

    def run_full_validation(self,
                          test_config: TestConfiguration,
                          include_robustness: bool = True) -> TestReport:
        """Run complete validation test suite.

        Args:
            test_config: Test configuration
            include_robustness: Whether to run robustness tests

        Returns:
            Comprehensive test report
        """
        test_suite_id = f"QA_{test_config.slide_id}_{int(time.time())}"
        report = TestReport(
            test_suite_id=test_suite_id,
            timestamp=time.time(),
            configuration=test_config,
            kpi_thresholds=self.kpi_thresholds
        )

        print(f"Starting QA validation: {test_suite_id}")

        try:
            # Run accuracy tests
            print("Running accuracy tests...")
            accuracy_results = self._run_accuracy_tests(test_config)
            report.test_results.extend(accuracy_results)

            # Run performance tests
            print("Running performance tests...")
            performance_results = self._run_performance_tests(test_config)
            report.test_results.extend(performance_results)

            # Run robustness tests if requested
            if include_robustness:
                print("Running robustness tests...")
                robustness_results = self._run_robustness_tests(test_config)
                report.test_results.extend(robustness_results)

            # Analyze results
            print("Analyzing results...")
            self._analyze_results(report)

            # Generate final report
            self._finalize_report(report)

        except Exception as e:
            report.overall_status = "error"
            print(f"QA validation failed: {e}")

        return report

    def _run_accuracy_tests(self, config: TestConfiguration) -> List[TestResult]:
        """Run accuracy and repeatability tests."""
        results = []

        # Test each position
        for i, (x, y, z_expected) in enumerate(config.test_positions):
            test_id = f"accuracy_{config.slide_id}_{i}"

            # Single measurement for basic accuracy
            result = self._run_single_test(test_id, x, y, z_expected)
            results.append(result)

            # Repeatability test (multiple measurements at same position)
            for rep in range(config.repeatability_count):
                rep_test_id = f"repeatability_{config.slide_id}_{i}_{rep}"
                rep_result = self._run_single_test(rep_test_id, x, y, z_expected)
                results.append(rep_result)

        # Test edge positions with relaxed thresholds
        for i, (x, y, z_expected) in enumerate(config.edge_positions):
            test_id = f"edge_{config.slide_id}_{i}"
            result = self._run_single_test(test_id, x, y, z_expected, is_edge=True)
            results.append(result)

        return results

    def _run_performance_tests(self, config: TestConfiguration) -> List[TestResult]:
        """Run performance and throughput tests."""
        results = []

        # Test performance across different positions
        for i, (x, y, z_expected) in enumerate(config.test_positions[:5]):  # Subset for timing
            test_id = f"performance_{config.slide_id}_{i}"

            # Multiple quick runs to measure timing distribution
            for run in range(10):
                run_test_id = f"{test_id}_run_{run}"
                result = self._run_single_test(run_test_id, x, y, z_expected)
                results.append(result)

        return results

    def _run_robustness_tests(self, config: TestConfiguration) -> List[TestResult]:
        """Run robustness tests under varying conditions."""
        results = []

        # Temperature variation tests
        if hasattr(self.autofocus_system, '_drift_tracker'):
            for temp in [config.temperature_range_c[0],
                        (config.temperature_range_c[0] + config.temperature_range_c[1]) / 2,
                        config.temperature_range_c[1]]:

                # Simulate temperature (in real system, would control chamber)
                test_id = f"temp_{config.slide_id}_{temp:.1f}C"
                x, y, z_expected = config.test_positions[0]  # Use first position

                result = self._run_single_test(test_id, x, y, z_expected)
                if result.environmental_conditions is None:
                    result.environmental_conditions = {}
                result.environmental_conditions["temperature_c"] = temp
                results.append(result)

        # Illumination variation tests
        for illum_var in config.illumination_variations:
            test_id = f"illum_{config.slide_id}_{illum_var}"
            x, y, z_expected = config.test_positions[0]

            # Modify illumination if possible
            self._set_illumination_variation(illum_var)

            result = self._run_single_test(test_id, x, y, z_expected)
            if result.environmental_conditions is None:
                result.environmental_conditions = {}
            result.environmental_conditions["illumination"] = illum_var
            results.append(result)

        # Tilt sensitivity tests
        for i, (x, y, z_expected) in enumerate(config.tilt_test_positions):
            test_id = f"tilt_{config.slide_id}_{i}"
            result = self._run_single_test(test_id, x, y, z_expected)
            results.append(result)

        return results

    def _run_single_test(self,
                        test_id: str,
                        x: float,
                        y: float,
                        z_expected: float,
                        is_edge: bool = False) -> TestResult:
        """Run single autofocus test."""
        start_time = time.time()

        result = TestResult(
            test_id=test_id,
            timestamp=start_time,
            position=(x, y, z_expected)
        )

        try:
            # Run autofocus
            z_measured = self.autofocus_system.autofocus_at_position(x=x, y=y)
            elapsed_ms = (time.time() - start_time) * 1000

            result.z_measured = z_measured
            result.elapsed_ms = elapsed_ms

            # Collect metrics
            result.metric_values = self._collect_current_metrics()

            # Measure MTF if possible
            if hasattr(self.autofocus_system, '_validator'):
                try:
                    mtf_result = self._measure_mtf_at_position(z_measured, z_expected)
                    result.mtf50_ratio = mtf_result.get('mtf50_ratio')
                except Exception:
                    pass

            # Check for outliers
            outlier_result = self.outlier_detector.analyze_focus_result(
                z_result=z_measured,
                metric_value=result.metric_values.get('tenengrad', 0),
                elapsed_ms=elapsed_ms
            )

            # Determine pass/fail
            focus_error = abs(z_measured - z_expected)
            threshold = (self.kpi_thresholds.max_focus_error_edge_um if is_edge
                        else self.kpi_thresholds.max_focus_error_um)

            if focus_error <= threshold and not outlier_result.is_outlier:
                result.status = "passed"
            else:
                result.status = "failed"

            # Log to telemetry
            self.telemetry_logger.log_autofocus_operation(
                tile_id=test_id,
                x_um=x,
                y_um=y,
                z_af_um=z_measured,
                elapsed_ms=elapsed_ms,
                status=result.status,
                focus_error_estimate_um=focus_error,
                metric_values=result.metric_values
            )

            # Log regulatory event
            self.regulatory_logger.log_autofocus_decision(
                tile_id=test_id,
                z_af_um=z_measured,
                confidence_metrics=result.metric_values or {}
            )

        except Exception as e:
            result.status = "error"
            result.error_message = str(e)
            result.elapsed_ms = (time.time() - start_time) * 1000

        return result

    def _collect_current_metrics(self) -> Dict[str, float]:
        """Collect current focus metrics."""
        try:
            frame = self.autofocus_system.camera.get_frame()

            from .metrics import tenengrad, variance_of_laplacian, brenner_gradient

            return {
                "tenengrad": tenengrad(frame),
                "laplacian": variance_of_laplacian(frame),
                "brenner": brenner_gradient(frame)
            }
        except Exception:
            return {}

    def _measure_mtf_at_position(self, z_measured: float, z_expected: float) -> Dict[str, float]:
        """Measure MTF at current position."""
        try:
            frame = self.autofocus_system.camera.get_frame()
            freqs, mtf = MTFMeasurement.slanted_edge_mtf(frame)
            mtf50_measured = MTFMeasurement.mtf50(freqs, mtf)

            # Move to expected position and measure reference MTF
            self.autofocus_system.camera.set_focus(z_expected)
            time.sleep(0.05)
            frame_ref = self.autofocus_system.camera.get_frame()
            freqs_ref, mtf_ref = MTFMeasurement.slanted_edge_mtf(frame_ref)
            mtf50_expected = MTFMeasurement.mtf50(freqs_ref, mtf_ref)

            # Restore measured position
            self.autofocus_system.camera.set_focus(z_measured)

            mtf50_ratio = mtf50_measured / mtf50_expected if mtf50_expected > 0 else 0

            return {
                "mtf50_measured": mtf50_measured,
                "mtf50_expected": mtf50_expected,
                "mtf50_ratio": mtf50_ratio
            }

        except Exception:
            return {}

    def _set_illumination_variation(self, variation: str) -> None:
        """Set illumination variation for testing."""
        if not hasattr(self.autofocus_system, '_illumination_manager'):
            return

        manager = self.autofocus_system._illumination_manager

        if variation == "+10%":
            manager.set_uniform(0.55)  # 10% brighter
        elif variation == "-10%":
            manager.set_uniform(0.45)  # 10% dimmer
        else:
            manager.set_uniform(0.5)   # Normal

    def _analyze_results(self, report: TestReport) -> None:
        """Analyze test results and compute KPIs."""
        results = report.test_results

        # Basic statistics
        report.total_tests = len(results)
        report.passed_tests = sum(1 for r in results if r.status == "passed")
        report.failed_tests = sum(1 for r in results if r.status in ["failed", "error"])

        # Accuracy analysis
        focus_errors = []
        repeatability_errors = []
        edge_errors = []

        for result in results:
            if result.z_measured is not None:
                error = abs(result.z_measured - result.position[2])
                focus_errors.append(error)

                if "repeatability" in result.test_id:
                    repeatability_errors.append(error)
                elif "edge" in result.test_id:
                    edge_errors.append(error)

        report.accuracy_analysis = {
            "mean_error_um": float(np.mean(focus_errors)) if focus_errors else 0,
            "max_error_um": float(np.max(focus_errors)) if focus_errors else 0,
            "p95_error_um": float(np.percentile(focus_errors, 95)) if focus_errors else 0,
            "repeatability_std_um": float(np.std(repeatability_errors)) if repeatability_errors else 0,
            "edge_mean_error_um": float(np.mean(edge_errors)) if edge_errors else 0
        }

        # Performance analysis
        elapsed_times = [r.elapsed_ms for r in results if r.elapsed_ms is not None]
        report.performance_analysis = {
            "mean_elapsed_ms": float(np.mean(elapsed_times)) if elapsed_times else 0,
            "p95_elapsed_ms": float(np.percentile(elapsed_times, 95)) if elapsed_times else 0,
            "p99_elapsed_ms": float(np.percentile(elapsed_times, 99)) if elapsed_times else 0,
            "success_rate": report.passed_tests / report.total_tests if report.total_tests > 0 else 0
        }

        # KPI evaluation
        kpis = {}

        if focus_errors:
            kpis["accuracy_pass"] = np.max(focus_errors) <= self.kpi_thresholds.max_focus_error_um
        if repeatability_errors:
            kpis["repeatability_pass"] = np.std(repeatability_errors) <= self.kpi_thresholds.max_repeatability_um
        if elapsed_times:
            kpis["performance_pass"] = np.percentile(elapsed_times, 95) <= self.kpi_thresholds.max_elapsed_p95_ms

        kpis["success_rate_pass"] = (report.passed_tests / report.total_tests) >= self.kpi_thresholds.min_success_rate

        report.kpi_results = kpis

        # Overall status
        if all(kpis.values()):
            report.overall_status = "passed"
        elif any(kpis.values()):
            report.overall_status = "partial"
        else:
            report.overall_status = "failed"

    def _finalize_report(self, report: TestReport) -> None:
        """Finalize and save test report."""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed JSON report
        json_path = self.output_dir / f"qa_report_{report.test_suite_id}_{timestamp_str}.json"
        report.save_json(json_path)

        # Save CSV summary
        csv_path = self.output_dir / f"qa_summary_{report.test_suite_id}_{timestamp_str}.csv"
        report.save_csv_summary(csv_path)

        # Generate PDF report if possible
        try:
            self._generate_pdf_report(report, timestamp_str)
        except Exception as e:
            print(f"Could not generate PDF report: {e}")

        print(f"QA validation complete: {report.overall_status}")
        print(f"Results saved to: {self.output_dir}")

    def _generate_pdf_report(self, report: TestReport, timestamp_str: str) -> None:
        """Generate PDF report (requires matplotlib/reportlab)."""
        try:
            import matplotlib.pyplot as plt

            # Create summary plots
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            # Focus error histogram
            focus_errors = [abs(r.z_measured - r.position[2])
                          for r in report.test_results
                          if r.z_measured is not None]
            if focus_errors:
                axes[0, 0].hist(focus_errors, bins=20)
                axes[0, 0].set_title("Focus Error Distribution")
                axes[0, 0].set_xlabel("Error (μm)")
                axes[0, 0].axvline(self.kpi_thresholds.max_focus_error_um,
                                 color='r', linestyle='--', label='Threshold')
                axes[0, 0].legend()

            # Timing histogram
            elapsed_times = [r.elapsed_ms for r in report.test_results
                           if r.elapsed_ms is not None]
            if elapsed_times:
                axes[0, 1].hist(elapsed_times, bins=20)
                axes[0, 1].set_title("Elapsed Time Distribution")
                axes[0, 1].set_xlabel("Time (ms)")
                axes[0, 1].axvline(self.kpi_thresholds.max_elapsed_p95_ms,
                                 color='r', linestyle='--', label='P95 Threshold')
                axes[0, 1].legend()

            # KPI status
            kpi_names = list(report.kpi_results.keys())
            kpi_values = [1 if v else 0 for v in report.kpi_results.values()]
            if kpi_names:
                axes[1, 0].bar(kpi_names, kpi_values)
                axes[1, 0].set_title("KPI Status")
                axes[1, 0].set_ylabel("Pass (1) / Fail (0)")
                axes[1, 0].tick_params(axis='x', rotation=45)

            # Status pie chart
            status_counts = {}
            for result in report.test_results:
                status_counts[result.status] = status_counts.get(result.status, 0) + 1

            if status_counts:
                axes[1, 1].pie(status_counts.values(), labels=status_counts.keys(), autopct='%1.1f%%')
                axes[1, 1].set_title("Test Results Distribution")

            plt.tight_layout()

            pdf_path = self.output_dir / f"qa_report_{report.test_suite_id}_{timestamp_str}.pdf"
            plt.savefig(pdf_path)
            plt.close()

        except ImportError:
            pass  # matplotlib not available

    def close(self) -> None:
        """Close QA harness and clean up resources."""
        self.telemetry_logger.close()
        self.regulatory_logger.close()


# Factory function for easy setup
def create_qa_harness(autofocus_system,
                     kpi_thresholds: Optional[KPIThresholds] = None,
                     output_dir: Union[str, Path] = "./qa_results") -> AutofocusQAHarness:
    """Create QA harness for autofocus validation.

    Args:
        autofocus_system: BloodSmearAutofocus instance
        kpi_thresholds: Custom KPI thresholds
        output_dir: Output directory for results

    Returns:
        Configured QA harness
    """
    return AutofocusQAHarness(autofocus_system, kpi_thresholds, output_dir)