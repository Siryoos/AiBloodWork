# Testing Framework Guide

## 🧪 Overview

The Dual-Lens Autofocus system includes comprehensive testing frameworks for validation, performance benchmarking, and quality assurance. This guide covers all testing capabilities from unit tests to production validation.

## 📋 Table of Contents

- [QA Framework](#qa-framework)
- [Performance Testing](#performance-testing)
- [Unit Testing](#unit-testing)
- [Integration Testing](#integration-testing)
- [Hardware Testing](#hardware-testing)
- [Continuous Testing](#continuous-testing)
- [Test Results Analysis](#test-results-analysis)

## 🏭 QA Framework

### DualLensQAHarness

The main QA validation system for production readiness testing.

```python
from autofocus.dual_lens_qa import DualLensQAHarness, DualLensQAConfig

# Configure QA testing
qa_config = DualLensQAConfig(
    # Performance targets
    handoff_time_target_ms=300.0,
    mapping_accuracy_target_um=1.0,
    surface_prediction_target_um=0.5,
    cross_lens_repeatability_target_um=0.3,

    # Test counts
    num_handoff_tests=50,
    num_surface_calibration_tests=20,
    num_parfocal_validation_tests=30,

    # Test ranges
    x_test_range_um=(-5000.0, 5000.0),
    y_test_range_um=(-5000.0, 5000.0),
    z_test_range_um=(-10.0, 10.0),

    # Advanced testing
    enable_temperature_tests=True,
    temperature_range_c=(20.0, 30.0),

    # Output configuration
    output_dir="qa_results",
    enable_detailed_logging=True
)

# Run comprehensive validation
qa_harness = DualLensQAHarness(qa_config)
summary = qa_harness.run_full_validation(system)

print(f"QA Status: {summary['overall_status']}")
```

### Individual QA Tests

#### 1. Handoff Performance Test

```python
from autofocus.dual_lens_qa import HandoffPerformanceTest

test = HandoffPerformanceTest()
result = test.run(system, config=qa_config)

if result.passed:
    print("✓ Handoff performance test PASSED")
    details = result.details
    print(f"  Average time: {details['performance_stats']['avg_handoff_time_ms']:.0f}ms")
    print(f"  Target met rate: {details['performance_stats']['target_met_rate']*100:.1f}%")
else:
    print("✗ Handoff performance test FAILED")
    print(f"  Error: {result.error_message}")
```

#### 2. Surface Calibration Test

```python
from autofocus.dual_lens_qa import SurfaceCalibrationTest

test = SurfaceCalibrationTest()
result = test.run(system, config=qa_config)

# Check per-lens results
for lens_id, lens_result in result.details['lens_results'].items():
    prediction_accuracy = lens_result['prediction_accuracy']
    print(f"{lens_id}: RMS error = {prediction_accuracy['rms_error_um']:.3f}μm")
```

#### 3. Parfocal Mapping Validation

```python
from autofocus.dual_lens_qa import ParfocalValidationTest

test = ParfocalValidationTest()
result = test.run(system, config=qa_config)

mapping_accuracy = result.details['mapping_accuracy']
print(f"P95 mapping error: {mapping_accuracy['p95_error_um']:.2f}μm")

consistency = result.details['round_trip_consistency']
print(f"Round-trip consistency: {consistency['avg_error_um']:.3f}μm")
```

#### 4. Temperature Compensation Test

```python
from autofocus.dual_lens_qa import TemperatureCompensationTest

test = TemperatureCompensationTest()
result = test.run(system, config=qa_config)

temp_sensitivity = result.details['temperature_sensitivity']
print(f"Thermal coefficient: {temp_sensitivity['coefficient_um_per_c']:.3f}μm/°C")
```

## ⚡ Performance Testing

### Benchmark Framework

```python
def run_performance_benchmark(system, num_tests=100):
    """Run comprehensive performance benchmark."""

    print(f"Running {num_tests}-test performance benchmark...")

    results = []
    timing_breakdown = {
        'lens_switch': [],
        'focus_move': [],
        'focus_search': [],
        'validation': []
    }

    for i in range(num_tests):
        # Random test conditions
        source_z = np.random.uniform(-5.0, 5.0)

        # Ensure clean start
        system.camera.set_active_lens(LensID.LENS_A)
        system.camera.set_focus(source_z)

        # Perform handoff
        result = system.handoff_a_to_b_optimized(source_z)
        results.append(result)

        if result.success:
            timing_breakdown['lens_switch'].append(result.lens_switch_ms)
            timing_breakdown['focus_move'].append(result.focus_move_ms)
            timing_breakdown['focus_search'].append(result.focus_search_ms)
            timing_breakdown['validation'].append(result.validation_ms)

        # Progress reporting
        if (i + 1) % 25 == 0:
            print(f"  Completed {i + 1}/{num_tests} tests...")

    return analyze_benchmark_results(results, timing_breakdown)

def analyze_benchmark_results(results, timing_breakdown):
    """Analyze benchmark results and generate report."""

    successful_results = [r for r in results if r.success]
    handoff_times = [r.elapsed_ms for r in successful_results]
    mapping_errors = [r.mapping_error_um for r in successful_results]

    analysis = {
        'summary': {
            'total_tests': len(results),
            'success_count': len(successful_results),
            'success_rate': len(successful_results) / len(results),
        },
        'performance': {
            'avg_time_ms': np.mean(handoff_times),
            'median_time_ms': np.median(handoff_times),
            'p95_time_ms': np.percentile(handoff_times, 95),
            'p99_time_ms': np.percentile(handoff_times, 99),
            'min_time_ms': np.min(handoff_times),
            'max_time_ms': np.max(handoff_times),
            'std_time_ms': np.std(handoff_times)
        },
        'accuracy': {
            'avg_error_um': np.mean(mapping_errors),
            'p95_error_um': np.percentile(mapping_errors, 95),
            'max_error_um': np.max(mapping_errors)
        },
        'timing_breakdown': {
            phase: {
                'avg_ms': np.mean(times),
                'contribution_pct': (np.mean(times) / np.mean(handoff_times) * 100) if times else 0
            }
            for phase, times in timing_breakdown.items()
        },
        'target_analysis': {
            'target_met_count': sum(1 for t in handoff_times if t <= 300),
            'target_met_rate': np.mean([t <= 300 for t in handoff_times])
        }
    }

    return analysis

# Example usage
benchmark_results = run_performance_benchmark(system, num_tests=100)
print_benchmark_report(benchmark_results)
```

### Stress Testing

```python
def run_stress_test(system, duration_minutes=60):
    """Run continuous stress test for specified duration."""

    print(f"Starting {duration_minutes}-minute stress test...")

    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)

    test_count = 0
    error_count = 0
    performance_samples = []

    while time.time() < end_time:
        try:
            # Random handoff operation
            source_z = np.random.uniform(-5.0, 5.0)

            system.camera.set_active_lens(LensID.LENS_A)
            system.camera.set_focus(source_z)

            result = system.handoff_a_to_b_optimized(source_z)

            test_count += 1

            if result.success:
                performance_samples.append({
                    'timestamp': time.time(),
                    'elapsed_ms': result.elapsed_ms,
                    'mapping_error_um': result.mapping_error_um
                })
            else:
                error_count += 1
                print(f"  Error {error_count}: {result.flags}")

            # Brief pause to prevent overheating
            time.sleep(0.1)

            # Progress reporting
            if test_count % 100 == 0:
                elapsed_min = (time.time() - start_time) / 60
                print(f"  {elapsed_min:.1f}min: {test_count} tests, {error_count} errors")

        except Exception as e:
            error_count += 1
            print(f"  Exception {error_count}: {e}")

    # Analyze stress test results
    total_duration = time.time() - start_time

    stress_results = {
        'duration_minutes': total_duration / 60,
        'total_tests': test_count,
        'error_count': error_count,
        'success_rate': (test_count - error_count) / test_count,
        'tests_per_minute': test_count / (total_duration / 60),
        'performance_stability': analyze_performance_stability(performance_samples)
    }

    return stress_results

def analyze_performance_stability(samples):
    """Analyze performance stability over time."""

    if len(samples) < 10:
        return {'status': 'insufficient_data'}

    times = [s['elapsed_ms'] for s in samples]
    errors = [s['mapping_error_um'] for s in samples]

    # Check for performance drift
    time_chunks = np.array_split(times, 10)  # Divide into 10 chunks
    chunk_means = [np.mean(chunk) for chunk in time_chunks]

    # Linear regression to detect drift
    x = np.arange(len(chunk_means))
    drift_slope = np.polyfit(x, chunk_means, 1)[0]

    stability_analysis = {
        'avg_performance_ms': np.mean(times),
        'performance_std_ms': np.std(times),
        'performance_drift_ms_per_chunk': drift_slope,
        'avg_accuracy_um': np.mean(errors),
        'accuracy_std_um': np.std(errors),
        'stability_rating': 'excellent' if abs(drift_slope) < 1.0 else 'good' if abs(drift_slope) < 5.0 else 'poor'
    }

    return stability_analysis
```

## 🔧 Unit Testing

### Test Structure

```python
import unittest
import numpy as np
from unittest.mock import Mock, patch

class TestOptimizedDualLensAutofocus(unittest.TestCase):
    """Unit tests for OptimizedDualLensAutofocus class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_camera = Mock(spec=CameraInterface)
        self.mock_stage = Mock()
        self.mock_illumination = Mock()

        # Configure mock responses
        self.mock_camera.get_active_lens.return_value = LensID.LENS_A
        self.mock_camera.get_focus.return_value = 0.0
        self.mock_camera.get_frame.return_value = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)

        self.mock_stage.get_xy.return_value = (0.0, 0.0)

        # Create test lens profiles
        self.lens_a = create_test_lens_profile(LensID.LENS_A)
        self.lens_b = create_test_lens_profile(LensID.LENS_B)

        # Create test parfocal mapping
        self.mapping = create_test_parfocal_mapping()

    def test_system_initialization(self):
        """Test system initialization."""
        system = OptimizedDualLensAutofocus(
            camera=self.mock_camera,
            stage_controller=self.mock_stage,
            illumination=self.mock_illumination,
            lens_a_profile=self.lens_a,
            lens_b_profile=self.lens_b,
            parfocal_mapping=self.mapping,
            optimization_level=OptimizationLevel.FAST
        )

        self.assertEqual(system.optimization_level, OptimizationLevel.FAST)
        self.assertIn(LensID.LENS_A, system.profiles)
        self.assertIn(LensID.LENS_B, system.profiles)

    def test_lens_profile_optimization(self):
        """Test lens profile optimization."""
        system = OptimizedDualLensAutofocus(
            camera=self.mock_camera,
            stage_controller=self.mock_stage,
            illumination=self.mock_illumination,
            lens_a_profile=self.lens_a,
            lens_b_profile=self.lens_b,
            parfocal_mapping=self.mapping,
            optimization_level=OptimizationLevel.ULTRA_FAST
        )

        # Check that profiles were optimized
        optimized_lens_a = system.profiles[LensID.LENS_A]
        self.assertLess(optimized_lens_a.settle_time_ms, self.lens_a.settle_time_ms)

    def test_focus_cache_functionality(self):
        """Test focus caching system."""
        cache = FocusCache()

        # Test caching
        cache.cache_focus(1000.0, 2000.0, LensID.LENS_A, 2.5)
        cached_value = cache.get_cached_focus(1000.0, 2000.0, LensID.LENS_A)
        self.assertEqual(cached_value, 2.5)

        # Test cache miss
        cached_value = cache.get_cached_focus(5000.0, 6000.0, LensID.LENS_A)
        self.assertIsNone(cached_value)

        # Test tolerance
        cached_value = cache.get_cached_focus(1005.0, 2005.0, LensID.LENS_A)  # Within tolerance
        self.assertEqual(cached_value, 2.5)

    def test_parfocal_mapping_cache(self):
        """Test parfocal mapping caching."""
        cache = FocusCache()

        # Test mapping cache
        cache.cache_mapping(2.0, 23.0, 4.1)
        cached_mapping = cache.get_cached_mapping(2.0, 23.0)
        self.assertEqual(cached_mapping, 4.1)

        # Test temperature tolerance
        cached_mapping = cache.get_cached_mapping(2.0, 23.3)  # Within 0.5°C tolerance
        self.assertEqual(cached_mapping, 4.1)

    @patch('time.time')
    def test_handoff_timing(self, mock_time):
        """Test handoff timing measurement."""
        # Mock time progression
        mock_time.side_effect = [0.0, 0.1]  # 100ms handoff

        system = OptimizedDualLensAutofocus(
            camera=self.mock_camera,
            stage_controller=self.mock_stage,
            illumination=self.mock_illumination,
            lens_a_profile=self.lens_a,
            lens_b_profile=self.lens_b,
            parfocal_mapping=self.mapping
        )

        result = system.handoff_a_to_b_optimized(2.0)

        self.assertTrue(result.success)
        self.assertEqual(result.elapsed_ms, 100.0)

    def tearDown(self):
        """Clean up test fixtures."""
        pass

def create_test_lens_profile(lens_id):
    """Create test lens profile."""
    config = AutofocusConfig.create_blood_smear_config()

    return LensProfile(
        lens_id=lens_id,
        name=f"Test Lens {lens_id.value}",
        magnification=20.0 if lens_id == LensID.LENS_A else 60.0,
        z_range_um=(-20.0, 20.0),
        af_config=config,
        focus_speed_um_per_s=300.0,
        settle_time_ms=10.0
    )

def create_test_parfocal_mapping():
    """Create test parfocal mapping."""
    mapping = EnhancedParfocalMapping()
    mapping.coefficients = [2.0, 0.95, 0.0, 0.0]  # Simple linear mapping
    mapping.rms_error_um = 0.1
    return mapping

# Run unit tests
if __name__ == '__main__':
    unittest.main()
```

## 🔗 Integration Testing

### Hardware Integration Tests

```python
class TestHardwareIntegration(unittest.TestCase):
    """Integration tests with actual hardware."""

    @classmethod
    def setUpClass(cls):
        """Set up hardware connections."""
        # Initialize actual hardware (adapt for your system)
        cls.camera = YourActualCameraController()
        cls.stage = YourActualStageController()
        cls.illumination = YourActualIlluminationController()

        # Load production configuration
        cls.lens_a, cls.lens_b = load_production_lens_profiles()
        cls.mapping = load_production_parfocal_mapping()

    def test_camera_interface_compliance(self):
        """Test camera interface compliance."""
        # Test all required methods exist and work
        self.assertTrue(hasattr(self.camera, 'get_frame'))
        self.assertTrue(hasattr(self.camera, 'set_active_lens'))
        self.assertTrue(hasattr(self.camera, 'get_active_lens'))
        self.assertTrue(hasattr(self.camera, 'set_focus'))
        self.assertTrue(hasattr(self.camera, 'get_focus'))

        # Test method functionality
        frame = self.camera.get_frame()
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(len(frame.shape), 3)  # Height, width, channels

        # Test lens switching
        self.camera.set_active_lens(LensID.LENS_A)
        self.assertEqual(self.camera.get_active_lens(), LensID.LENS_A)

        self.camera.set_active_lens(LensID.LENS_B)
        self.assertEqual(self.camera.get_active_lens(), LensID.LENS_B)

        # Test focus control
        self.camera.set_focus(5.0)
        focus_pos = self.camera.get_focus()
        self.assertAlmostEqual(focus_pos, 5.0, delta=0.1)

    def test_system_calibration_accuracy(self):
        """Test system calibration accuracy."""
        # Create system
        system = OptimizedDualLensAutofocus(
            camera=self.camera,
            stage_controller=self.stage,
            illumination=self.illumination,
            lens_a_profile=self.lens_a,
            lens_b_profile=self.lens_b,
            parfocal_mapping=self.mapping
        )

        # Test parfocal mapping accuracy
        test_positions = [-3.0, -1.0, 0.0, 1.0, 3.0]

        for z_a in test_positions:
            # Set Lens-A position
            self.camera.set_active_lens(LensID.LENS_A)
            self.camera.set_focus(z_a)

            # Perform handoff
            result = system.handoff_a_to_b_optimized(z_a)

            self.assertTrue(result.success, f"Handoff failed at z_a={z_a}")
            self.assertLess(result.elapsed_ms, 500.0, f"Handoff too slow: {result.elapsed_ms}ms")
            self.assertLess(result.mapping_error_um, 2.0, f"Mapping error too large: {result.mapping_error_um}μm")

    def test_thermal_stability(self):
        """Test thermal stability over time."""
        # This test should run for an extended period (30+ minutes)
        # to verify thermal stability

        system = OptimizedDualLensAutofocus(
            camera=self.camera,
            stage_controller=self.stage,
            illumination=self.illumination,
            lens_a_profile=self.lens_a,
            lens_b_profile=self.lens_b,
            parfocal_mapping=self.mapping
        )

        # Record initial performance
        initial_results = []
        for _ in range(10):
            result = system.handoff_a_to_b_optimized(0.0)
            if result.success:
                initial_results.append(result.mapping_error_um)

        initial_avg_error = np.mean(initial_results)

        # Wait for thermal drift (in production, this would be longer)
        time.sleep(300)  # 5 minutes

        # Record final performance
        final_results = []
        for _ in range(10):
            result = system.handoff_a_to_b_optimized(0.0)
            if result.success:
                final_results.append(result.mapping_error_um)

        final_avg_error = np.mean(final_results)

        # Check thermal drift
        thermal_drift = abs(final_avg_error - initial_avg_error)
        self.assertLess(thermal_drift, 0.2, f"Thermal drift too large: {thermal_drift}μm")
```

## 📈 Test Results Analysis

### Automated Report Generation

```python
def generate_test_report(qa_results, benchmark_results, output_dir="test_reports"):
    """Generate comprehensive test report."""

    report_time = time.strftime("%Y%m%d_%H%M%S")
    report_dir = Path(output_dir) / f"test_report_{report_time}"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Generate HTML report
    html_report = generate_html_report(qa_results, benchmark_results)
    with open(report_dir / "test_report.html", 'w') as f:
        f.write(html_report)

    # Generate JSON summary
    json_summary = {
        'timestamp': report_time,
        'qa_results': qa_results,
        'benchmark_results': benchmark_results,
        'overall_status': determine_overall_status(qa_results, benchmark_results)
    }

    with open(report_dir / "test_summary.json", 'w') as f:
        json.dump(json_summary, f, indent=2, default=str)

    # Generate performance plots (if matplotlib available)
    try:
        generate_performance_plots(benchmark_results, report_dir)
    except ImportError:
        print("Matplotlib not available - skipping plots")

    print(f"Test report generated: {report_dir}")
    return report_dir

def determine_overall_status(qa_results, benchmark_results):
    """Determine overall test status."""

    # Check QA results
    qa_passed = qa_results.get('overall_status') == 'PASS'

    # Check performance benchmarks
    perf_passed = True
    if 'target_analysis' in benchmark_results:
        target_met_rate = benchmark_results['target_analysis']['target_met_rate']
        perf_passed = target_met_rate >= 0.95

    if qa_passed and perf_passed:
        return 'PRODUCTION_READY'
    elif qa_passed or perf_passed:
        return 'NEEDS_OPTIMIZATION'
    else:
        return 'SYSTEM_FAILURE'

def generate_performance_plots(benchmark_results, output_dir):
    """Generate performance visualization plots."""
    import matplotlib.pyplot as plt

    # Performance distribution histogram
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    # Plot handoff time distribution
    # (Implementation depends on your data structure)

    plt.subplot(2, 2, 2)
    # Plot accuracy distribution

    plt.subplot(2, 2, 3)
    # Plot timing breakdown pie chart

    plt.subplot(2, 2, 4)
    # Plot performance over time

    plt.tight_layout()
    plt.savefig(output_dir / "performance_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
```

## 🔄 Continuous Testing

### Automated Testing Pipeline

```python
class ContinuousTestingPipeline:
    """Automated testing pipeline for continuous validation."""

    def __init__(self, system, config):
        self.system = system
        self.config = config
        self.test_scheduler = None
        self.test_history = []

    def start_continuous_testing(self, interval_hours=24):
        """Start continuous testing pipeline."""
        print(f"Starting continuous testing (every {interval_hours} hours)")

        def run_scheduled_tests():
            while True:
                try:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Running scheduled tests...")

                    # Run QA validation
                    qa_results = self.run_qa_validation()

                    # Run performance benchmark
                    benchmark_results = self.run_performance_benchmark()

                    # Store results
                    test_result = {
                        'timestamp': time.time(),
                        'qa_results': qa_results,
                        'benchmark_results': benchmark_results
                    }

                    self.test_history.append(test_result)

                    # Check for regressions
                    self.check_for_regressions(test_result)

                    # Generate report
                    generate_test_report(qa_results, benchmark_results)

                    print("✓ Scheduled testing completed")

                except Exception as e:
                    print(f"✗ Scheduled testing failed: {e}")
                    self.send_alert("TESTING_FAILURE", str(e))

                # Sleep until next test
                time.sleep(interval_hours * 3600)

        self.test_scheduler = threading.Thread(target=run_scheduled_tests, daemon=True)
        self.test_scheduler.start()

    def run_qa_validation(self):
        """Run QA validation."""
        qa_config = DualLensQAConfig(
            num_handoff_tests=20,  # Reduced for continuous testing
            handoff_time_target_ms=300.0,
            mapping_accuracy_target_um=1.0
        )

        qa_harness = DualLensQAHarness(qa_config)
        return qa_harness.run_full_validation(self.system)

    def run_performance_benchmark(self):
        """Run performance benchmark."""
        return run_performance_benchmark(self.system, num_tests=50)

    def check_for_regressions(self, current_result):
        """Check for performance regressions."""
        if len(self.test_history) < 2:
            return  # Need historical data for comparison

        # Compare with previous result
        previous_result = self.test_history[-2]

        # Check QA regression
        current_qa = current_result['qa_results']['overall_status']
        previous_qa = previous_result['qa_results']['overall_status']

        if current_qa != 'PASS' and previous_qa == 'PASS':
            self.send_alert("QA_REGRESSION", f"QA status changed from PASS to {current_qa}")

        # Check performance regression
        current_perf = current_result['benchmark_results']['performance']
        previous_perf = previous_result['benchmark_results']['performance']

        current_avg = current_perf['avg_time_ms']
        previous_avg = previous_perf['avg_time_ms']

        performance_change = (current_avg - previous_avg) / previous_avg

        if performance_change > 0.2:  # 20% performance degradation
            self.send_alert("PERFORMANCE_REGRESSION",
                          f"Performance degraded by {performance_change*100:.1f}%")

    def send_alert(self, alert_type, message):
        """Send testing alert."""
        alert = {
            'timestamp': time.time(),
            'type': alert_type,
            'message': message
        }

        print(f"🚨 TESTING ALERT [{alert_type}]: {message}")

        # TODO: Integrate with your alerting system
        # send_email_alert(alert)
        # send_slack_notification(alert)

    def get_test_history_summary(self, days=7):
        """Get test history summary for specified period."""
        cutoff_time = time.time() - (days * 24 * 3600)
        recent_tests = [t for t in self.test_history if t['timestamp'] >= cutoff_time]

        if not recent_tests:
            return {'status': 'no_data'}

        qa_pass_rate = np.mean([
            1 if t['qa_results']['overall_status'] == 'PASS' else 0
            for t in recent_tests
        ])

        avg_performance = np.mean([
            t['benchmark_results']['performance']['avg_time_ms']
            for t in recent_tests
        ])

        return {
            'period_days': days,
            'test_count': len(recent_tests),
            'qa_pass_rate': qa_pass_rate,
            'avg_performance_ms': avg_performance,
            'system_health': 'GOOD' if qa_pass_rate >= 0.95 else 'DEGRADED'
        }

# Usage example
pipeline = ContinuousTestingPipeline(system, config)
pipeline.start_continuous_testing(interval_hours=12)  # Test every 12 hours
```

## 📊 Test Metrics and KPIs

### Key Performance Indicators

```python
class TestKPITracker:
    """Track key performance indicators for testing."""

    def __init__(self):
        self.kpi_history = []

    def calculate_kpis(self, test_results):
        """Calculate KPIs from test results."""

        kpis = {
            'timestamp': time.time(),

            # Performance KPIs
            'avg_handoff_time_ms': test_results['performance']['avg_time_ms'],
            'p95_handoff_time_ms': test_results['performance']['p95_time_ms'],
            'handoff_success_rate': test_results['summary']['success_rate'],
            'target_achievement_rate': test_results['target_analysis']['target_met_rate'],

            # Accuracy KPIs
            'avg_mapping_error_um': test_results['accuracy']['avg_error_um'],
            'p95_mapping_error_um': test_results['accuracy']['p95_error_um'],

            # System health KPIs
            'system_availability': self.calculate_availability(),
            'mtbf_hours': self.calculate_mtbf(),
            'performance_stability': self.calculate_stability()
        }

        self.kpi_history.append(kpis)
        return kpis

    def calculate_availability(self):
        """Calculate system availability percentage."""
        # Implementation depends on your failure tracking
        return 99.5  # Example: 99.5% availability

    def calculate_mtbf(self):
        """Calculate Mean Time Between Failures."""
        # Implementation depends on your failure tracking
        return 168.0  # Example: 168 hours (1 week) MTBF

    def calculate_stability(self):
        """Calculate performance stability metric."""
        if len(self.kpi_history) < 10:
            return 1.0  # Assume stable with insufficient data

        recent_times = [kpi['avg_handoff_time_ms'] for kpi in self.kpi_history[-10:]]
        stability = 1.0 / (1.0 + np.std(recent_times) / np.mean(recent_times))
        return stability

    def get_kpi_trends(self, days=30):
        """Get KPI trends over specified period."""
        cutoff_time = time.time() - (days * 24 * 3600)
        recent_kpis = [kpi for kpi in self.kpi_history if kpi['timestamp'] >= cutoff_time]

        if len(recent_kpis) < 2:
            return {'status': 'insufficient_data'}

        trends = {}
        for metric in ['avg_handoff_time_ms', 'target_achievement_rate', 'avg_mapping_error_um']:
            values = [kpi[metric] for kpi in recent_kpis]
            timestamps = [kpi['timestamp'] for kpi in recent_kpis]

            # Calculate trend slope
            x = np.array(timestamps)
            y = np.array(values)
            slope = np.polyfit(x, y, 1)[0]

            trends[metric] = {
                'current_value': values[-1],
                'trend_slope': slope,
                'trend_direction': 'improving' if slope < 0 else 'degrading' if slope > 0 else 'stable'
            }

        return trends
```

---

This comprehensive testing guide provides everything needed to validate, benchmark, and continuously monitor the dual-lens autofocus system. The framework ensures production readiness through systematic testing at all levels from individual components to complete system integration.