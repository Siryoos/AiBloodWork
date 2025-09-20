# Performance Profiling Guide

## Overview

This guide provides comprehensive instructions for profiling and optimizing the dual-lens autofocus system performance in production environments. Learn how to identify bottlenecks, measure system performance, and optimize for your specific use case.

## Table of Contents

1. [Performance Metrics](#performance-metrics)
2. [Profiling Tools](#profiling-tools)
3. [Benchmarking Framework](#benchmarking-framework)
4. [Performance Analysis](#performance-analysis)
5. [Optimization Strategies](#optimization-strategies)
6. [Production Monitoring](#production-monitoring)
7. [Troubleshooting](#troubleshooting)

## Performance Metrics

### Key Performance Indicators

The dual-lens autofocus system tracks several critical performance metrics:

#### Timing Metrics
- **Handoff Time**: Total time for A→B lens transition (target: ≤300ms)
- **Lens Switch Time**: Physical lens switching duration (typical: <50ms)
- **Focus Move Time**: Focus positioning duration (typical: <80ms)
- **Focus Search Time**: Autofocus search duration (typical: <100ms)
- **Validation Time**: Mapping accuracy validation (typical: <20ms)

#### Accuracy Metrics
- **Mapping Error**: Parfocal mapping accuracy (target: ≤1.0μm)
- **RMS Error**: Root-mean-square calibration error (target: ≤0.1μm)
- **Repeatability**: Position repeatability standard deviation
- **Temperature Stability**: Thermal drift (target: ≤0.05μm/°C)

#### System Metrics
- **Cache Hit Rate**: Focus prediction cache efficiency (target: >50%)
- **Success Rate**: Successful handoff percentage (target: >99%)
- **Throughput**: Handoffs per minute in production
- **Resource Usage**: CPU, memory, and thread utilization

### Performance Targets

| Metric | Target | Excellent | Good | Needs Improvement |
|--------|--------|-----------|------|-------------------|
| Handoff Time | ≤300ms | <150ms | <250ms | >300ms |
| Mapping Error | ≤1.0μm | <0.1μm | <0.5μm | >1.0μm |
| Success Rate | >99% | >99.9% | >99.5% | <99% |
| Cache Hit Rate | >50% | >70% | >60% | <50% |

## Profiling Tools

### Built-in Performance Monitoring

The system includes comprehensive built-in profiling capabilities:

```python
from autofocus.dual_lens_optimized import create_optimized_dual_lens_system
from autofocus.dual_lens_optimized import OptimizationLevel

# Create system with performance monitoring
system = create_optimized_dual_lens_system(
    camera=camera,
    stage_controller=stage,
    illumination=illumination,
    lens_a_profile=lens_a,
    lens_b_profile=lens_b,
    parfocal_mapping=mapping,
    optimization_level=OptimizationLevel.FAST,
    enable_telemetry=True  # Enable detailed performance tracking
)

# Perform operations
result = system.handoff_a_to_b_optimized(source_z=5.0)

# Get detailed performance data
stats = system.get_performance_stats()
print(f"Average handoff time: {stats['avg_handoff_time_ms']:.1f}ms")
print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
print(f"Success rate: {stats['success_rate']:.1%}")
```

### QA Framework Profiling

Use the comprehensive QA framework for detailed performance analysis:

```python
from autofocus.dual_lens_qa import DualLensQA, QATestSuite

# Create QA framework
qa = DualLensQA(system, enable_detailed_timing=True)

# Run performance test suite
results = qa.run_test_suite(QATestSuite.PERFORMANCE)

# Analyze timing breakdown
timing_report = results['performance']['timing_analysis']
print(f"Lens switch: {timing_report['lens_switch_avg_ms']:.1f}ms")
print(f"Focus move: {timing_report['focus_move_avg_ms']:.1f}ms")
print(f"Focus search: {timing_report['focus_search_avg_ms']:.1f}ms")
```

### External Profiling Tools

#### Python cProfile Integration

```python
import cProfile
import pstats
from io import StringIO

def profile_handoff_operation():
    """Profile a complete handoff operation."""
    pr = cProfile.Profile()
    pr.enable()

    # Perform handoff operation
    result = system.handoff_a_to_b_optimized(source_z=3.0)

    pr.disable()

    # Analyze results
    s = StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()

    print(s.getvalue())
    return result

# Run profiling
profile_handoff_operation()
```

#### Memory Profiling with memory_profiler

```python
from memory_profiler import profile

@profile
def memory_intensive_operation():
    """Profile memory usage during intensive operations."""
    # Perform multiple handoffs
    for i in range(100):
        z_source = np.random.uniform(-5, 5)
        result = system.handoff_a_to_b_optimized(z_source)

        # Force garbage collection periodically
        if i % 10 == 0:
            import gc
            gc.collect()

# Run memory profiling
memory_intensive_operation()
```

## Benchmarking Framework

### Comprehensive Benchmark Suite

Create comprehensive benchmarks to evaluate system performance:

```python
import time
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    test_name: str
    num_iterations: int
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    p95_time_ms: float
    success_rate: float
    avg_error_um: float

class PerformanceBenchmark:
    """Comprehensive performance benchmark suite."""

    def __init__(self, system):
        self.system = system
        self.results: List[BenchmarkResult] = []

    def benchmark_handoff_performance(self, num_iterations: int = 100) -> BenchmarkResult:
        """Benchmark handoff performance across range of positions."""

        times = []
        errors = []
        successes = 0

        for i in range(num_iterations):
            # Random source position
            z_source = np.random.uniform(-5.0, 5.0)

            # Time the handoff
            start_time = time.time()
            result = self.system.handoff_a_to_b_optimized(z_source)
            elapsed_ms = (time.time() - start_time) * 1000

            times.append(elapsed_ms)

            if result.success:
                successes += 1
                errors.append(result.mapping_error_um)

            # Progress indicator
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i + 1}/{num_iterations}")

        # Calculate statistics
        benchmark_result = BenchmarkResult(
            test_name="Handoff Performance",
            num_iterations=num_iterations,
            avg_time_ms=np.mean(times),
            min_time_ms=np.min(times),
            max_time_ms=np.max(times),
            p95_time_ms=np.percentile(times, 95),
            success_rate=successes / num_iterations,
            avg_error_um=np.mean(errors) if errors else float('inf')
        )

        self.results.append(benchmark_result)
        return benchmark_result

    def benchmark_focus_accuracy(self, num_positions: int = 50) -> BenchmarkResult:
        """Benchmark focus positioning accuracy."""

        times = []
        errors = []

        # Test across full range
        z_positions = np.linspace(-8, 8, num_positions)

        for z_target in z_positions:
            # Time focus operation
            start_time = time.time()
            self.system.camera.set_focus(z_target)
            elapsed_ms = (time.time() - start_time) * 1000

            # Measure actual position (simulated)
            z_actual = self.system.camera.get_focus()
            error_um = abs(z_actual - z_target)

            times.append(elapsed_ms)
            errors.append(error_um)

        benchmark_result = BenchmarkResult(
            test_name="Focus Accuracy",
            num_iterations=num_positions,
            avg_time_ms=np.mean(times),
            min_time_ms=np.min(times),
            max_time_ms=np.max(times),
            p95_time_ms=np.percentile(times, 95),
            success_rate=1.0,  # All focus operations succeed
            avg_error_um=np.mean(errors)
        )

        self.results.append(benchmark_result)
        return benchmark_result

    def benchmark_cache_efficiency(self, num_operations: int = 200) -> Dict:
        """Benchmark cache hit rate and efficiency."""

        # Clear cache
        self.system.camera._focus_prediction_cache.clear()

        initial_stats = self.system.get_performance_stats()

        # Perform operations with some repetition to test caching
        for i in range(num_operations):
            if i % 3 == 0:
                # Repeat some positions to test cache hits
                z_target = 2.0 + (i % 5) * 1.0
            else:
                z_target = np.random.uniform(-5, 5)

            self.system.camera.set_focus(z_target)

        final_stats = self.system.get_performance_stats()

        cache_analysis = {
            'total_operations': num_operations,
            'cache_hit_rate': final_stats['cache_hit_rate'],
            'cache_size': len(self.system.camera._focus_prediction_cache),
            'avg_lookup_time_us': final_stats.get('avg_cache_lookup_time_us', 0)
        }

        return cache_analysis

    def run_full_benchmark(self) -> Dict:
        """Run complete benchmark suite."""

        print("🚀 Starting comprehensive performance benchmark...")

        # 1. Handoff performance
        print("\n1. Benchmarking handoff performance...")
        handoff_result = self.benchmark_handoff_performance(100)

        # 2. Focus accuracy
        print("\n2. Benchmarking focus accuracy...")
        focus_result = self.benchmark_focus_accuracy(50)

        # 3. Cache efficiency
        print("\n3. Benchmarking cache efficiency...")
        cache_result = self.benchmark_cache_efficiency(200)

        # 4. System statistics
        system_stats = self.system.get_performance_stats()

        return {
            'handoff_performance': handoff_result,
            'focus_accuracy': focus_result,
            'cache_efficiency': cache_result,
            'system_statistics': system_stats,
            'benchmark_timestamp': time.time()
        }

    def generate_report(self, results: Dict) -> str:
        """Generate human-readable benchmark report."""

        report = []
        report.append("=" * 60)
        report.append("DUAL-LENS AUTOFOCUS PERFORMANCE REPORT")
        report.append("=" * 60)

        # Handoff performance
        handoff = results['handoff_performance']
        report.append(f"\n📊 Handoff Performance:")
        report.append(f"  Average time: {handoff.avg_time_ms:.1f}ms")
        report.append(f"  P95 time: {handoff.p95_time_ms:.1f}ms")
        report.append(f"  Success rate: {handoff.success_rate:.1%}")
        report.append(f"  Average error: {handoff.avg_error_um:.3f}μm")

        # Focus accuracy
        focus = results['focus_accuracy']
        report.append(f"\n🎯 Focus Accuracy:")
        report.append(f"  Average time: {focus.avg_time_ms:.1f}ms")
        report.append(f"  P95 time: {focus.p95_time_ms:.1f}ms")
        report.append(f"  Average error: {focus.avg_error_um:.3f}μm")

        # Cache efficiency
        cache = results['cache_efficiency']
        report.append(f"\n⚡ Cache Efficiency:")
        report.append(f"  Hit rate: {cache['cache_hit_rate']:.1%}")
        report.append(f"  Cache size: {cache['cache_size']} entries")

        # Performance assessment
        report.append(f"\n✅ Performance Assessment:")

        # Check targets
        if handoff.avg_time_ms <= 300:
            report.append(f"  ✓ Handoff time target met ({handoff.avg_time_ms:.1f}ms ≤ 300ms)")
        else:
            report.append(f"  ❌ Handoff time exceeds target ({handoff.avg_time_ms:.1f}ms > 300ms)")

        if handoff.avg_error_um <= 1.0:
            report.append(f"  ✓ Mapping accuracy target met ({handoff.avg_error_um:.3f}μm ≤ 1.0μm)")
        else:
            report.append(f"  ❌ Mapping accuracy exceeds target ({handoff.avg_error_um:.3f}μm > 1.0μm)")

        if handoff.success_rate >= 0.99:
            report.append(f"  ✓ Success rate target met ({handoff.success_rate:.1%} ≥ 99%)")
        else:
            report.append(f"  ❌ Success rate below target ({handoff.success_rate:.1%} < 99%)")

        return "\n".join(report)

# Usage example
benchmark = PerformanceBenchmark(system)
results = benchmark.run_full_benchmark()
report = benchmark.generate_report(results)
print(report)
```

## Performance Analysis

### Timing Analysis

#### Detailed Timing Breakdown

```python
def analyze_timing_breakdown(system, num_samples: int = 50):
    """Analyze detailed timing breakdown of handoff operations."""

    timing_data = {
        'lens_switch': [],
        'focus_move': [],
        'focus_search': [],
        'validation': [],
        'total': []
    }

    for i in range(num_samples):
        z_source = np.random.uniform(-4, 4)

        # Detailed timing measurement
        start_time = time.time()
        result = system.handoff_a_to_b_optimized(z_source)
        total_time = (time.time() - start_time) * 1000

        if result.success:
            timing_data['lens_switch'].append(result.lens_switch_ms)
            timing_data['focus_move'].append(result.focus_move_ms)
            timing_data['focus_search'].append(result.focus_search_ms)
            timing_data['validation'].append(result.validation_ms)
            timing_data['total'].append(total_time)

    # Statistical analysis
    analysis = {}
    for phase, times in timing_data.items():
        if times:
            analysis[phase] = {
                'mean_ms': np.mean(times),
                'std_ms': np.std(times),
                'min_ms': np.min(times),
                'max_ms': np.max(times),
                'p95_ms': np.percentile(times, 95),
                'contribution_pct': (np.mean(times) / np.mean(timing_data['total'])) * 100
            }

    return analysis

# Run timing analysis
timing_analysis = analyze_timing_breakdown(system)

# Print results
print("Timing Breakdown Analysis:")
print(f"{'Phase':<15} {'Mean':<8} {'Std':<8} {'P95':<8} {'Contribution':<12}")
print("-" * 60)

for phase, stats in timing_analysis.items():
    print(f"{phase:<15} {stats['mean_ms']:<8.1f} {stats['std_ms']:<8.1f} "
          f"{stats['p95_ms']:<8.1f} {stats['contribution_pct']:<12.1f}%")
```

#### Bottleneck Identification

```python
def identify_bottlenecks(timing_analysis):
    """Identify performance bottlenecks from timing data."""

    bottlenecks = []

    # Sort phases by contribution
    phases_by_contribution = sorted(
        timing_analysis.items(),
        key=lambda x: x[1]['contribution_pct'],
        reverse=True
    )

    print("\n🔍 Bottleneck Analysis:")
    print(f"{'Rank':<5} {'Phase':<15} {'Contribution':<12} {'Assessment':<15}")
    print("-" * 55)

    for rank, (phase, stats) in enumerate(phases_by_contribution, 1):
        contribution = stats['contribution_pct']

        if contribution > 40:
            assessment = "Major bottleneck"
            bottlenecks.append((phase, "major"))
        elif contribution > 25:
            assessment = "Minor bottleneck"
            bottlenecks.append((phase, "minor"))
        else:
            assessment = "Optimized"

        print(f"{rank:<5} {phase:<15} {contribution:<12.1f}% {assessment:<15}")

    return bottlenecks

# Identify bottlenecks
bottlenecks = identify_bottlenecks(timing_analysis)
```

### Accuracy Analysis

#### Mapping Error Distribution

```python
def analyze_mapping_accuracy(system, num_tests: int = 100):
    """Analyze mapping accuracy across different positions."""

    positions = np.linspace(-6, 6, num_tests)
    errors = []

    for z_a in positions:
        # Perform handoff
        result = system.handoff_a_to_b_optimized(z_a)

        if result.success:
            errors.append(result.mapping_error_um)

    if errors:
        accuracy_stats = {
            'mean_error_um': np.mean(errors),
            'std_error_um': np.std(errors),
            'max_error_um': np.max(errors),
            'rms_error_um': np.sqrt(np.mean(np.array(errors)**2)),
            'p95_error_um': np.percentile(errors, 95),
            'positions_tested': len(errors)
        }

        # Error distribution analysis
        excellent_count = sum(1 for e in errors if e < 0.1)
        good_count = sum(1 for e in errors if 0.1 <= e < 0.5)
        poor_count = sum(1 for e in errors if e >= 0.5)

        accuracy_stats['excellent_rate'] = excellent_count / len(errors)
        accuracy_stats['good_rate'] = good_count / len(errors)
        accuracy_stats['poor_rate'] = poor_count / len(errors)

        return accuracy_stats

    return None

# Analyze accuracy
accuracy_stats = analyze_mapping_accuracy(system)

if accuracy_stats:
    print("\n🎯 Mapping Accuracy Analysis:")
    print(f"  RMS error: {accuracy_stats['rms_error_um']:.3f}μm")
    print(f"  Mean error: {accuracy_stats['mean_error_um']:.3f}μm")
    print(f"  Max error: {accuracy_stats['max_error_um']:.3f}μm")
    print(f"  P95 error: {accuracy_stats['p95_error_um']:.3f}μm")
    print(f"\n  Error Distribution:")
    print(f"    Excellent (<0.1μm): {accuracy_stats['excellent_rate']:.1%}")
    print(f"    Good (0.1-0.5μm): {accuracy_stats['good_rate']:.1%}")
    print(f"    Poor (≥0.5μm): {accuracy_stats['poor_rate']:.1%}")
```

## Optimization Strategies

### 1. Optimization Level Selection

Choose the appropriate optimization level based on requirements:

```python
# Ultra-fast for high-throughput production
system_ultra_fast = create_optimized_dual_lens_system(
    optimization_level=OptimizationLevel.ULTRA_FAST
)

# Fast for balanced performance/accuracy
system_fast = create_optimized_dual_lens_system(
    optimization_level=OptimizationLevel.FAST
)

# Standard for maximum accuracy
system_standard = create_optimized_dual_lens_system(
    optimization_level=OptimizationLevel.STANDARD
)
```

### 2. Cache Optimization

Optimize focus prediction caching:

```python
# Configure cache parameters
system.camera._focus_prediction_cache.maxsize = 100  # Increase cache size
system.camera._cache_hit_threshold_um = 0.05  # Tighter hit threshold

# Pre-populate cache with common positions
common_positions = np.linspace(-5, 5, 21)
for z in common_positions:
    system.camera.set_focus(z)  # Populate cache
```

### 3. Thermal Optimization

Optimize temperature compensation:

```python
# Enhanced temperature monitoring
def optimize_thermal_compensation(mapping, temperature_samples):
    """Optimize thermal compensation coefficients."""

    # Collect temperature-dependent calibration data
    thermal_data = []

    for temp in temperature_samples:
        # Simulate temperature-dependent measurements
        calibration_points = generate_calibration_data_at_temperature(temp)
        thermal_data.extend(calibration_points)

    # Recalibrate with thermal data
    result = mapping.calibrate_enhanced(thermal_data)

    return result

# Run thermal optimization
temp_range = [18, 20, 22, 24, 26, 28]  # °C
thermal_result = optimize_thermal_compensation(mapping, temp_range)
```

### 4. Concurrent Operation Tuning

Optimize thread pool and concurrent operations:

```python
# Tune thread pool size
system.camera.executor = ThreadPoolExecutor(max_workers=4)  # Increase workers

# Enable aggressive concurrent operations
system.enable_concurrent_focus_prediction = True
system.enable_background_validation = True
```

## Production Monitoring

### Real-time Performance Monitoring

```python
class ProductionMonitor:
    """Real-time production performance monitoring."""

    def __init__(self, system):
        self.system = system
        self.metrics = {
            'handoff_times': [],
            'mapping_errors': [],
            'success_count': 0,
            'failure_count': 0,
            'start_time': time.time()
        }

    def log_handoff(self, result):
        """Log handoff operation result."""

        if result.success:
            self.metrics['success_count'] += 1
            self.metrics['handoff_times'].append(result.elapsed_ms)
            self.metrics['mapping_errors'].append(result.mapping_error_um)
        else:
            self.metrics['failure_count'] += 1

        # Maintain rolling window
        if len(self.metrics['handoff_times']) > 1000:
            self.metrics['handoff_times'] = self.metrics['handoff_times'][-500:]
            self.metrics['mapping_errors'] = self.metrics['mapping_errors'][-500:]

    def get_current_performance(self):
        """Get current performance statistics."""

        total_ops = self.metrics['success_count'] + self.metrics['failure_count']
        runtime_hours = (time.time() - self.metrics['start_time']) / 3600

        if self.metrics['handoff_times']:
            stats = {
                'total_operations': total_ops,
                'success_rate': self.metrics['success_count'] / total_ops if total_ops > 0 else 0,
                'avg_handoff_time_ms': np.mean(self.metrics['handoff_times']),
                'p95_handoff_time_ms': np.percentile(self.metrics['handoff_times'], 95),
                'avg_mapping_error_um': np.mean(self.metrics['mapping_errors']),
                'throughput_per_hour': total_ops / runtime_hours if runtime_hours > 0 else 0,
                'runtime_hours': runtime_hours
            }
        else:
            stats = {'error': 'No successful operations recorded'}

        return stats

    def check_performance_alerts(self):
        """Check for performance alerts."""

        alerts = []

        if len(self.metrics['handoff_times']) >= 10:
            recent_avg = np.mean(self.metrics['handoff_times'][-10:])

            if recent_avg > 300:
                alerts.append(f"ALERT: Handoff time trending high ({recent_avg:.1f}ms)")

            if recent_avg > 500:
                alerts.append(f"CRITICAL: Handoff time critical ({recent_avg:.1f}ms)")

        total_ops = self.metrics['success_count'] + self.metrics['failure_count']
        if total_ops >= 100:
            success_rate = self.metrics['success_count'] / total_ops

            if success_rate < 0.99:
                alerts.append(f"ALERT: Success rate below target ({success_rate:.1%})")

            if success_rate < 0.95:
                alerts.append(f"CRITICAL: Success rate critical ({success_rate:.1%})")

        return alerts

# Usage in production
monitor = ProductionMonitor(system)

# In production loop
for scan_position in scan_positions:
    result = system.handoff_a_to_b_optimized(scan_position.z)
    monitor.log_handoff(result)

    # Check for alerts every 100 operations
    if monitor.metrics['success_count'] % 100 == 0:
        alerts = monitor.check_performance_alerts()
        for alert in alerts:
            print(f"⚠️  {alert}")

# Get performance summary
perf_summary = monitor.get_current_performance()
print(f"Current performance: {perf_summary}")
```

## Troubleshooting

### Common Performance Issues

#### 1. Slow Handoff Times

**Symptoms:**
- Handoff times consistently >300ms
- P95 times >500ms

**Diagnosis:**
```python
# Check timing breakdown
timing_breakdown = analyze_timing_breakdown(system, 20)
bottlenecks = identify_bottlenecks(timing_breakdown)

# Check cache efficiency
cache_stats = system.get_performance_stats()
print(f"Cache hit rate: {cache_stats['cache_hit_rate']:.1%}")
```

**Solutions:**
1. Switch to ULTRA_FAST optimization level
2. Increase cache size
3. Pre-populate cache with common positions
4. Check hardware settle times in lens profiles

#### 2. Poor Mapping Accuracy

**Symptoms:**
- Mapping errors >1.0μm
- RMS errors >0.1μm

**Diagnosis:**
```python
# Check mapping quality
mapping_report = system.parfocal_mapping.get_mapping_accuracy_report()
print(f"RMS error: {mapping_report['calibration']['rms_error_um']:.3f}μm")
print(f"Model type: {mapping_report['calibration']['model_type']}")

# Check temperature compensation
temp_metrics = mapping_report['temperature_compensation']
print(f"Thermal stability: {temp_metrics['thermal_stability_um_per_c']:.3f}μm/°C")
```

**Solutions:**
1. Recalibrate with more calibration points
2. Use ADAPTIVE model type for automatic optimization
3. Check temperature compensation
4. Validate hardware alignment

#### 3. Low Success Rate

**Symptoms:**
- Success rate <99%
- Frequent handoff failures

**Diagnosis:**
```python
# Check failure modes
qa = DualLensQA(system)
diagnostic_results = qa.run_diagnostic_tests()

# Check lens ranges and limits
for lens_id, profile in system.camera.profiles.items():
    print(f"{lens_id.value}: Range {profile.z_range_um}")
```

**Solutions:**
1. Check lens position ranges
2. Validate parfocal mapping calibration
3. Check hardware error conditions
4. Verify lens profiles match hardware

### Performance Debugging Tools

#### Debug Mode

```python
# Enable debug mode for detailed logging
system.enable_debug_mode = True
system.debug_log_level = 'VERBOSE'

# Perform operation with debugging
result = system.handoff_a_to_b_optimized(3.0)

# Review debug log
print(system.get_debug_log())
```

#### Hardware Validation

```python
# Validate hardware performance
def validate_hardware_performance(system):
    """Validate hardware meets performance specifications."""

    validation_results = {}

    # Test lens switching speed
    lens_switch_times = []
    for i in range(10):
        start = time.time()
        system.camera.set_active_lens(LensID.LENS_A)
        system.camera.set_active_lens(LensID.LENS_B)
        elapsed = (time.time() - start) * 1000
        lens_switch_times.append(elapsed)

    validation_results['lens_switch'] = {
        'avg_ms': np.mean(lens_switch_times),
        'target_ms': 50,
        'pass': np.mean(lens_switch_times) < 50
    }

    # Test focus speed
    focus_move_times = []
    for i in range(10):
        start = time.time()
        system.camera.set_focus(5.0)
        system.camera.set_focus(-5.0)
        elapsed = (time.time() - start) * 1000
        focus_move_times.append(elapsed)

    validation_results['focus_move'] = {
        'avg_ms': np.mean(focus_move_times),
        'target_ms': 80,
        'pass': np.mean(focus_move_times) < 80
    }

    return validation_results

# Run hardware validation
hw_validation = validate_hardware_performance(system)
print("Hardware Validation Results:")
for test, result in hw_validation.items():
    status = "PASS" if result['pass'] else "FAIL"
    print(f"  {test}: {result['avg_ms']:.1f}ms (target: {result['target_ms']}ms) - {status}")
```

## Conclusion

This comprehensive performance profiling guide provides the tools and techniques needed to optimize dual-lens autofocus system performance for production use. Regular profiling and monitoring ensure consistent performance and early detection of issues.

Key best practices:
- Use built-in performance monitoring for continuous tracking
- Run comprehensive benchmarks during development and deployment
- Monitor production performance with real-time alerts
- Optimize based on specific use case requirements
- Validate hardware performance regularly

For additional support or advanced optimization techniques, consult the [API Reference](api_reference.md) and [Troubleshooting Guide](troubleshooting.md).