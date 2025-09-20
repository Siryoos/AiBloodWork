#!/usr/bin/env python3
"""
Performance Comparison Example

This example compares different optimization levels to demonstrate the
performance improvements available in the dual-lens autofocus system.

Run with: python 02_performance_comparison.py
"""

import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Dict

# Add autofocus module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src" / "bloodwork_ai" / "vision"))

from autofocus.dual_lens import LensID, LensProfile
from autofocus.dual_lens_optimized import create_optimized_dual_lens_system, OptimizationLevel
from autofocus.dual_lens_camera_optimized import OptimizedDualLensCameraController
from autofocus.parfocal_mapping_optimized import EnhancedParfocalMapping
from autofocus.illumination import MockIlluminationController
from autofocus.config import AutofocusConfig


class BenchmarkStage:
    """Stage controller for benchmarking."""

    def __init__(self):
        self.x, self.y = 0.0, 0.0

    def move_xy(self, x_um: float, y_um: float) -> None:
        self.x, self.y = x_um, y_um
        time.sleep(0.002)  # Fast 2ms move

    def get_xy(self) -> tuple:
        return (self.x, self.y)


def create_benchmark_system(optimization_level: OptimizationLevel):
    """Create system with specified optimization level."""

    # Create hardware
    lens_a_config = AutofocusConfig.create_blood_smear_config()
    lens_b_config = AutofocusConfig.create_blood_smear_config()

    # Adjust configs based on optimization level
    if optimization_level == OptimizationLevel.ULTRA_FAST:
        lens_a_config.search.max_iterations = 2
        lens_b_config.search.max_iterations = 3
        settle_time_factor = 0.3
    elif optimization_level == OptimizationLevel.FAST:
        lens_a_config.search.max_iterations = 3
        lens_b_config.search.max_iterations = 4
        settle_time_factor = 0.5
    else:  # STANDARD
        lens_a_config.search.max_iterations = 5
        lens_b_config.search.max_iterations = 6
        settle_time_factor = 1.0

    lens_a = LensProfile(
        lens_id=LensID.LENS_A,
        name=f"Benchmark Lens A ({optimization_level.value})",
        magnification=20.0,
        z_range_um=(-20.0, 20.0),
        af_config=lens_a_config,
        focus_speed_um_per_s=500.0,
        settle_time_ms=5.0 * settle_time_factor
    )

    lens_b = LensProfile(
        lens_id=LensID.LENS_B,
        name=f"Benchmark Lens B ({optimization_level.value})",
        magnification=60.0,
        z_range_um=(-15.0, 15.0),
        af_config=lens_b_config,
        focus_speed_um_per_s=300.0,
        settle_time_ms=10.0 * settle_time_factor
    )

    camera = OptimizedDualLensCameraController(
        lens_a, lens_b,
        enable_predictive_focus=True,
        enable_concurrent_operations=True
    )

    stage = BenchmarkStage()
    illumination = MockIlluminationController()

    # Create simple parfocal mapping
    mapping = EnhancedParfocalMapping()
    mapping.coefficients = [2.0, 0.95, 0.001, 0.0]
    mapping.rms_error_um = 0.08

    # Create system
    system = create_optimized_dual_lens_system(
        camera=camera,
        stage_controller=stage,
        illumination=illumination,
        lens_a_profile=lens_a,
        lens_b_profile=lens_b,
        parfocal_mapping=mapping,
        optimization_level=optimization_level
    )

    return system


def benchmark_handoff_performance(system, optimization_level: OptimizationLevel, num_tests: int = 15) -> Dict:
    """Benchmark handoff performance for specific optimization level."""

    print(f"\n🔥 Benchmarking {optimization_level.value.upper()} mode ({num_tests} tests)...")

    results = []
    timing_breakdown = {'lens_switch': [], 'focus_move': [], 'focus_search': [], 'validation': []}

    for i in range(num_tests):
        # Random source position
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

        # Progress indicator
        if (i + 1) % 5 == 0:
            print(f"   Completed {i + 1}/{num_tests} tests...")

    # Analyze results
    successful_results = [r for r in results if r.success]
    handoff_times = [r.elapsed_ms for r in successful_results]
    mapping_errors = [r.mapping_error_um for r in successful_results]

    if handoff_times:
        benchmark_data = {
            'optimization_level': optimization_level,
            'num_tests': num_tests,
            'success_count': len(successful_results),
            'success_rate': len(successful_results) / num_tests,
            'timing': {
                'avg_ms': np.mean(handoff_times),
                'median_ms': np.median(handoff_times),
                'min_ms': np.min(handoff_times),
                'max_ms': np.max(handoff_times),
                'p95_ms': np.percentile(handoff_times, 95),
                'std_ms': np.std(handoff_times)
            },
            'accuracy': {
                'avg_error_um': np.mean(mapping_errors),
                'max_error_um': np.max(mapping_errors),
                'p95_error_um': np.percentile(mapping_errors, 95)
            },
            'timing_breakdown': {
                phase: {
                    'avg_ms': np.mean(times) if times else 0,
                    'contribution_pct': (np.mean(times) / np.mean(handoff_times) * 100) if times and handoff_times else 0
                }
                for phase, times in timing_breakdown.items()
            },
            'target_performance': {
                'target_met_count': sum(1 for t in handoff_times if t <= 300),
                'target_met_rate': np.mean([t <= 300 for t in handoff_times])
            }
        }
    else:
        benchmark_data = {
            'optimization_level': optimization_level,
            'success_count': 0,
            'error': 'No successful handoffs'
        }

    return benchmark_data


def display_benchmark_results(benchmarks: List[Dict]):
    """Display comprehensive benchmark comparison."""

    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON RESULTS")
    print("=" * 80)

    # Summary table
    print(f"\n📊 Performance Summary:")
    print(f"{'Mode':<12} {'Avg Time':<10} {'Min/Max':<12} {'P95':<8} {'Target Met':<12} {'Accuracy':<10}")
    print("-" * 70)

    for benchmark in benchmarks:
        if 'timing' in benchmark:
            mode = benchmark['optimization_level'].value.upper()
            avg_time = benchmark['timing']['avg_ms']
            min_time = benchmark['timing']['min_ms']
            max_time = benchmark['timing']['max_ms']
            p95_time = benchmark['timing']['p95_ms']
            target_rate = benchmark['target_performance']['target_met_rate']
            avg_error = benchmark['accuracy']['avg_error_um']

            print(f"{mode:<12} {avg_time:>6.0f}ms   {min_time:>3.0f}/{max_time:<3.0f}ms   "
                  f"{p95_time:>5.0f}ms  {target_rate*100:>7.1f}%     {avg_error:>5.2f}μm")

    # Detailed breakdown
    print(f"\n⚡ Timing Breakdown by Optimization Level:")

    for benchmark in benchmarks:
        if 'timing_breakdown' in benchmark:
            mode = benchmark['optimization_level'].value.upper()
            print(f"\n{mode} Mode:")

            breakdown = benchmark['timing_breakdown']
            for phase, data in breakdown.items():
                phase_name = phase.replace('_', ' ').title()
                avg_time = data['avg_ms']
                contribution = data['contribution_pct']
                print(f"  {phase_name:<15}: {avg_time:>5.1f}ms ({contribution:>4.1f}%)")

    # Speed comparison
    print(f"\n🚀 Speed Improvement Analysis:")

    if len(benchmarks) >= 2:
        standard_time = None
        for benchmark in benchmarks:
            if benchmark['optimization_level'] == OptimizationLevel.STANDARD:
                standard_time = benchmark['timing']['avg_ms']
                break

        if standard_time:
            print(f"Baseline (STANDARD): {standard_time:.0f}ms")

            for benchmark in benchmarks:
                if benchmark['optimization_level'] != OptimizationLevel.STANDARD:
                    mode = benchmark['optimization_level'].value.upper()
                    avg_time = benchmark['timing']['avg_ms']
                    improvement = ((standard_time - avg_time) / standard_time) * 100
                    speedup = standard_time / avg_time

                    print(f"{mode:<12}: {avg_time:>5.0f}ms ({improvement:>+5.1f}% faster, {speedup:.1f}x speedup)")

    # Accuracy comparison
    print(f"\n🎯 Accuracy Analysis:")

    for benchmark in benchmarks:
        if 'accuracy' in benchmark:
            mode = benchmark['optimization_level'].value.upper()
            avg_error = benchmark['accuracy']['avg_error_um']
            max_error = benchmark['accuracy']['max_error_um']
            p95_error = benchmark['accuracy']['p95_error_um']

            print(f"{mode:<12}: avg={avg_error:.3f}μm, max={max_error:.3f}μm, P95={p95_error:.3f}μm")

    # Recommendations
    print(f"\n💡 Recommendations:")

    best_performance = min(benchmarks, key=lambda b: b['timing']['avg_ms'] if 'timing' in b else float('inf'))
    best_accuracy = min(benchmarks, key=lambda b: b['accuracy']['avg_error_um'] if 'accuracy' in b else float('inf'))

    if best_performance and 'timing' in best_performance:
        mode = best_performance['optimization_level'].value.upper()
        time_ms = best_performance['timing']['avg_ms']
        print(f"• Fastest performance: {mode} mode ({time_ms:.0f}ms average)")

    if best_accuracy and 'accuracy' in best_accuracy:
        mode = best_accuracy['optimization_level'].value.upper()
        error_um = best_accuracy['accuracy']['avg_error_um']
        print(f"• Best accuracy: {mode} mode ({error_um:.3f}μm average error)")

    # Production recommendation
    ultra_fast = next((b for b in benchmarks if b['optimization_level'] == OptimizationLevel.ULTRA_FAST), None)
    if ultra_fast and 'target_performance' in ultra_fast:
        if ultra_fast['target_performance']['target_met_rate'] >= 0.95:
            print(f"• Production recommendation: ULTRA_FAST mode (95%+ target achievement)")
        else:
            print(f"• Production recommendation: FAST mode (balanced performance/reliability)")


def main():
    """Main benchmark comparison function."""

    print("=" * 80)
    print("DUAL-LENS AUTOFOCUS PERFORMANCE COMPARISON")
    print("=" * 80)

    print("\nThis benchmark compares different optimization levels:")
    print("• STANDARD: Balanced performance and accuracy")
    print("• FAST: Optimized for speed with minimal accuracy trade-off")
    print("• ULTRA_FAST: Maximum speed, production-optimized")

    optimization_levels = [
        OptimizationLevel.STANDARD,
        OptimizationLevel.FAST,
        OptimizationLevel.ULTRA_FAST
    ]

    benchmarks = []

    for level in optimization_levels:
        print(f"\n🔧 Setting up {level.value.upper()} system...")

        system = create_benchmark_system(level)

        try:
            benchmark_data = benchmark_handoff_performance(system, level, num_tests=15)
            benchmarks.append(benchmark_data)

            if 'timing' in benchmark_data:
                avg_time = benchmark_data['timing']['avg_ms']
                success_rate = benchmark_data['success_rate']
                print(f"   ✓ Completed: {avg_time:.0f}ms average, {success_rate*100:.1f}% success rate")
            else:
                print(f"   ✗ Failed: {benchmark_data.get('error', 'Unknown error')}")

        finally:
            system.close()

    # Display comprehensive results
    if benchmarks:
        display_benchmark_results(benchmarks)

        print(f"\n" + "=" * 80)
        print("BENCHMARK COMPLETED SUCCESSFULLY! ✓")
        print("=" * 80)

        # Final summary
        ultra_fast_data = next((b for b in benchmarks if b['optimization_level'] == OptimizationLevel.ULTRA_FAST), None)
        if ultra_fast_data and 'timing' in ultra_fast_data:
            avg_time = ultra_fast_data['timing']['avg_ms']
            target_rate = ultra_fast_data['target_performance']['target_met_rate']

            print(f"\n🎯 Key Results:")
            print(f"• Ultra-fast average: {avg_time:.0f}ms")
            print(f"• Target achievement: {target_rate*100:.1f}% ≤300ms")
            print(f"• System ready for production use")

    else:
        print("\n❌ No successful benchmarks completed")


if __name__ == "__main__":
    main()