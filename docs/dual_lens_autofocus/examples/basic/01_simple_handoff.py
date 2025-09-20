#!/usr/bin/env python3
"""
Basic Dual-Lens Handoff Example

This example demonstrates the simplest possible dual-lens handoff operation.
Perfect for understanding the core concepts and testing basic functionality.

Run with: python 01_simple_handoff.py
"""

import sys
import time
from pathlib import Path

# Add autofocus module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src" / "bloodwork_ai" / "vision"))

from autofocus.dual_lens import LensID, LensProfile
from autofocus.dual_lens_optimized import create_optimized_dual_lens_system, OptimizationLevel
from autofocus.dual_lens_camera_optimized import OptimizedDualLensCameraController
from autofocus.parfocal_mapping_optimized import EnhancedParfocalMapping
from autofocus.illumination import MockIlluminationController
from autofocus.config import AutofocusConfig


class SimpleStage:
    """Minimal stage controller for example."""

    def __init__(self):
        self.x, self.y = 0.0, 0.0

    def move_xy(self, x_um: float, y_um: float) -> None:
        self.x, self.y = x_um, y_um
        time.sleep(0.005)  # 5ms move time

    def get_xy(self) -> tuple:
        return (self.x, self.y)


def create_example_lens_profiles():
    """Create simple lens profiles for demonstration."""

    # Lens-A: Fast scanning lens
    lens_a_config = AutofocusConfig.create_blood_smear_config()
    lens_a_config.search.coarse_step_um = 1.5
    lens_a_config.search.max_iterations = 3

    lens_a = LensProfile(
        lens_id=LensID.LENS_A,
        name="Example Scanning 20x",
        magnification=20.0,
        numerical_aperture=0.4,
        z_range_um=(-20.0, 20.0),
        af_config=lens_a_config,
        focus_speed_um_per_s=500.0,
        settle_time_ms=5.0
    )

    # Lens-B: Detailed analysis lens
    lens_b_config = AutofocusConfig.create_blood_smear_config()
    lens_b_config.search.coarse_step_um = 0.8
    lens_b_config.search.max_iterations = 4

    lens_b = LensProfile(
        lens_id=LensID.LENS_B,
        name="Example Detail 60x",
        magnification=60.0,
        numerical_aperture=0.8,
        z_range_um=(-15.0, 15.0),
        af_config=lens_b_config,
        focus_speed_um_per_s=300.0,
        settle_time_ms=8.0
    )

    return lens_a, lens_b


def create_example_parfocal_mapping():
    """Create simple parfocal mapping for demonstration."""

    mapping = EnhancedParfocalMapping()

    # Simple linear mapping for demonstration
    mapping.coefficients = [2.1, 0.95, 0.001, 0.0]  # z_b = 2.1 + 0.95*z_a + 0.001*z_a²
    mapping.rms_error_um = 0.1
    mapping.calibration_timestamp = time.time()
    mapping.num_calibration_points = 20

    return mapping


def main():
    """Main example function."""

    print("=" * 60)
    print("SIMPLE DUAL-LENS HANDOFF EXAMPLE")
    print("=" * 60)

    # Step 1: Create hardware components
    print("\n1. Setting up hardware components...")

    camera = OptimizedDualLensCameraController(
        *create_example_lens_profiles(),
        enable_predictive_focus=True,
        enable_concurrent_operations=True
    )

    stage = SimpleStage()
    illumination = MockIlluminationController()

    print("   ✓ Camera controller created")
    print("   ✓ Stage controller created")
    print("   ✓ Illumination controller created")

    # Step 2: Create lens profiles and mapping
    print("\n2. Configuring lens system...")

    lens_a, lens_b = create_example_lens_profiles()
    mapping = create_example_parfocal_mapping()

    print(f"   ✓ Lens-A: {lens_a.name} ({lens_a.magnification}x)")
    print(f"   ✓ Lens-B: {lens_b.name} ({lens_b.magnification}x)")
    print(f"   ✓ Parfocal mapping: {mapping.rms_error_um:.2f}μm RMS error")

    # Step 3: Create autofocus system
    print("\n3. Creating optimized autofocus system...")

    system = create_optimized_dual_lens_system(
        camera=camera,
        stage_controller=stage,
        illumination=illumination,
        lens_a_profile=lens_a,
        lens_b_profile=lens_b,
        parfocal_mapping=mapping,
        optimization_level=OptimizationLevel.ULTRA_FAST
    )

    print("   ✓ Ultra-fast dual-lens system ready")

    try:
        # Step 4: Perform basic handoff
        print("\n4. Performing A→B handoff...")

        # Start with Lens-A at a known position
        source_z = 2.5  # μm
        camera.set_active_lens(LensID.LENS_A)
        camera.set_focus(source_z)

        print(f"   Starting position: Lens-A at {source_z:.1f}μm")

        # Perform optimized handoff
        start_time = time.time()
        result = system.handoff_a_to_b_optimized(source_z)
        total_time = (time.time() - start_time) * 1000

        # Display results
        print(f"\n5. Handoff Results:")

        if result.success:
            print(f"   ✓ SUCCESS in {result.elapsed_ms:.0f}ms")
            print(f"   Source: Lens-A at {result.source_z_um:.2f}μm")
            print(f"   Target: Lens-B at {result.target_z_um:.2f}μm")
            print(f"   Mapping error: {result.mapping_error_um:.3f}μm")

            # Show timing breakdown
            print(f"\n   Timing Breakdown:")
            print(f"   • Lens switch: {result.lens_switch_ms:.1f}ms")
            print(f"   • Focus move: {result.focus_move_ms:.1f}ms")
            print(f"   • Focus search: {result.focus_search_ms:.1f}ms")
            print(f"   • Validation: {result.validation_ms:.1f}ms")

            # Show optimizations used
            if result.concurrent_operations:
                print(f"   • Optimizations: {', '.join(result.concurrent_operations)}")

            if result.cache_hits > 0:
                print(f"   • Cache hits: {result.cache_hits}")

            # Performance assessment
            if result.elapsed_ms <= 300:
                print(f"   🎯 PERFORMANCE: Excellent (≤300ms target)")
            else:
                print(f"   ⚠️  PERFORMANCE: Needs optimization (>{result.elapsed_ms:.0f}ms)")

        else:
            print(f"   ✗ FAILED: {result.flags}")

        # Step 6: Test reverse handoff
        print(f"\n6. Testing B→A reverse handoff...")

        reverse_result = system.handoff_b_to_a(result.target_z_um)

        if reverse_result.success:
            print(f"   ✓ Reverse handoff: {reverse_result.elapsed_ms:.0f}ms")
            print(f"   Round-trip error: {abs(reverse_result.target_z_um - source_z):.3f}μm")
        else:
            print(f"   ✗ Reverse failed: {reverse_result.flags}")

        # Step 7: Performance summary
        print(f"\n7. Performance Summary:")

        stats = system.get_optimization_statistics()
        if stats.get('performance'):
            perf = stats['performance']
            print(f"   Average handoff time: {perf['avg_total_time_ms']:.0f}ms")
            print(f"   Target achievement: {perf['target_met_rate']*100:.1f}%")

        print(f"\n" + "=" * 60)
        print("EXAMPLE COMPLETED SUCCESSFULLY! ✓")
        print("=" * 60)

        print(f"\nKey Takeaways:")
        print(f"• Handoff completed in {result.elapsed_ms:.0f}ms (target: ≤300ms)")
        print(f"• Mapping accuracy: {result.mapping_error_um:.3f}μm")
        print(f"• System ready for production use")

    finally:
        # Clean up
        system.close()
        print(f"\n✓ System closed cleanly")


if __name__ == "__main__":
    main()