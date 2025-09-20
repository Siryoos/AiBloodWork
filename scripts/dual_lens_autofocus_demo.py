#!/usr/bin/env python3
"""
Dual-Lens Autofocus System Demonstration

This script demonstrates the complete dual-lens autofocus system with:
- Two permanently installed lenses (10-20x scanning + 40-60x detailed)
- Cross-lens parfocal mapping with ≤300ms handoff
- Per-lens focus surface models
- Temperature compensation
- Production QA validation

Usage:
    python scripts/dual_lens_autofocus_demo.py
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Add the autofocus module path directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "bloodwork_ai" / "vision"))

try:
    # Import dual-lens components
    from autofocus.dual_lens import (
        LensID, LensProfile, ParfocalMapping, DualLensAutofocusManager
    )
    from autofocus.dual_lens_camera import (
        DualLensCameraController, AcquisitionMode, DualLensFrame
    )
    from autofocus.dual_lens_qa import (
        DualLensQAHarness, DualLensQAConfig
    )
    from autofocus.config import AutofocusConfig
    from autofocus.illumination import MockIlluminationController

    print("✓ All dual-lens imports successful")

except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("Make sure you're in the project root directory")
    sys.exit(1)


class MockDualLensStage:
    """Mock XY stage for dual-lens demo."""

    def __init__(self):
        self._x = 0.0
        self._y = 0.0

    def move_xy(self, x: float, y: float) -> None:
        self._x, self._y = float(x), float(y)
        time.sleep(0.01)  # Simulate movement

    def get_xy(self) -> tuple:
        return (self._x, self._y)


def create_demo_lens_profiles() -> tuple:
    """Create realistic lens profiles for demonstration."""

    # Lens-A: 10-20x scanning lens (lower magnification, larger FOV)
    lens_a_config = AutofocusConfig.create_blood_smear_config()
    lens_a_config.search.coarse_step_um = 1.0  # Faster coarse steps for scanning
    lens_a_config.search.fine_step_um = 0.2

    lens_a = LensProfile(
        lens_id=LensID.LENS_A,
        name="Scanning Lens 20x/0.4",
        magnification=20.0,
        numerical_aperture=0.4,
        working_distance_mm=2.5,
        field_of_view_um=500.0,
        depth_of_field_um=3.0,
        z_range_um=(-25.0, 25.0),
        z_resolution_um=0.1,
        parfocal_offset_um=0.0,  # Reference lens
        af_config=lens_a_config,
        metric_weights={"tenengrad": 0.5, "laplacian": 0.3, "brenner": 0.2},
        preferred_illum_pattern="BRIGHTFIELD",
        illum_intensity_factor=1.0,
        focus_speed_um_per_s=300.0,
        settle_time_ms=8.0
    )

    # Lens-B: 40-60x detailed analysis lens (higher magnification, smaller FOV)
    lens_b_config = AutofocusConfig.create_blood_smear_config()
    lens_b_config.search.coarse_step_um = 0.5  # Finer steps for precision
    lens_b_config.search.fine_step_um = 0.1
    lens_b_config.metric.normalization_percentile = 99.5  # Higher precision

    lens_b = LensProfile(
        lens_id=LensID.LENS_B,
        name="Detail Lens 60x/0.8",
        magnification=60.0,
        numerical_aperture=0.8,
        working_distance_mm=1.5,
        field_of_view_um=200.0,
        depth_of_field_um=1.2,
        z_range_um=(-20.0, 20.0),
        z_resolution_um=0.05,
        parfocal_offset_um=2.5,  # Slight offset from Lens-A
        af_config=lens_b_config,
        metric_weights={"tenengrad": 0.3, "laplacian": 0.4, "brenner": 0.3},
        preferred_illum_pattern="LED_ANGLE_25",
        illum_intensity_factor=0.8,
        focus_speed_um_per_s=150.0,  # Slower for precision
        settle_time_ms=15.0
    )

    return lens_a, lens_b


def create_demo_parfocal_mapping() -> ParfocalMapping:
    """Create realistic parfocal mapping for demonstration."""
    return ParfocalMapping(
        linear_coeff=0.95,  # Slight magnification difference
        quadratic_coeff=0.001,  # Small field curvature difference
        offset_um=2.3,  # Constant offset
        calibration_timestamp=time.time(),
        temperature_c=23.0,
        num_calibration_points=25,
        rms_error_um=0.15,  # Good calibration
        temp_coeff_um_per_c=0.05  # Temperature compensation
    )


def demo_lens_profiles():
    """Demonstrate lens profile configuration."""
    print("\n" + "="*50)
    print("DUAL-LENS PROFILE CONFIGURATION")
    print("="*50)

    lens_a, lens_b = create_demo_lens_profiles()

    print(f"\n📍 Lens-A (Scanning): {lens_a.name}")
    print(f"  Magnification: {lens_a.magnification}x, NA: {lens_a.numerical_aperture}")
    print(f"  FOV: {lens_a.field_of_view_um} μm, DOF: {lens_a.depth_of_field_um} μm")
    print(f"  Z range: {lens_a.z_range_um[0]} to {lens_a.z_range_um[1]} μm")
    print(f"  Focus speed: {lens_a.focus_speed_um_per_s} μm/s")
    print(f"  Preferred illumination: {lens_a.preferred_illum_pattern}")

    print(f"\n📍 Lens-B (Detailed): {lens_b.name}")
    print(f"  Magnification: {lens_b.magnification}x, NA: {lens_b.numerical_aperture}")
    print(f"  FOV: {lens_b.field_of_view_um} μm, DOF: {lens_b.depth_of_field_um} μm")
    print(f"  Z range: {lens_b.z_range_um[0]} to {lens_b.z_range_um[1]} μm")
    print(f"  Focus speed: {lens_b.focus_speed_um_per_s} μm/s")
    print(f"  Preferred illumination: {lens_b.preferred_illum_pattern}")

    return lens_a, lens_b


def demo_parfocal_mapping():
    """Demonstrate parfocal mapping functionality."""
    print("\n" + "="*50)
    print("PARFOCAL MAPPING DEMONSTRATION")
    print("="*50)

    mapping = create_demo_parfocal_mapping()

    print(f"📐 Mapping Coefficients:")
    print(f"  Linear: {mapping.linear_coeff:.3f}")
    print(f"  Quadratic: {mapping.quadratic_coeff:.6f}")
    print(f"  Offset: {mapping.offset_um:.2f} μm")
    print(f"  RMS Error: {mapping.rms_error_um:.2f} μm")

    # Test mapping at different positions
    test_positions = [-5.0, -2.0, 0.0, 2.0, 5.0]

    print(f"\n🔄 Cross-Lens Mapping Test:")
    print(f"{'Lens-A (μm)':>12} {'→ Lens-B (μm)':>14} {'→ Lens-A (μm)':>14} {'Round-trip Error':>16}")
    print("-" * 60)

    for z_a in test_positions:
        z_b = mapping.map_lens_a_to_b(z_a)
        z_a_recovered = mapping.map_lens_b_to_a(z_b)
        round_trip_error = abs(z_a_recovered - z_a)

        print(f"{z_a:>10.2f} {z_b:>12.2f} {z_a_recovered:>12.2f} {round_trip_error:>14.3f}")

    return mapping


def demo_camera_coordination():
    """Demonstrate dual-lens camera coordination."""
    print("\n" + "="*50)
    print("CAMERA COORDINATION DEMONSTRATION")
    print("="*50)

    lens_a, lens_b = create_demo_lens_profiles()
    camera = DualLensCameraController(lens_a, lens_b)

    print("📸 Testing lens switching and focus control...")

    # Test individual lens operations
    for lens_id in [LensID.LENS_A, LensID.LENS_B]:
        print(f"\n  Switching to {lens_id.value}...")
        camera.set_active_lens(lens_id)

        print(f"    Active lens: {camera.get_active_lens().value}")

        # Test focus positioning
        test_z = 2.0
        camera.set_focus(test_z)
        actual_z = camera.get_focus()
        print(f"    Focus: {test_z:.1f} → {actual_z:.1f} μm")

        # Capture frame
        frame = camera.get_frame()
        print(f"    Frame shape: {frame.shape}")

    # Test simultaneous capture
    print(f"\n🔄 Testing alternating capture...")
    dual_frame = camera.get_dual_frame(
        lens_a_z_um=1.0,
        lens_b_z_um=1.5,
        mode=AcquisitionMode.ALTERNATING
    )

    print(f"  Lens-A frame: {dual_frame.lens_a_frame.shape if dual_frame.has_lens_a_data() else 'None'}")
    print(f"  Lens-B frame: {dual_frame.lens_b_frame.shape if dual_frame.has_lens_b_data() else 'None'}")
    print(f"  Focus positions: A={dual_frame.lens_a_z_um:.2f}, B={dual_frame.lens_b_z_um:.2f} μm")

    # Test focus bracketing
    print(f"\n📚 Testing focus bracketing (Lens-B)...")
    bracket_sequence = camera.focus_bracketing_sequence(
        LensID.LENS_B, center_z_um=0.0, range_um=4.0, num_steps=5
    )

    print(f"  Captured {len(bracket_sequence)} frames:")
    for i, (z_pos, frame) in enumerate(bracket_sequence):
        print(f"    Step {i+1}: z={z_pos:.2f} μm, shape={frame.shape}")

    return camera


def demo_handoff_performance():
    """Demonstrate fast handoff performance."""
    print("\n" + "="*50)
    print("FAST HANDOFF DEMONSTRATION (≤300ms)")
    print("="*50)

    lens_a, lens_b = create_demo_lens_profiles()
    camera = DualLensCameraController(lens_a, lens_b)
    stage = MockDualLensStage()
    illumination = MockIlluminationController()
    mapping = create_demo_parfocal_mapping()

    # Create dual-lens autofocus system
    system = DualLensAutofocusManager(
        camera=camera,
        stage_controller=stage,
        illumination=illumination,
        lens_a_profile=lens_a,
        lens_b_profile=lens_b,
        parfocal_mapping=mapping
    )

    print("🚀 Testing A→B handoffs...")

    handoff_times = []
    mapping_errors = []

    for i in range(10):
        # Random starting position
        z_start = np.random.uniform(-5.0, 5.0)

        # Ensure we start from Lens-A for each test
        system.camera.set_active_lens(LensID.LENS_A)
        system.camera.set_focus(z_start)

        # Perform handoff
        result = system.handoff_a_to_b(z_start)

        if result.success:
            handoff_times.append(result.elapsed_ms)
            mapping_errors.append(result.mapping_error_um)

            print(f"  Test {i+1:2d}: {z_start:5.1f} → {result.target_z_um:5.1f} μm "
                  f"({result.elapsed_ms:5.0f}ms, error: {result.mapping_error_um:.2f}μm)")

            if result.flags:
                print(f"          Flags: {result.flags}")
        else:
            print(f"  Test {i+1:2d}: FAILED - {result.flags}")

    # Performance analysis
    if handoff_times:
        avg_time = np.mean(handoff_times)
        p95_time = np.percentile(handoff_times, 95)
        max_time = np.max(handoff_times)

        avg_error = np.mean(mapping_errors)
        max_error = np.max(mapping_errors)

        print(f"\n📊 Handoff Performance Summary:")
        print(f"  Average time: {avg_time:.0f} ms")
        print(f"  P95 time: {p95_time:.0f} ms")
        print(f"  Max time: {max_time:.0f} ms")
        print(f"  Target (≤300ms): {'✓ MET' if p95_time <= 300 else '⚠ EXCEEDED'}")

        print(f"\n  Average mapping error: {avg_error:.2f} μm")
        print(f"  Max mapping error: {max_error:.2f} μm")
        print(f"  Target (≤1.0μm): {'✓ MET' if max_error <= 1.0 else '⚠ EXCEEDED'}")

    return system


def demo_focus_surface_calibration():
    """Demonstrate focus surface calibration."""
    print("\n" + "="*50)
    print("FOCUS SURFACE CALIBRATION")
    print("="*50)

    lens_a, lens_b = create_demo_lens_profiles()
    camera = DualLensCameraController(lens_a, lens_b)

    # Generate calibration data
    print("🎯 Calibrating focus surfaces...")

    for lens_id in [LensID.LENS_A, LensID.LENS_B]:
        print(f"\n  Calibrating {lens_id.value}...")

        # Generate realistic calibration points
        calibration_points = []
        for _ in range(20):
            x = np.random.uniform(-2000, 2000)  # ±2mm range
            y = np.random.uniform(-2000, 2000)

            # Simulate realistic focus surface with slight tilt and curvature
            z_base = np.random.uniform(-2, 2)
            tilt_x = 0.0001 * x
            tilt_y = 0.0001 * y
            curvature = 0.000001 * (x**2 + y**2)
            z = z_base + tilt_x + tilt_y + curvature

            calibration_points.append((x, y, z))

        # Perform calibration
        result = camera.calibrate_focus_surface(lens_id, calibration_points)

        if result["success"]:
            model = result["model"]
            print(f"    RMS error: {model['rms_error_um']:.3f} μm")
            print(f"    Max error: {model['max_error_um']:.3f} μm")
            print(f"    Fit quality: {result['fit_quality']}")

            # Test prediction
            test_x, test_y = 1000.0, 1500.0
            predicted_z = camera.predict_focus_position(lens_id, test_x, test_y)
            print(f"    Prediction test: ({test_x}, {test_y}) → {predicted_z:.2f} μm")

    return camera


def demo_temperature_compensation():
    """Demonstrate temperature compensation."""
    print("\n" + "="*50)
    print("TEMPERATURE COMPENSATION")
    print("="*50)

    mapping = create_demo_parfocal_mapping()

    print("🌡️  Testing temperature compensation...")

    base_temp = 23.0
    test_z = 0.0

    print(f"\n{'Temperature (°C)':>15} {'Lens-B Position (μm)':>20} {'Thermal Shift (μm)':>18}")
    print("-" * 55)

    base_position = mapping.map_lens_a_to_b(test_z, base_temp)

    for temp in [18.0, 20.0, 23.0, 26.0, 30.0]:
        position = mapping.map_lens_a_to_b(test_z, temp)
        thermal_shift = position - base_position

        print(f"{temp:>13.1f} {position:>18.3f} {thermal_shift:>16.3f}")

    print(f"\nTemperature coefficient: {mapping.temp_coeff_um_per_c:.3f} μm/°C")


def demo_qa_validation():
    """Demonstrate comprehensive QA validation."""
    print("\n" + "="*50)
    print("DUAL-LENS QA VALIDATION")
    print("="*50)

    # Create system
    lens_a, lens_b = create_demo_lens_profiles()
    camera = DualLensCameraController(lens_a, lens_b)
    stage = MockDualLensStage()
    illumination = MockIlluminationController()
    mapping = create_demo_parfocal_mapping()

    system = DualLensAutofocusManager(
        camera=camera,
        stage_controller=stage,
        illumination=illumination,
        lens_a_profile=lens_a,
        lens_b_profile=lens_b,
        parfocal_mapping=mapping
    )

    # Configure QA testing
    qa_config = DualLensQAConfig(
        num_handoff_tests=20,  # Reduced for demo
        num_surface_calibration_tests=15,
        num_parfocal_validation_tests=15,
        handoff_time_target_ms=300.0,
        mapping_accuracy_target_um=1.0,
        surface_prediction_target_um=0.5,
        enable_temperature_tests=True,
        output_dir="demo_qa_results"
    )

    # Run QA validation
    qa_harness = DualLensQAHarness(qa_config)
    summary = qa_harness.run_full_validation(system)

    print(f"\n🏆 QA Validation Results:")
    print(f"  Overall Status: {summary['overall_status']}")
    print(f"  Total Duration: {summary['total_duration_s']:.1f}s")

    for test_name, test_result in summary["test_summary"].items():
        status_icon = "✓" if test_result["status"] == "PASS" else "✗"
        print(f"  {status_icon} {test_name}: {test_result['status']} ({test_result['duration_s']:.1f}s)")

    if "key_metrics" in summary:
        metrics = summary["key_metrics"]
        print(f"\n📈 Key Performance Metrics:")
        if metrics.get("avg_handoff_time_ms"):
            print(f"  Average handoff time: {metrics['avg_handoff_time_ms']:.0f} ms")
        if metrics.get("p95_handoff_time_ms"):
            print(f"  P95 handoff time: {metrics['p95_handoff_time_ms']:.0f} ms")
        if metrics.get("avg_mapping_error_um"):
            print(f"  Average mapping error: {metrics['avg_mapping_error_um']:.2f} μm")
        if metrics.get("handoff_success_rate"):
            print(f"  Handoff success rate: {metrics['handoff_success_rate']:.1%}")

    return summary


def main():
    """Main demonstration function."""
    print("DUAL-LENS AUTOFOCUS SYSTEM DEMONSTRATION")
    print("Production-grade hematology autofocus with cross-lens mapping")
    print("="*70)

    try:
        # Run all demonstrations
        print("\n🔬 Running dual-lens demonstrations...")

        lens_a, lens_b = demo_lens_profiles()
        mapping = demo_parfocal_mapping()
        camera = demo_camera_coordination()
        system = demo_handoff_performance()
        demo_focus_surface_calibration()
        demo_temperature_compensation()
        qa_summary = demo_qa_validation()

        # Overall summary
        print("\n" + "="*70)
        print("DEMONSTRATION COMPLETE")
        print("="*70)

        print("\n✓ Features Demonstrated:")
        print("  • Dual-lens configuration (20x scanning + 60x detailed)")
        print("  • Cross-lens parfocal mapping with polynomial models")
        print("  • Fast handoff A↔B with ≤300ms target")
        print("  • Per-lens focus surface calibration")
        print("  • Temperature-compensated mapping")
        print("  • Dual-camera acquisition coordination")
        print("  • Focus bracketing and simultaneous capture modes")
        print("  • Comprehensive QA validation framework")

        # Check if performance targets were met
        print(f"\n📊 Performance Summary:")

        # Get handoff performance
        handoff_stats = system.get_handoff_performance_stats()
        if handoff_stats.get("avg_handoff_time_ms"):
            avg_handoff = handoff_stats["avg_handoff_time_ms"]
            p95_handoff = handoff_stats["p95_handoff_time_ms"]
            print(f"  Average handoff time: {avg_handoff:.0f} ms")
            print(f"  P95 handoff time: {p95_handoff:.0f} ms")
            print(f"  Handoff target (≤300ms): {'✓ MET' if p95_handoff <= 300 else '⚠ NEEDS TUNING'}")

            avg_error = handoff_stats["avg_mapping_error_um"]
            p95_error = handoff_stats["p95_mapping_error_um"]
            print(f"  Average mapping error: {avg_error:.2f} μm")
            print(f"  P95 mapping error: {p95_error:.2f} μm")
            print(f"  Mapping target (≤1.0μm): {'✓ MET' if p95_error <= 1.0 else '⚠ NEEDS TUNING'}")

        qa_status = qa_summary.get("overall_status", "UNKNOWN")
        print(f"  QA Validation: {qa_status}")

        # Camera performance
        camera_stats = camera.get_acquisition_stats()
        if camera_stats.get("throughput_fps"):
            print(f"  Camera throughput: {camera_stats['throughput_fps']:.1f} FPS")

        print(f"\n🚀 System Status: {'PRODUCTION READY' if qa_status == 'PASS' else 'NEEDS ATTENTION'}")
        print("   Dual-lens autofocus system ready for hematology integration")

        return 0

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())