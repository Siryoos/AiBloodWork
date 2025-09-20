#!/usr/bin/env python3
"""
Complete Integration Tutorial

This comprehensive tutorial demonstrates how to integrate the dual-lens autofocus
system into a production hematology slide scanner from hardware setup to
operational deployment.

This tutorial covers:
1. Hardware interface implementation
2. System calibration procedures
3. Production configuration
4. Quality assurance validation
5. Performance monitoring
6. Troubleshooting and maintenance

Run with: python complete_integration_tutorial.py
"""

import sys
import time
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

# Add autofocus module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src" / "bloodwork_ai" / "vision"))

from autofocus.dual_lens import LensID, LensProfile, CameraInterface
from autofocus.dual_lens_optimized import create_optimized_dual_lens_system, OptimizationLevel
from autofocus.dual_lens_camera_optimized import OptimizedDualLensCameraController
from autofocus.parfocal_mapping_optimized import EnhancedParfocalMapping, create_enhanced_parfocal_mapping
from autofocus.dual_lens_qa import DualLensQAHarness, DualLensQAConfig
from autofocus.illumination import IlluminationController
from autofocus.config import AutofocusConfig


# ============================================================================
# STEP 1: HARDWARE INTERFACE IMPLEMENTATION
# ============================================================================

class ProductionCameraController(CameraInterface):
    """Production camera controller implementing the CameraInterface protocol.

    This example shows how to implement the required interface for your
    specific camera hardware.
    """

    def __init__(self, camera_hardware_device):
        """Initialize with your specific camera hardware.

        Args:
            camera_hardware_device: Your camera hardware driver/SDK interface
        """
        self.camera_hw = camera_hardware_device
        self.current_lens = LensID.LENS_A
        self.focus_positions = {LensID.LENS_A: 0.0, LensID.LENS_B: 0.0}

        # Performance tracking
        self.acquisition_count = 0
        self.lens_switch_count = 0

        print("✓ Production camera controller initialized")

    def get_frame(self) -> np.ndarray:
        """Capture frame from current active lens.

        Replace this implementation with your camera's capture method.
        """
        # Example implementation - replace with your hardware calls
        start_time = time.time()

        # Your camera capture code here:
        # frame = self.camera_hw.capture_frame()

        # For tutorial, simulate realistic frame
        if self.current_lens == LensID.LENS_A:
            frame = self._simulate_scanning_frame()
        else:
            frame = self._simulate_detail_frame()

        self.acquisition_count += 1

        # Simulate realistic capture time
        elapsed = time.time() - start_time
        if elapsed < 0.01:  # Ensure minimum 10ms capture time
            time.sleep(0.01 - elapsed)

        return frame

    def set_active_lens(self, lens_id: LensID) -> None:
        """Switch to specified lens.

        Replace this implementation with your lens switching mechanism.
        """
        if self.current_lens != lens_id:
            start_time = time.time()

            # Your lens switching code here:
            # self.camera_hw.switch_to_lens(lens_id.value)

            self.current_lens = lens_id
            self.lens_switch_count += 1

            # Realistic lens switching time (adjust for your hardware)
            elapsed = time.time() - start_time
            settle_time = 0.03  # 30ms settling time
            if elapsed < settle_time:
                time.sleep(settle_time - elapsed)

            print(f"   Switched to {lens_id.value}")

    def get_active_lens(self) -> LensID:
        """Get currently active lens."""
        return self.current_lens

    def set_focus(self, z_um: float) -> None:
        """Set focus position for active lens.

        Replace this implementation with your focus control.
        """
        start_time = time.time()

        # Your focus control code here:
        # self.camera_hw.set_focus_position(z_um)

        # Calculate movement time based on distance
        current_z = self.focus_positions[self.current_lens]
        move_distance = abs(z_um - current_z)
        move_time = move_distance / 300.0  # 300 μm/s focus speed

        if move_time > 0.001:  # Only sleep for significant moves
            time.sleep(min(move_time, 0.05))  # Cap at 50ms

        self.focus_positions[self.current_lens] = z_um

    def get_focus(self) -> float:
        """Get focus position of active lens."""
        return self.focus_positions[self.current_lens]

    def _simulate_scanning_frame(self) -> np.ndarray:
        """Simulate scanning lens frame (replace with actual capture)."""
        # 20x magnification - larger field of view
        frame = np.random.randint(180, 220, (600, 600, 3), dtype=np.uint8)

        # Add some blood cells for realism
        for _ in range(25):
            cx, cy = np.random.randint(50, 550, 2)
            radius = np.random.randint(10, 16)
            y, x = np.ogrid[:600, :600]
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            frame[mask] = [200, 160, 160]

        return frame

    def _simulate_detail_frame(self) -> np.ndarray:
        """Simulate detail lens frame (replace with actual capture)."""
        # 60x magnification - smaller field of view, more detail
        frame = np.random.randint(180, 220, (400, 400, 3), dtype=np.uint8)

        # Add detailed blood cells
        for _ in range(15):
            cx, cy = np.random.randint(40, 360, 2)
            radius = np.random.randint(20, 30)
            y, x = np.ogrid[:400, :400]
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            frame[mask] = [200, 160, 160]

            # Central pallor
            center_mask = (x - cx)**2 + (y - cy)**2 <= (radius//3)**2
            frame[center_mask] = [180, 140, 140]

        return frame


class ProductionStageController:
    """Production XY stage controller.

    Implement this class for your specific stage hardware.
    """

    def __init__(self, stage_hardware_device):
        """Initialize with your stage hardware.

        Args:
            stage_hardware_device: Your stage hardware driver/SDK interface
        """
        self.stage_hw = stage_hardware_device
        self.current_x = 0.0
        self.current_y = 0.0
        self.move_count = 0

        print("✓ Production stage controller initialized")

    def move_xy(self, x_um: float, y_um: float) -> None:
        """Move to specified XY position.

        Replace this implementation with your stage movement.
        """
        start_time = time.time()

        # Your stage movement code here:
        # self.stage_hw.move_to_position(x_um, y_um)

        # Calculate movement time based on distance
        move_distance = np.sqrt((x_um - self.current_x)**2 + (y_um - self.current_y)**2)
        move_time = move_distance / 10000.0  # 10 mm/s stage speed

        if move_time > 0.001:  # Only sleep for significant moves
            time.sleep(min(move_time, 0.1))  # Cap at 100ms

        self.current_x = x_um
        self.current_y = y_um
        self.move_count += 1

    def get_xy(self) -> tuple:
        """Get current XY position."""
        return (self.current_x, self.current_y)


class ProductionIlluminationController(IlluminationController):
    """Production illumination controller.

    Implement this class for your illumination system.
    """

    def __init__(self, illumination_hardware_device):
        """Initialize with your illumination hardware.

        Args:
            illumination_hardware_device: Your illumination hardware interface
        """
        self.illum_hw = illumination_hardware_device
        self.current_pattern = "BRIGHTFIELD"
        self.current_intensity = 0.5

        print("✓ Production illumination controller initialized")

    def set_pattern_by_name(self, pattern_name: str) -> None:
        """Set illumination pattern.

        Replace this implementation with your illumination control.
        """
        # Your illumination pattern code here:
        # self.illum_hw.set_pattern(pattern_name)

        self.current_pattern = pattern_name
        time.sleep(0.005)  # 5ms settling time

    def set_intensity(self, intensity: float) -> None:
        """Set illumination intensity.

        Replace this implementation with your intensity control.
        """
        # Your intensity control code here:
        # self.illum_hw.set_intensity(intensity)

        self.current_intensity = intensity
        time.sleep(0.002)  # 2ms settling time


# ============================================================================
# STEP 2: PRODUCTION CONFIGURATION
# ============================================================================

@dataclass
class ProductionConfig:
    """Production system configuration."""

    # Lens specifications (customize for your hardware)
    lens_a_magnification: float = 20.0
    lens_a_na: float = 0.4
    lens_a_fov_um: float = 500.0
    lens_a_z_range_um: tuple = (-25.0, 25.0)
    lens_a_focus_speed: float = 500.0  # μm/s

    lens_b_magnification: float = 60.0
    lens_b_na: float = 0.8
    lens_b_fov_um: float = 200.0
    lens_b_z_range_um: tuple = (-20.0, 20.0)
    lens_b_focus_speed: float = 300.0  # μm/s

    # Performance targets
    handoff_time_target_ms: float = 300.0
    mapping_accuracy_target_um: float = 1.0

    # Optimization settings
    optimization_level: OptimizationLevel = OptimizationLevel.ULTRA_FAST
    enable_caching: bool = True
    enable_concurrent_ops: bool = True

    # Quality assurance
    qa_validation_interval_hours: float = 24.0
    qa_alert_threshold: float = 0.8  # Success rate threshold


def create_production_lens_profiles(config: ProductionConfig) -> tuple:
    """Create production lens profiles from configuration."""

    print("🔧 Creating production lens profiles...")

    # Lens-A configuration (scanning)
    lens_a_config = AutofocusConfig.create_blood_smear_config()
    lens_a_config.search.coarse_step_um = 1.5
    lens_a_config.search.fine_step_um = 0.3
    lens_a_config.search.max_iterations = 3  # Fast for scanning

    lens_a = LensProfile(
        lens_id=LensID.LENS_A,
        name=f"Production Scanning {config.lens_a_magnification}x/{config.lens_a_na}",
        magnification=config.lens_a_magnification,
        numerical_aperture=config.lens_a_na,
        field_of_view_um=config.lens_a_fov_um,
        z_range_um=config.lens_a_z_range_um,
        af_config=lens_a_config,
        focus_speed_um_per_s=config.lens_a_focus_speed,
        settle_time_ms=5.0,
        metric_weights={"tenengrad": 0.6, "laplacian": 0.4},
        preferred_illum_pattern="BRIGHTFIELD"
    )

    # Lens-B configuration (detailed)
    lens_b_config = AutofocusConfig.create_blood_smear_config()
    lens_b_config.search.coarse_step_um = 0.8
    lens_b_config.search.fine_step_um = 0.15
    lens_b_config.search.max_iterations = 5  # More precision

    lens_b = LensProfile(
        lens_id=LensID.LENS_B,
        name=f"Production Detail {config.lens_b_magnification}x/{config.lens_b_na}",
        magnification=config.lens_b_magnification,
        numerical_aperture=config.lens_b_na,
        field_of_view_um=config.lens_b_fov_um,
        z_range_um=config.lens_b_z_range_um,
        af_config=lens_b_config,
        focus_speed_um_per_s=config.lens_b_focus_speed,
        settle_time_ms=8.0,
        metric_weights={"tenengrad": 0.3, "laplacian": 0.4, "brenner": 0.3},
        preferred_illum_pattern="LED_ANGLE_25"
    )

    print(f"   ✓ Lens-A: {lens_a.name}")
    print(f"   ✓ Lens-B: {lens_b.name}")

    return lens_a, lens_b


# ============================================================================
# STEP 3: CALIBRATION PROCEDURES
# ============================================================================

def perform_system_calibration(camera, stage, config: ProductionConfig) -> EnhancedParfocalMapping:
    """Perform comprehensive system calibration."""

    print("\n" + "="*60)
    print("SYSTEM CALIBRATION PROCEDURE")
    print("="*60)

    print("\n📐 Starting parfocal mapping calibration...")
    print("   This process will take approximately 5-10 minutes")
    print("   Please ensure:")
    print("   • Slide is properly loaded and focused")
    print("   • System is thermally stable")
    print("   • No vibrations are present")

    input("\nPress Enter to start calibration...")

    calibration_data = []

    # Step 1: Define calibration positions
    z_a_positions = np.linspace(-8, 8, 25)  # 25 points across range
    num_positions = len(z_a_positions)

    print(f"\n   Collecting {num_positions} calibration points...")

    # Step 2: Find good focus area on slide
    print("   Moving to calibration area...")
    calibration_x, calibration_y = 5000.0, 5000.0  # Center of slide
    stage.move_xy(calibration_x, calibration_y)

    # Step 3: Collect calibration data
    for i, z_a in enumerate(z_a_positions):
        print(f"   Point {i+1}/{num_positions}: z_a = {z_a:.1f}μm", end=" ")

        # Focus Lens-A at target position
        camera.set_active_lens(LensID.LENS_A)
        camera.set_focus(z_a)

        # Switch to Lens-B and find best focus
        camera.set_active_lens(LensID.LENS_B)

        # Search for optimal focus with Lens-B
        z_b_search_range = np.linspace(-10, 10, 21)
        best_z_b = None
        best_metric = 0

        for z_b_test in z_b_search_range:
            camera.set_focus(z_b_test)
            frame = camera.get_frame()

            # Calculate focus metric
            from autofocus.metrics import tenengrad
            metric = tenengrad(frame)

            if metric > best_metric:
                best_metric = metric
                best_z_b = z_b_test

        # Record calibration point
        temperature = 23.0  # TODO: Add actual temperature measurement
        calibration_data.append((z_a, best_z_b, temperature))

        print(f"→ z_b = {best_z_b:.2f}μm")

    # Step 4: Create enhanced mapping
    print(f"\n   Creating enhanced parfocal mapping...")
    mapping = create_enhanced_parfocal_mapping(calibration_data)

    # Step 5: Validate calibration
    report = mapping.get_mapping_accuracy_report()
    cal = report['calibration']

    print(f"\n   ✓ Calibration complete:")
    print(f"     Model type: {cal['model_type'].upper()}")
    print(f"     RMS error: {cal['rms_error_um']:.3f}μm")
    print(f"     Max error: {cal['max_error_um']:.3f}μm")

    # Step 6: Save calibration
    calibration_file = "production_parfocal_calibration.json"
    mapping.save_mapping(calibration_file)
    print(f"     Saved to: {calibration_file}")

    # Step 7: Assess calibration quality
    if cal['rms_error_um'] < 0.2:
        print(f"   🎯 Calibration quality: EXCELLENT")
    elif cal['rms_error_um'] < 0.5:
        print(f"   ✓ Calibration quality: GOOD")
    else:
        print(f"   ⚠️ Calibration quality: NEEDS IMPROVEMENT")
        print(f"      Consider recalibrating with more points")

    return mapping


# ============================================================================
# STEP 4: PRODUCTION SYSTEM CREATION
# ============================================================================

def create_production_system(config: ProductionConfig) -> tuple:
    """Create complete production autofocus system."""

    print("\n" + "="*60)
    print("PRODUCTION SYSTEM SETUP")
    print("="*60)

    print("\n🏭 Initializing production components...")

    # Step 1: Initialize hardware (replace with your hardware)
    print("   Initializing hardware interfaces...")

    # TODO: Replace these with your actual hardware initialization
    camera_hw = None  # Your camera hardware object
    stage_hw = None   # Your stage hardware object
    illum_hw = None   # Your illumination hardware object

    camera = ProductionCameraController(camera_hw)
    stage = ProductionStageController(stage_hw)
    illumination = ProductionIlluminationController(illum_hw)

    # Step 2: Create lens profiles
    lens_a, lens_b = create_production_lens_profiles(config)

    # Step 3: Load or create parfocal mapping
    print("   Loading parfocal mapping...")
    try:
        mapping = EnhancedParfocalMapping.load_mapping("production_parfocal_calibration.json")
        print(f"     ✓ Loaded existing calibration")
    except FileNotFoundError:
        print(f"     No existing calibration found")
        print(f"     Performing new calibration...")
        mapping = perform_system_calibration(camera, stage, config)

    # Step 4: Create optimized autofocus system
    print("   Creating optimized autofocus system...")
    system = create_optimized_dual_lens_system(
        camera=camera,
        stage_controller=stage,
        illumination=illumination,
        lens_a_profile=lens_a,
        lens_b_profile=lens_b,
        parfocal_mapping=mapping,
        optimization_level=config.optimization_level
    )

    print(f"   ✓ Production system ready")
    print(f"     Optimization level: {config.optimization_level.value.upper()}")
    print(f"     Performance target: ≤{config.handoff_time_target_ms}ms")

    return system, camera, stage, illumination


# ============================================================================
# STEP 5: QUALITY ASSURANCE VALIDATION
# ============================================================================

def run_production_qa_validation(system, config: ProductionConfig) -> bool:
    """Run comprehensive QA validation for production system."""

    print("\n" + "="*60)
    print("QUALITY ASSURANCE VALIDATION")
    print("="*60)

    print("\n🧪 Running production QA validation...")

    # Configure QA testing
    qa_config = DualLensQAConfig(
        num_handoff_tests=30,  # Comprehensive testing
        num_surface_calibration_tests=20,
        num_parfocal_validation_tests=25,
        handoff_time_target_ms=config.handoff_time_target_ms,
        mapping_accuracy_target_um=config.mapping_accuracy_target_um,
        surface_prediction_target_um=0.5,
        enable_temperature_tests=True,
        output_dir="production_qa_results"
    )

    # Run validation
    qa_harness = DualLensQAHarness(qa_config)
    summary = qa_harness.run_full_validation(system)

    # Display results
    print(f"\n📊 QA Validation Results:")
    print(f"   Overall Status: {summary['overall_status']}")
    print(f"   Total Duration: {summary['total_duration_s']:.1f}s")

    # Individual test results
    all_passed = True
    for test_name, result in summary['test_summary'].items():
        status_icon = "✓" if result['status'] == 'PASS' else "✗"
        print(f"   {status_icon} {test_name}: {result['status']}")
        if result['status'] != 'PASS':
            all_passed = False

    # Key performance metrics
    if 'key_metrics' in summary:
        metrics = summary['key_metrics']
        print(f"\n🎯 Key Performance Metrics:")
        print(f"   Average handoff time: {metrics['avg_handoff_time_ms']:.0f}ms")
        print(f"   P95 handoff time: {metrics['p95_handoff_time_ms']:.0f}ms")
        print(f"   Handoff success rate: {metrics['handoff_success_rate']*100:.1f}%")

        # Performance assessment
        if metrics['p95_handoff_time_ms'] <= config.handoff_time_target_ms:
            print(f"   ✅ Performance target: MET")
        else:
            print(f"   ⚠️ Performance target: NEEDS OPTIMIZATION")
            all_passed = False

    # Production readiness assessment
    if all_passed:
        print(f"\n🎉 PRODUCTION VALIDATION: PASSED")
        print(f"   System is ready for production deployment")
    else:
        print(f"\n⚠️ PRODUCTION VALIDATION: FAILED")
        print(f"   System requires optimization before deployment")

    return all_passed


# ============================================================================
# STEP 6: PERFORMANCE MONITORING
# ============================================================================

class ProductionMonitor:
    """Production performance monitoring system."""

    def __init__(self, system, config: ProductionConfig):
        self.system = system
        self.config = config
        self.monitoring_active = False
        self.monitor_thread = None

        # Performance tracking
        self.handoff_history = []
        self.alerts_sent = []

    def start_monitoring(self):
        """Start continuous performance monitoring."""
        print("\n📊 Starting production performance monitoring...")

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        print("   ✓ Performance monitoring active")

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        print("   ✓ Performance monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Get current statistics
                stats = self.system.get_optimization_statistics()

                if stats.get('performance'):
                    perf = stats['performance']

                    # Check performance thresholds
                    if perf['avg_total_time_ms'] > self.config.handoff_time_target_ms:
                        self._send_alert("PERFORMANCE_DEGRADED",
                                       f"Average handoff time: {perf['avg_total_time_ms']:.0f}ms")

                    if perf['target_met_rate'] < self.config.qa_alert_threshold:
                        self._send_alert("TARGET_ACHIEVEMENT_LOW",
                                       f"Target achievement rate: {perf['target_met_rate']*100:.1f}%")

                # Log statistics every hour
                current_time = time.time()
                if len(self.handoff_history) == 0 or current_time - self.handoff_history[-1]['timestamp'] > 3600:
                    self.handoff_history.append({
                        'timestamp': current_time,
                        'stats': stats
                    })

                    print(f"[{time.strftime('%H:%M:%S')}] Performance update: "
                          f"{perf.get('avg_total_time_ms', 0):.0f}ms avg")

                # Sleep for monitoring interval
                time.sleep(60)  # Check every minute

            except Exception as e:
                print(f"   ⚠️ Monitoring error: {e}")
                time.sleep(60)

    def _send_alert(self, alert_type: str, message: str):
        """Send performance alert."""
        alert = {
            'timestamp': time.time(),
            'type': alert_type,
            'message': message
        }

        # Avoid duplicate alerts
        if not any(a['type'] == alert_type for a in self.alerts_sent[-10:]):
            self.alerts_sent.append(alert)
            print(f"   🚨 ALERT [{alert_type}]: {message}")

            # TODO: Integrate with your alerting system
            # send_email_alert(alert)
            # send_slack_notification(alert)

    def get_monitoring_report(self) -> Dict[str, Any]:
        """Get comprehensive monitoring report."""
        return {
            'monitoring_duration_hours': len(self.handoff_history),
            'total_alerts': len(self.alerts_sent),
            'recent_alerts': self.alerts_sent[-5:],
            'performance_history': self.handoff_history[-24:],  # Last 24 hours
            'system_health': 'GOOD' if len(self.alerts_sent) == 0 else 'DEGRADED'
        }


# ============================================================================
# STEP 7: MAIN INTEGRATION DEMONSTRATION
# ============================================================================

def main():
    """Main integration tutorial demonstration."""

    print("=" * 80)
    print("COMPLETE DUAL-LENS AUTOFOCUS INTEGRATION TUTORIAL")
    print("=" * 80)

    print("\nThis tutorial demonstrates complete production integration:")
    print("• Hardware interface implementation")
    print("• System configuration and calibration")
    print("• Quality assurance validation")
    print("• Performance monitoring setup")
    print("• Production deployment procedures")

    # Step 1: Load production configuration
    print("\n🔧 Loading production configuration...")
    config = ProductionConfig(
        optimization_level=OptimizationLevel.ULTRA_FAST,
        handoff_time_target_ms=250.0,  # Aggressive target
        mapping_accuracy_target_um=0.8
    )
    print("   ✓ Production configuration loaded")

    try:
        # Step 2: Create production system
        system, camera, stage, illumination = create_production_system(config)

        # Step 3: Run QA validation
        qa_passed = run_production_qa_validation(system, config)

        if not qa_passed:
            print("\n⚠️ QA validation failed - system needs optimization")
            print("   Please review QA results and adjust configuration")
            return 1

        # Step 4: Demonstrate production operation
        print("\n" + "="*60)
        print("PRODUCTION OPERATION DEMONSTRATION")
        print("="*60)

        print("\n🚀 Demonstrating production autofocus operations...")

        # Example scanning workflow
        scan_positions = [
            (1000, 2000), (2000, 2000), (3000, 2000),
            (1000, 3000), (2000, 3000), (3000, 3000)
        ]

        total_handoffs = 0
        total_time = 0

        for i, (x, y) in enumerate(scan_positions):
            print(f"\n   Position {i+1}: ({x}, {y})")

            # Scanning autofocus
            stage.move_xy(x, y)
            z_a = system.autofocus_scanning(x, y)
            print(f"     Lens-A focus: {z_a:.2f}μm")

            # Fast handoff to detailed lens
            result = system.handoff_a_to_b_optimized(z_a)
            if result.success:
                print(f"     Handoff: {result.elapsed_ms:.0f}ms, error: {result.mapping_error_um:.2f}μm")
                total_handoffs += 1
                total_time += result.elapsed_ms

            # Detailed analysis (simulated)
            z_b = system.autofocus_detailed(x, y, z_guess_um=result.target_z_um)
            print(f"     Lens-B focus: {z_b:.3f}μm")

        # Performance summary
        if total_handoffs > 0:
            avg_handoff_time = total_time / total_handoffs
            print(f"\n📊 Production Performance Summary:")
            print(f"   Total handoffs: {total_handoffs}")
            print(f"   Average handoff time: {avg_handoff_time:.0f}ms")
            print(f"   Target achievement: {'✓ MET' if avg_handoff_time <= config.handoff_time_target_ms else '⚠ EXCEEDED'}")

        # Step 5: Setup monitoring
        print("\n🔍 Setting up production monitoring...")
        monitor = ProductionMonitor(system, config)
        monitor.start_monitoring()

        # Simulate monitoring for a short period
        print("   Monitoring system performance (30 seconds)...")
        time.sleep(30)

        # Get monitoring report
        report = monitor.get_monitoring_report()
        print(f"\n   Monitoring Report:")
        print(f"   • System health: {report['system_health']}")
        print(f"   • Total alerts: {report['total_alerts']}")

        monitor.stop_monitoring()

        # Step 6: Final recommendations
        print("\n" + "="*80)
        print("INTEGRATION TUTORIAL COMPLETED SUCCESSFULLY! ✓")
        print("="*80)

        print("\n🎯 Production Deployment Checklist:")
        print("   ✅ Hardware interfaces implemented")
        print("   ✅ System calibration completed")
        print("   ✅ QA validation passed")
        print("   ✅ Performance monitoring setup")

        print("\n💡 Next Steps for Production:")
        print("   • Integrate with your slide scanner control software")
        print("   • Setup automated QA validation schedule")
        print("   • Configure alerting and monitoring systems")
        print("   • Train operators on system operation")
        print("   • Establish maintenance procedures")

        print(f"\n🚀 System Status: READY FOR PRODUCTION DEPLOYMENT")

    except Exception as e:
        print(f"\n❌ Integration error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Clean up
        if 'system' in locals():
            system.close()
        print("\n✓ System cleanup completed")

    return 0


if __name__ == "__main__":
    exit(main())