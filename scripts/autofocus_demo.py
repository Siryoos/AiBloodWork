#!/usr/bin/env python3
"""
Comprehensive demonstration of the blood smear autofocus system.

This script demonstrates:
1. System setup and configuration
2. Basic autofocus operations
3. Focus surface building
4. Validation and testing
5. Thermal compensation
6. Illumination optimization

Usage:
    python scripts/autofocus_demo.py [--mock] [--config CONFIG_PATH]
"""

import argparse
import time
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bloodwork_ai.vision.autofocus import (
    create_blood_smear_autofocus,
    AutofocusConfig,
    AutofocusTestSuite,
    FocusAccuracyTest,
    MockIlluminationController,
    IlluminationPatterns,
)


class MockCamera:
    """Mock camera for demonstration purposes."""

    def __init__(self, noise_level: float = 0.1):
        self._focus = 0.0
        self._noise_level = noise_level
        self._optimal_focus = 0.0  # Ground truth optimal focus

    def get_frame(self) -> np.ndarray:
        """Generate synthetic blood smear image with focus-dependent sharpness."""
        # Create a simple synthetic image with edges
        img = np.random.randint(180, 220, (480, 640, 3), dtype=np.uint8)

        # Add some cell-like circular features
        y, x = np.ogrid[:480, :640]
        for _ in range(50):  # 50 synthetic cells
            cx, cy = np.random.randint(50, 590), np.random.randint(50, 430)
            r = np.random.randint(8, 15)
            mask = (x - cx)**2 + (y - cy)**2 <= r**2

            # Cell intensity depends on focus quality
            focus_quality = self._get_focus_quality()
            cell_intensity = int(120 + 80 * focus_quality)

            img[mask] = [cell_intensity, cell_intensity + 20, cell_intensity + 10]

        # Add noise
        noise = np.random.normal(0, self._noise_level * 255, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Simulate defocus blur
        focus_error = abs(self._focus - self._optimal_focus)
        if focus_error > 0.5:
            import cv2
            kernel_size = int(2 * focus_error + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), focus_error / 2)

        return img

    def _get_focus_quality(self) -> float:
        """Get focus quality (0-1) based on distance from optimal."""
        focus_error = abs(self._focus - self._optimal_focus)
        # Gaussian-like falloff
        return np.exp(-(focus_error / 3.0) ** 2)

    def set_focus(self, value: float) -> None:
        self._focus = float(value)

    def get_focus(self) -> float:
        return self._focus

    def get_focus_range(self) -> tuple:
        return (-30.0, 30.0)


class MockStage:
    """Mock XY stage for demonstration."""

    def __init__(self):
        self._x = 0.0
        self._y = 0.0

    def move_xy(self, x: float, y: float) -> None:
        self._x, self._y = float(x), float(y)
        time.sleep(0.01)  # Simulate movement time

    def get_xy(self) -> tuple:
        return (self._x, self._y)


class MockTemperatureSensor:
    """Mock temperature sensor with realistic drift."""

    def __init__(self):
        self._base_temp = 20.0
        self._start_time = time.time()

    def get_temperature(self) -> float:
        # Simulate slow temperature drift
        elapsed = time.time() - self._start_time
        drift = 0.1 * np.sin(elapsed / 60.0)  # ±0.1°C over 60 seconds
        noise = np.random.normal(0, 0.02)  # Small measurement noise
        return self._base_temp + drift + noise


def demonstrate_basic_autofocus():
    """Demonstrate basic autofocus functionality."""
    print("\n=== Basic Autofocus Demonstration ===")

    # Create mock hardware
    camera = MockCamera()
    illumination = MockIlluminationController(num_channels=8)

    # Create autofocus system
    config = AutofocusConfig.create_blood_smear_config()
    autofocus = create_blood_smear_autofocus(
        camera=camera,
        illumination=illumination,
        config=config
    )

    print("System created successfully")
    print(f"Initial status: {autofocus.get_system_status()}")

    # Set camera to some offset position
    camera.set_focus(5.0)
    print(f"\nStarting focus position: {camera.get_focus():.2f} um")

    # Run autofocus
    start_time = time.time()
    best_focus = autofocus.autofocus_at_position()
    duration = time.time() - start_time

    print(f"Autofocus completed in {duration:.2f}s")
    print(f"Best focus found: {best_focus:.2f} um")
    print(f"Focus error: {abs(best_focus - camera._optimal_focus):.2f} um")

    return autofocus


def demonstrate_surface_mapping(autofocus):
    """Demonstrate focus surface building."""
    print("\n=== Focus Surface Mapping Demonstration ===")

    # Add stage to system (needed for surface mapping)
    stage = MockStage()
    autofocus.stage = stage

    # Create a simple surface with some tilt
    def set_surface_focus(x, y):
        # Simulate a tilted sample: z = 0.1*x + 0.05*y
        optimal_z = 0.1 * x + 0.05 * y
        autofocus.camera._optimal_focus = optimal_z

    # Define tile region
    tile_bbox = (-10, -10, 10, 10)  # 20x20 um tile

    print("Building focus surface...")
    start_time = time.time()

    # Build surface with custom callback to simulate tilted sample
    original_autofocus_fn = None

    def custom_autofocus(z_guess=None):
        x, y = stage.get_xy()
        set_surface_focus(x, y)
        return autofocus.autofocus_at_position(z_guess=z_guess, use_surface_prediction=False)

    # Temporarily replace autofocus function
    surface_model, samples = autofocus.build_focus_surface(tile_bbox, grid_points=3)

    duration = time.time() - start_time
    print(f"Surface built in {duration:.2f}s with {len(samples)} sample points")
    print(f"Surface model type: {surface_model.kind}")

    # Test surface prediction
    test_x, test_y = 5.0, 5.0
    predicted_z = surface_model.predict(test_x, test_y)
    expected_z = 0.1 * test_x + 0.05 * test_y  # True surface equation
    print(f"Surface prediction at ({test_x}, {test_y}): {predicted_z:.2f} um")
    print(f"Expected value: {expected_z:.2f} um")
    print(f"Prediction error: {abs(predicted_z - expected_z):.2f} um")

    return surface_model


def demonstrate_validation(autofocus):
    """Demonstrate validation and testing."""
    print("\n=== Validation and Testing Demonstration ===")

    # Create a simple test suite
    test_suite = AutofocusTestSuite()

    # Add basic accuracy test
    test_suite.add_test(FocusAccuracyTest(
        name="demo_accuracy_test",
        z_positions=[-5, 0, 5],
        expected_accuracy_um=1.0,
        expected_repeatability_um=0.5,
        num_repeats=3
    ))

    # Run validation
    print("Running validation tests...")
    start_time = time.time()

    results = test_suite.run_all_tests(
        autofocus_fn=lambda z_guess: autofocus.autofocus_at_position(z_guess=z_guess),
        verbose=True
    )

    duration = time.time() - start_time
    print(f"\nValidation completed in {duration:.2f}s")

    # Print summary
    summary = results["summary"]
    print(f"Tests passed: {summary['all_tests_passed']}")
    print(f"Success rate: {summary['success_rate']:.1%}")

    return results


def demonstrate_thermal_compensation():
    """Demonstrate thermal drift compensation."""
    print("\n=== Thermal Compensation Demonstration ===")

    # Create system with temperature sensor
    camera = MockCamera()
    temp_sensor = MockTemperatureSensor()

    config = AutofocusConfig.create_blood_smear_config()
    config.thermal.enable_compensation = True

    autofocus = create_blood_smear_autofocus(
        camera=camera,
        temperature_sensor=temp_sensor,
        config=config
    )

    print("System with thermal compensation created")

    # Simulate some measurements over time
    print("Simulating thermal drift measurements...")
    for i in range(5):
        # Simulate temperature change affecting optimal focus
        temp = temp_sensor.get_temperature()
        temp_drift = (temp - 20.0) * 0.5  # 0.5 um/°C coefficient
        camera._optimal_focus = temp_drift

        # Run autofocus
        best_z = autofocus.autofocus_at_position()

        print(f"Measurement {i+1}: T={temp:.2f}°C, Z={best_z:.2f}um")

        # Wait a bit to simulate time passage
        time.sleep(0.1)

    # Get thermal statistics
    if autofocus._drift_tracker:
        stats = autofocus._drift_tracker.get_drift_statistics()
        print(f"Thermal statistics: {stats}")


def demonstrate_illumination_optimization():
    """Demonstrate illumination optimization."""
    print("\n=== Illumination Optimization Demonstration ===")

    camera = MockCamera()
    illumination = MockIlluminationController(num_channels=8)

    autofocus = create_blood_smear_autofocus(
        camera=camera,
        illumination=illumination
    )

    if autofocus._illumination_manager:
        print("Testing different illumination patterns...")

        patterns = [
            IlluminationPatterns.uniform(0.5, 8),
            IlluminationPatterns.brightfield(0.6, 8),
            IlluminationPatterns.darkfield(0.7, 8),
        ]

        from src.bloodwork_ai.vision.autofocus.metrics import tenengrad

        best_score = 0
        best_pattern = None

        for pattern in patterns:
            autofocus._illumination_manager.set_pattern(pattern)
            frame = camera.get_frame()
            score = tenengrad(frame)

            print(f"Pattern '{pattern.name}': score = {score:.2f}")

            if score > best_score:
                best_score = score
                best_pattern = pattern

        print(f"Best pattern: {best_pattern.name if best_pattern else 'None'}")


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="Blood smear autofocus demonstration")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--mock", action="store_true", help="Use mock hardware (default)")
    args = parser.parse_args()

    print("Blood Smear Autofocus System Demonstration")
    print("=" * 50)

    try:
        # Load configuration if provided
        if args.config:
            config = AutofocusConfig.load(args.config)
            print(f"Loaded configuration from {args.config}")
        else:
            config = AutofocusConfig.create_blood_smear_config()
            print("Using default blood smear configuration")

        # Run demonstrations
        autofocus = demonstrate_basic_autofocus()

        demonstrate_surface_mapping(autofocus)

        demonstrate_validation(autofocus)

        demonstrate_thermal_compensation()

        demonstrate_illumination_optimization()

        print("\n=== Demonstration Complete ===")
        print("All components of the autofocus system have been demonstrated.")

        # Clean up
        autofocus.close()

    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())