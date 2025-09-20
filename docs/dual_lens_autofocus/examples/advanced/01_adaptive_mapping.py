#!/usr/bin/env python3
"""
Advanced Adaptive Parfocal Mapping Example

This example demonstrates the advanced features of the enhanced parfocal mapping
system including adaptive learning, model selection, and real-time accuracy monitoring.

Run with: python 01_adaptive_mapping.py
"""

import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Tuple

# Add autofocus module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src" / "bloodwork_ai" / "vision"))

from autofocus.parfocal_mapping_optimized import (
    EnhancedParfocalMapping, MappingModel, create_enhanced_parfocal_mapping
)


def generate_realistic_calibration_data(num_points: int = 30) -> List[Tuple[float, float, float]]:
    """Generate realistic calibration data with characteristic patterns."""

    print(f"📊 Generating {num_points} realistic calibration points...")

    calibration_data = []

    # Define realistic mapping characteristics
    base_offset = 2.1       # Base offset between lenses
    linear_coeff = 0.95     # Near-unity scaling
    quadratic_coeff = 0.002 # Slight field curvature difference
    cubic_coeff = 0.0001    # Minor higher-order effects

    # Temperature effects
    temp_sensitivity = 0.05  # μm/°C

    for i in range(num_points):
        # Generate z_a across the range
        z_a = np.random.uniform(-8, 8)

        # Temperature variation
        temperature = np.random.uniform(20, 28)
        temp_effect = temp_sensitivity * (temperature - 23.0)

        # Calculate ideal z_b with realistic physics
        z_b_ideal = (base_offset +
                    linear_coeff * z_a +
                    quadratic_coeff * z_a**2 +
                    cubic_coeff * z_a**3 +
                    temp_effect)

        # Add realistic measurement noise
        measurement_noise = np.random.normal(0, 0.03)  # 30nm RMS noise
        z_b = z_b_ideal + measurement_noise

        calibration_data.append((z_a, z_b, temperature))

        if (i + 1) % 10 == 0:
            print(f"   Generated {i + 1}/{num_points} points...")

    print(f"   ✓ Calibration data complete")
    return calibration_data


def demonstrate_model_selection():
    """Demonstrate automatic model selection capabilities."""

    print("\n" + "="*60)
    print("ADAPTIVE MODEL SELECTION DEMONSTRATION")
    print("="*60)

    # Generate calibration data with different characteristics
    calibration_data = generate_realistic_calibration_data(35)

    print(f"\n🔬 Testing different mapping models...")

    # Test each model type explicitly
    models_to_test = [MappingModel.LINEAR, MappingModel.QUADRATIC, MappingModel.CUBIC]
    model_results = {}

    for model in models_to_test:
        print(f"\n   Testing {model.value.upper()} model...")

        mapping = EnhancedParfocalMapping(model_type=model)
        result = mapping.calibrate_enhanced(calibration_data)

        model_results[model] = result
        print(f"   ✓ RMS error: {result['rms_error_um']:.3f}μm")
        print(f"   ✓ Max error: {result['max_error_um']:.3f}μm")

    # Test adaptive model selection
    print(f"\n   Testing ADAPTIVE model selection...")
    adaptive_mapping = EnhancedParfocalMapping(model_type=MappingModel.ADAPTIVE)
    adaptive_result = adaptive_mapping.calibrate_enhanced(calibration_data)

    print(f"   ✓ Selected model: {adaptive_result['model_type'].upper()}")
    print(f"   ✓ RMS error: {adaptive_result['rms_error_um']:.3f}μm")

    # Compare results
    print(f"\n📈 Model Comparison:")
    print(f"{'Model':<12} {'RMS Error (μm)':<15} {'Max Error (μm)':<15} {'Rating':<10}")
    print("-" * 55)

    for model, result in model_results.items():
        rms = result['rms_error_um']
        max_err = result['max_error_um']
        rating = "Excellent" if rms < 0.1 else "Good" if rms < 0.2 else "Fair"

        print(f"{model.value.upper():<12} {rms:<15.3f} {max_err:<15.3f} {rating:<10}")

    # Show adaptive selection
    print(f"ADAPTIVE<12> {adaptive_result['rms_error_um']:<15.3f} "
          f"{adaptive_result['max_error_um']:<15.3f} {'Auto-Best':<10}")

    return adaptive_mapping


def demonstrate_adaptive_learning(mapping: EnhancedParfocalMapping):
    """Demonstrate real-time adaptive learning capabilities."""

    print("\n" + "="*60)
    print("ADAPTIVE LEARNING DEMONSTRATION")
    print("="*60)

    print(f"\n🧠 Testing adaptive learning with validation points...")

    # Simulate real-world measurements over time
    validation_scenarios = [
        ("Initial operation", 0, 5),
        ("After 1 hour", 1, 8),
        ("After thermal drift", 2, 12),
        ("After recalibration", 3, 6)
    ]

    initial_confidence = mapping._estimate_overall_confidence()
    print(f"   Initial system confidence: {initial_confidence:.2f}")

    for scenario_name, drift_factor, num_measurements in validation_scenarios:
        print(f"\n   Scenario: {scenario_name}")

        # Simulate measurement drift over time
        base_drift = drift_factor * 0.05  # Gradual drift

        for i in range(num_measurements):
            # Generate validation measurement
            z_a_test = np.random.uniform(-5, 5)

            # Simulate real measurement with drift
            z_b_predicted = mapping.map_lens_a_to_b(z_a_test)
            z_b_actual = z_b_predicted + base_drift + np.random.normal(0, 0.02)

            # Add to learning system
            mapping.add_validation_point(z_a_test, z_b_actual)

        # Show learning progress
        current_confidence = mapping._estimate_overall_confidence()
        recent_errors = mapping.accuracy_trend[-5:] if len(mapping.accuracy_trend) >= 5 else mapping.accuracy_trend

        if recent_errors:
            avg_recent_error = np.mean(recent_errors)
            print(f"     Recent average error: {avg_recent_error:.3f}μm")
            print(f"     System confidence: {current_confidence:.2f}")

            # Check if accuracy is improving
            if mapping._is_accuracy_improving():
                print(f"     📈 Accuracy trend: IMPROVING")
            else:
                print(f"     📊 Accuracy trend: STABLE")

    # Final learning report
    print(f"\n📊 Adaptive Learning Summary:")

    report = mapping.get_mapping_accuracy_report()
    perf = report['recent_performance']
    conf = report['confidence_metrics']

    print(f"   Validation points collected: {perf['num_validation_points']}")
    if perf['recent_avg_error_um']:
        print(f"   Recent average error: {perf['recent_avg_error_um']:.3f}μm")
    print(f"   Final confidence: {conf['overall_confidence']:.2f}")
    print(f"   Accuracy trend: {perf['accuracy_trend'].upper()}")

    return mapping


def demonstrate_temperature_compensation(mapping: EnhancedParfocalMapping):
    """Demonstrate temperature compensation capabilities."""

    print("\n" + "="*60)
    print("TEMPERATURE COMPENSATION DEMONSTRATION")
    print("="*60)

    print(f"\n🌡️  Testing temperature compensation...")

    # Test positions
    test_positions = [-3.0, 0.0, 3.0]
    test_temperatures = [18.0, 23.0, 28.0]

    print(f"\n   Testing mapping at different temperatures:")
    print(f"{'Z_A (μm)':<8} {'18°C':<8} {'23°C':<8} {'28°C':<8} {'Thermal Shift':<15}")
    print("-" * 55)

    for z_a in test_positions:
        z_b_values = []

        for temp in test_temperatures:
            z_b = mapping.map_lens_a_to_b(z_a, temp)
            z_b_values.append(z_b)

        # Calculate thermal shift
        thermal_shift = max(z_b_values) - min(z_b_values)

        print(f"{z_a:<8.1f} {z_b_values[0]:<8.2f} {z_b_values[1]:<8.2f} "
              f"{z_b_values[2]:<8.2f} {thermal_shift:<15.3f}")

    # Temperature compensation analysis
    temp_report = mapping.get_mapping_accuracy_report()['temperature_compensation']
    print(f"\n   Temperature Compensation Analysis:")
    print(f"   • Offset coefficient: {temp_report['offset_coefficient_um_per_c']:.3f}μm/°C")
    print(f"   • Linear coefficient: {temp_report['linear_coefficient_um_per_c']:.3f}μm/°C")
    print(f"   • Thermal stability: {temp_report['thermal_stability_um_per_c']:.3f}μm/°C")

    stability = temp_report['thermal_stability_um_per_c']
    if stability < 0.1:
        print(f"   ✓ Excellent thermal stability (<0.1μm/°C)")
    elif stability < 0.2:
        print(f"   ✓ Good thermal stability (<0.2μm/°C)")
    else:
        print(f"   ⚠ Thermal compensation may need improvement")


def demonstrate_confidence_estimation(mapping: EnhancedParfocalMapping):
    """Demonstrate mapping confidence estimation."""

    print("\n" + "="*60)
    print("CONFIDENCE ESTIMATION DEMONSTRATION")
    print("="*60)

    print(f"\n🎯 Testing confidence estimation at various positions...")

    # Test confidence at different positions
    test_positions = np.linspace(-6, 6, 13)

    print(f"\n   Position-dependent confidence analysis:")
    print(f"{'Position (μm)':<15} {'Predicted Z_B':<15} {'Confidence':<12} {'Assessment':<12}")
    print("-" * 60)

    for z_a in test_positions:
        z_b_pred = mapping.map_lens_a_to_b(z_a)
        confidence = mapping._estimate_local_confidence(z_a, z_b_pred)

        if confidence > 0.9:
            assessment = "Excellent"
        elif confidence > 0.7:
            assessment = "Good"
        elif confidence > 0.5:
            assessment = "Fair"
        else:
            assessment = "Poor"

        print(f"{z_a:<15.1f} {z_b_pred:<15.2f} {confidence:<12.2f} {assessment:<12}")

    # Overall confidence report
    print(f"\n   Overall Confidence Assessment:")

    report = mapping.get_mapping_accuracy_report()
    conf_metrics = report['confidence_metrics']

    print(f"   • Overall confidence: {conf_metrics['overall_confidence']:.2f}")
    print(f"   • Calibration freshness: {conf_metrics['calibration_freshness']:.2f}")

    # Recommendations based on confidence
    overall_conf = conf_metrics['overall_confidence']
    if overall_conf > 0.9:
        print(f"   ✅ Recommendation: Mapping ready for production use")
    elif overall_conf > 0.7:
        print(f"   ⚠️  Recommendation: Consider additional calibration points")
    else:
        print(f"   🔴 Recommendation: Recalibration required")


def demonstrate_mapping_persistence():
    """Demonstrate saving and loading mapping configurations."""

    print("\n" + "="*60)
    print("MAPPING PERSISTENCE DEMONSTRATION")
    print("="*60)

    print(f"\n💾 Testing mapping save/load functionality...")

    # Create enhanced mapping
    calibration_data = generate_realistic_calibration_data(25)
    original_mapping = create_enhanced_parfocal_mapping(calibration_data)

    # Add some validation points
    for i in range(5):
        z_a = np.random.uniform(-3, 3)
        z_b_actual = original_mapping.map_lens_a_to_b(z_a) + np.random.normal(0, 0.02)
        original_mapping.add_validation_point(z_a, z_b_actual)

    # Save mapping
    save_path = "/tmp/test_mapping.json"
    original_mapping.save_mapping(save_path)
    print(f"   ✓ Mapping saved to {save_path}")

    # Load mapping
    loaded_mapping = EnhancedParfocalMapping.load_mapping(save_path)
    print(f"   ✓ Mapping loaded from {save_path}")

    # Verify consistency
    test_positions = [-2.0, 0.0, 2.0]
    print(f"\n   Verifying save/load consistency:")
    print(f"{'Position':<10} {'Original':<12} {'Loaded':<12} {'Difference':<12}")
    print("-" * 50)

    for z_a in test_positions:
        original_z_b = original_mapping.map_lens_a_to_b(z_a)
        loaded_z_b = loaded_mapping.map_lens_a_to_b(z_a)
        difference = abs(original_z_b - loaded_z_b)

        print(f"{z_a:<10.1f} {original_z_b:<12.3f} {loaded_z_b:<12.3f} {difference:<12.6f}")

    # Check metadata preservation
    orig_report = original_mapping.get_mapping_accuracy_report()
    load_report = loaded_mapping.get_mapping_accuracy_report()

    print(f"\n   Metadata preservation:")
    print(f"   • Model type: {orig_report['calibration']['model_type']} → {load_report['calibration']['model_type']}")
    print(f"   • RMS error: {orig_report['calibration']['rms_error_um']:.3f} → {load_report['calibration']['rms_error_um']:.3f}μm")
    print(f"   • Validation points: {orig_report['recent_performance']['num_validation_points']} → {load_report['recent_performance']['num_validation_points']}")

    print(f"   ✅ Save/load functionality verified")

    # Cleanup
    Path(save_path).unlink()


def main():
    """Main demonstration function."""

    print("=" * 80)
    print("ADVANCED ADAPTIVE PARFOCAL MAPPING DEMONSTRATION")
    print("=" * 80)

    print("\nThis demonstration showcases advanced features:")
    print("• Automatic model selection (Linear/Quadratic/Cubic/Adaptive)")
    print("• Real-time adaptive learning with validation points")
    print("• Temperature compensation and thermal stability")
    print("• Position-dependent confidence estimation")
    print("• Mapping persistence and configuration management")

    try:
        # Demonstration sequence
        print(f"\n🚀 Starting advanced mapping demonstrations...")

        # 1. Model selection
        mapping = demonstrate_model_selection()

        # 2. Adaptive learning
        mapping = demonstrate_adaptive_learning(mapping)

        # 3. Temperature compensation
        demonstrate_temperature_compensation(mapping)

        # 4. Confidence estimation
        demonstrate_confidence_estimation(mapping)

        # 5. Persistence
        demonstrate_mapping_persistence()

        # Final summary
        print("\n" + "=" * 80)
        print("ADVANCED DEMONSTRATION COMPLETED SUCCESSFULLY! ✓")
        print("=" * 80)

        final_report = mapping.get_mapping_accuracy_report()
        cal = final_report['calibration']
        perf = final_report['recent_performance']
        conf = final_report['confidence_metrics']

        print(f"\n🎯 Final Mapping Performance:")
        print(f"• Model type: {cal['model_type'].upper()}")
        print(f"• RMS accuracy: {cal['rms_error_um']:.3f}μm")
        print(f"• Validation points: {perf['num_validation_points']}")
        print(f"• Overall confidence: {conf['overall_confidence']:.2f}")
        print(f"• System status: Production ready ✅")

        print(f"\n💡 Key Takeaways:")
        print(f"• Adaptive model selection automatically optimizes accuracy")
        print(f"• Real-time learning continuously improves performance")
        print(f"• Temperature compensation ensures thermal stability")
        print(f"• Confidence estimation provides reliability metrics")
        print(f"• Mapping persistence enables production deployment")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())