# Getting Started with Dual-Lens Autofocus

## 🎯 Welcome!

This guide will walk you through setting up and using the Dual-Lens Autofocus system from installation to production deployment. Whether you're integrating with existing hardware or starting fresh, this tutorial provides step-by-step instructions.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8+ (3.10+ recommended)
- **Operating System**: Windows 10+, macOS 10.15+, or Linux
- **Memory**: 4GB RAM minimum, 8GB+ recommended
- **Hardware**: Dual-lens microscope system with motorized focus

### Dependencies
```bash
# Core dependencies
numpy>=1.20.0
opencv-python>=4.5.0
scipy>=1.7.0

# Optional performance dependencies
numba>=0.55.0  # For optimized metrics (recommended)
asyncio>=3.4.0  # For concurrent operations

# Development dependencies (optional)
pytest>=6.0.0
sphinx>=4.0.0
black>=21.0.0
```

## 🚀 Installation

### Method 1: Development Installation

```bash
# Clone the repository
git clone <your-repository-url>
cd AiBloodWork

# Create virtual environment
python -m venv dual_lens_env
source dual_lens_env/bin/activate  # On Windows: dual_lens_env\Scripts\activate

# Install in development mode
pip install -e .

# Install optional performance dependencies
pip install numba  # Highly recommended for 2-3x speed improvement
```

### Method 2: Package Installation

```bash
# Install from PyPI (when available)
pip install bloodwork-ai-autofocus

# Or install from wheel
pip install bloodwork_ai_autofocus-1.0.0-py3-none-any.whl
```

### Verify Installation

```bash
# Run the demonstration to verify installation
python scripts/dual_lens_optimized_demo.py
```

Expected output:
```
✓ All optimized dual-lens imports successful
OPTIMIZED DUAL-LENS AUTOFOCUS SYSTEM
Ultra-fast hematology autofocus with ≤300ms handoff target
...
```

## 🔧 Basic Setup

### Step 1: Camera Interface Implementation

First, implement the camera interface for your hardware:

```python
# your_camera.py
from autofocus.dual_lens import CameraInterface, LensID
import numpy as np

class YourCameraController(CameraInterface):
    """Camera interface implementation for your hardware."""

    def __init__(self, camera_device):
        self.camera = camera_device
        self.current_lens = LensID.LENS_A
        self.focus_position = 0.0

    def get_frame(self) -> np.ndarray:
        """Capture frame from current active lens."""
        # Replace with your camera's capture method
        frame = self.camera.capture()
        return frame

    def set_active_lens(self, lens_id: LensID) -> None:
        """Switch to specified lens."""
        # Replace with your lens switching mechanism
        self.camera.switch_lens(lens_id.value)
        self.current_lens = lens_id

        # Add settling time if needed
        import time
        time.sleep(0.05)  # 50ms settling time

    def get_active_lens(self) -> LensID:
        """Get currently active lens."""
        return self.current_lens

    def set_focus(self, z_um: float) -> None:
        """Set focus position for active lens."""
        # Replace with your focus control
        self.camera.set_focus_position(z_um)
        self.focus_position = z_um

        # Add settling time if needed
        time.sleep(0.01)  # 10ms settling

    def get_focus(self) -> float:
        """Get focus position of active lens."""
        return self.focus_position
```

### Step 2: Stage Controller Implementation

```python
# your_stage.py
class YourStageController:
    """Stage controller for XY positioning."""

    def __init__(self, stage_device):
        self.stage = stage_device
        self.x_position = 0.0
        self.y_position = 0.0

    def move_xy(self, x_um: float, y_um: float) -> None:
        """Move to specified XY position."""
        # Replace with your stage movement
        self.stage.move_to(x_um, y_um)
        self.x_position = x_um
        self.y_position = y_um

        # Add movement time if needed
        import time
        time.sleep(0.01)  # 10ms movement time

    def get_xy(self) -> tuple:
        """Get current XY position."""
        return (self.x_position, self.y_position)
```

### Step 3: Illumination Controller

```python
# your_illumination.py
from autofocus.illumination import IlluminationController

class YourIlluminationController(IlluminationController):
    """Illumination controller for your hardware."""

    def __init__(self, illumination_device):
        self.illumination = illumination_device
        self.current_pattern = "BRIGHTFIELD"
        self.current_intensity = 0.5

    def set_pattern_by_name(self, pattern_name: str) -> None:
        """Set illumination pattern."""
        # Replace with your illumination control
        self.illumination.set_pattern(pattern_name)
        self.current_pattern = pattern_name

    def set_intensity(self, intensity: float) -> None:
        """Set illumination intensity."""
        # Replace with your intensity control
        self.illumination.set_intensity(intensity)
        self.current_intensity = intensity
```

## 📐 Configuration

### Step 4: Create Lens Profiles

```python
# lens_config.py
from autofocus.dual_lens import LensProfile, LensID
from autofocus.config import AutofocusConfig

def create_production_lens_profiles():
    """Create optimized lens profiles for your system."""

    # Configure autofocus algorithm for Lens-A (scanning)
    lens_a_config = AutofocusConfig.create_blood_smear_config()
    lens_a_config.search.coarse_step_um = 1.5  # Larger steps for speed
    lens_a_config.search.fine_step_um = 0.3
    lens_a_config.search.max_iterations = 3    # Fewer iterations for speed

    lens_a = LensProfile(
        lens_id=LensID.LENS_A,
        name="Production Scanning 20x/0.4",
        serial_number="LENS_A_001",  # Your lens serial number

        # Optical properties (adjust for your lens)
        magnification=20.0,
        numerical_aperture=0.4,
        working_distance_mm=2.5,
        field_of_view_um=500.0,
        depth_of_field_um=3.0,

        # Mechanical properties (adjust for your hardware)
        z_range_um=(-25.0, 25.0),    # Your focus range
        z_resolution_um=0.1,         # Your focus resolution
        parfocal_offset_um=0.0,      # Reference lens

        # Performance settings (optimize for your hardware)
        af_config=lens_a_config,
        focus_speed_um_per_s=500.0,  # Your focus speed
        settle_time_ms=5.0,          # Your settling time

        # Focus metrics (tune for your images)
        metric_weights={
            "tenengrad": 0.6,        # Good for edge content
            "laplacian": 0.4         # Good for texture
        },

        # Illumination settings
        preferred_illum_pattern="BRIGHTFIELD",
        illum_intensity_factor=1.0
    )

    # Configure autofocus algorithm for Lens-B (detailed)
    lens_b_config = AutofocusConfig.create_blood_smear_config()
    lens_b_config.search.coarse_step_um = 0.8   # Finer steps for precision
    lens_b_config.search.fine_step_um = 0.15
    lens_b_config.search.max_iterations = 5     # More iterations for accuracy

    lens_b = LensProfile(
        lens_id=LensID.LENS_B,
        name="Production Detail 60x/0.8",
        serial_number="LENS_B_001",

        # Optical properties
        magnification=60.0,
        numerical_aperture=0.8,
        working_distance_mm=1.5,
        field_of_view_um=200.0,
        depth_of_field_um=1.2,

        # Mechanical properties
        z_range_um=(-20.0, 20.0),
        z_resolution_um=0.05,
        parfocal_offset_um=2.5,      # Measured offset from Lens-A

        # Performance settings
        af_config=lens_b_config,
        focus_speed_um_per_s=300.0,  # Slower for precision
        settle_time_ms=10.0,         # Longer settling for stability

        # Focus metrics
        metric_weights={
            "tenengrad": 0.3,
            "laplacian": 0.4,
            "brenner": 0.3           # Good for fine detail
        },

        # Illumination settings
        preferred_illum_pattern="LED_ANGLE_25",
        illum_intensity_factor=0.8
    )

    return lens_a, lens_b
```

### Step 5: Parfocal Mapping Calibration

```python
# calibration.py
from autofocus.parfocal_mapping_optimized import create_enhanced_parfocal_mapping
import numpy as np

def perform_parfocal_calibration(system):
    """Perform parfocal mapping calibration."""

    print("Starting parfocal calibration...")

    # Step 1: Collect calibration data
    calibration_data = []

    # Define test positions across the focus range
    z_a_positions = np.linspace(-8, 8, 25)  # 25 points across range

    for i, z_a in enumerate(z_a_positions):
        print(f"Calibration point {i+1}/25: z_a = {z_a:.1f}μm")

        # Move to test position (use a good focus area on your slide)
        system.stage.move_xy(x_um=1000, y_um=2000)

        # Focus with Lens-A
        system.camera.set_active_lens(LensID.LENS_A)
        system.camera.set_focus(z_a)

        # Switch to Lens-B and find best focus
        system.camera.set_active_lens(LensID.LENS_B)

        # Search for best focus with Lens-B
        best_z_b = None
        best_metric = 0

        for z_b_test in np.linspace(-10, 10, 21):  # Search range for Lens-B
            system.camera.set_focus(z_b_test)
            frame = system.camera.get_frame()

            # Calculate focus metric
            from autofocus.metrics import tenengrad
            metric = tenengrad(frame)

            if metric > best_metric:
                best_metric = metric
                best_z_b = z_b_test

        # Record calibration point
        temperature = 23.0  # Measure actual temperature
        calibration_data.append((z_a, best_z_b, temperature))

        print(f"  Found correspondence: {z_a:.1f} → {best_z_b:.1f}μm")

    # Step 2: Create enhanced mapping
    print("Creating enhanced parfocal mapping...")
    mapping = create_enhanced_parfocal_mapping(calibration_data)

    # Step 3: Validate mapping
    report = mapping.get_mapping_accuracy_report()
    print(f"Calibration complete:")
    print(f"  Model type: {report['calibration']['model_type']}")
    print(f"  RMS error: {report['calibration']['rms_error_um']:.3f}μm")
    print(f"  Max error: {report['calibration']['max_error_um']:.3f}μm")

    # Step 4: Save mapping for production use
    mapping.save_mapping("production_parfocal_mapping.json")
    print("Parfocal mapping saved to production_parfocal_mapping.json")

    return mapping
```

## 🚀 Basic Usage

### Step 6: Create Your First System

```python
# basic_usage.py
from autofocus.dual_lens_optimized import create_optimized_dual_lens_system, OptimizationLevel
from autofocus.parfocal_mapping_optimized import EnhancedParfocalMapping

# Import your hardware implementations
from your_camera import YourCameraController
from your_stage import YourStageController
from your_illumination import YourIlluminationController
from lens_config import create_production_lens_profiles

def create_production_system():
    """Create production dual-lens autofocus system."""

    # Step 1: Initialize hardware
    print("Initializing hardware...")
    camera = YourCameraController(your_camera_device)
    stage = YourStageController(your_stage_device)
    illumination = YourIlluminationController(your_illumination_device)

    # Step 2: Load lens profiles
    print("Loading lens profiles...")
    lens_a, lens_b = create_production_lens_profiles()

    # Step 3: Load parfocal mapping
    print("Loading parfocal mapping...")
    try:
        mapping = EnhancedParfocalMapping.load_mapping("production_parfocal_mapping.json")
        print(f"Loaded mapping with {mapping.rms_error_um:.3f}μm RMS error")
    except FileNotFoundError:
        print("No saved mapping found. Please run calibration first.")
        return None

    # Step 4: Create optimized system
    print("Creating optimized autofocus system...")
    system = create_optimized_dual_lens_system(
        camera=camera,
        stage_controller=stage,
        illumination=illumination,
        lens_a_profile=lens_a,
        lens_b_profile=lens_b,
        parfocal_mapping=mapping,
        optimization_level=OptimizationLevel.ULTRA_FAST  # For production
    )

    print("✓ Production system ready!")
    return system

def basic_autofocus_example():
    """Basic autofocus usage example."""

    system = create_production_system()
    if system is None:
        return

    try:
        print("\n" + "="*50)
        print("BASIC AUTOFOCUS EXAMPLE")
        print("="*50)

        # Example 1: Scanning autofocus
        print("\n1. Scanning autofocus with Lens-A...")
        x, y = 1000.0, 2000.0  # Target position

        z_a = system.autofocus_scanning(x_um=x, y_um=y)
        print(f"   Focused at ({x}, {y}) → z = {z_a:.2f}μm")

        # Example 2: Fast handoff to detailed lens
        print("\n2. Fast handoff A→B...")
        result = system.handoff_a_to_b_optimized(source_z_um=z_a)

        if result.success:
            print(f"   Handoff successful in {result.elapsed_ms:.0f}ms")
            print(f"   Mapping error: {result.mapping_error_um:.2f}μm")
            print(f"   Target position: {result.target_z_um:.2f}μm")

            # Show optimization details
            if result.concurrent_operations:
                print(f"   Optimizations: {', '.join(result.concurrent_operations)}")
        else:
            print(f"   Handoff failed: {result.flags}")
            return

        # Example 3: Detailed autofocus
        print("\n3. Detailed autofocus with Lens-B...")
        z_b = system.autofocus_detailed(
            x_um=x,
            y_um=y,
            z_guess_um=result.target_z_um
        )
        print(f"   Precise focus → z = {z_b:.3f}μm")

        # Example 4: Performance statistics
        print("\n4. Performance statistics...")
        stats = system.get_optimization_statistics()

        if stats.get('performance'):
            perf = stats['performance']
            print(f"   Average handoff time: {perf['avg_total_time_ms']:.0f}ms")
            print(f"   Target achievement: {perf['target_met_rate']*100:.1f}%")

        if stats.get('optimization_features'):
            opt = stats['optimization_features']
            print(f"   Cache hit rate: {opt['cache_hit_rate']*100:.1f}%")

        print("\n✓ Basic example completed successfully!")

    finally:
        # Always clean up
        system.close()
        print("\n✓ System closed")

if __name__ == "__main__":
    basic_autofocus_example()
```

## 🧪 Testing Your Setup

### Step 7: Validation and Testing

```python
# validation.py
from autofocus.dual_lens_qa import DualLensQAHarness, DualLensQAConfig

def validate_system_performance():
    """Validate system meets performance requirements."""

    system = create_production_system()
    if system is None:
        return

    try:
        print("\n" + "="*50)
        print("SYSTEM PERFORMANCE VALIDATION")
        print("="*50)

        # Configure validation
        qa_config = DualLensQAConfig(
            num_handoff_tests=20,           # Quick validation
            handoff_time_target_ms=300.0,   # Target ≤300ms
            mapping_accuracy_target_um=1.0, # Target ≤1.0μm
            surface_prediction_target_um=0.5,
            enable_temperature_tests=True
        )

        # Run validation
        print("Running QA validation...")
        qa_harness = DualLensQAHarness(qa_config)
        summary = qa_harness.run_full_validation(system)

        # Display results
        print(f"\nValidation Results:")
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Total Duration: {summary['total_duration_s']:.1f}s")

        # Individual test results
        for test_name, result in summary['test_summary'].items():
            status_icon = "✓" if result['status'] == 'PASS' else "✗"
            print(f"  {status_icon} {test_name}: {result['status']}")

        # Key metrics
        if 'key_metrics' in summary:
            metrics = summary['key_metrics']
            print(f"\nKey Performance Metrics:")
            print(f"  Average handoff time: {metrics['avg_handoff_time_ms']:.0f}ms")
            print(f"  P95 handoff time: {metrics['p95_handoff_time_ms']:.0f}ms")
            print(f"  Success rate: {metrics['handoff_success_rate']*100:.1f}%")

        # Recommendations
        if summary['overall_status'] == 'PASS':
            print("\n🎉 System validation PASSED!")
            print("   Your system is ready for production use.")
        else:
            print("\n⚠️  System validation FAILED")
            print("   Please review the test results and optimize configuration.")

    finally:
        system.close()

if __name__ == "__main__":
    validate_system_performance()
```

## 🔧 Troubleshooting Common Issues

### Issue 1: Slow Handoff Performance

```python
# If handoff times exceed 300ms, try these optimizations:

# 1. Increase focus speeds in lens profiles
lens_a.focus_speed_um_per_s = 600.0  # Increase from 500
lens_b.focus_speed_um_per_s = 400.0  # Increase from 300

# 2. Reduce settling times
lens_a.settle_time_ms = 3.0  # Reduce from 5ms
lens_b.settle_time_ms = 8.0  # Reduce from 10ms

# 3. Use ULTRA_FAST optimization level
system = create_optimized_dual_lens_system(
    # ... other parameters
    optimization_level=OptimizationLevel.ULTRA_FAST
)

# 4. Enable all optimizations
camera = OptimizedDualLensCameraController(
    lens_a, lens_b,
    enable_predictive_focus=True,      # Enable prediction
    enable_concurrent_operations=True   # Enable concurrency
)
```

### Issue 2: Poor Mapping Accuracy

```python
# If mapping errors exceed 1.0μm:

# 1. Increase calibration points
z_a_positions = np.linspace(-10, 10, 35)  # More points

# 2. Use more precise focus search during calibration
for z_b_test in np.linspace(-10, 10, 41):  # Finer search

# 3. Measure actual temperature during calibration
import your_temperature_sensor
temperature = your_temperature_sensor.get_temperature()

# 4. Enable adaptive learning
mapping.learning_rate = 0.15  # Increase learning rate
mapping.add_validation_point(z_a_measured, z_b_measured)
```

### Issue 3: Focus Metric Issues

```python
# If autofocus fails to find good focus:

# 1. Tune metric weights for your samples
lens_profile.metric_weights = {
    "tenengrad": 0.7,    # Increase for edge content
    "laplacian": 0.3,    # Decrease if noisy
    "brenner": 0.0       # Disable if not helpful
}

# 2. Adjust search parameters
config.search.coarse_step_um = 2.0    # Larger steps
config.search.fine_step_um = 0.5      # Larger fine steps
config.search.max_iterations = 5      # More iterations

# 3. Check illumination
# Ensure consistent, bright illumination for your samples
```

## 📈 Performance Optimization Tips

### Hardware Optimization

1. **Focus Control**: Use high-speed focus motors (≥300 μm/s)
2. **Lens Switching**: Minimize mechanical switching time (≤50ms)
3. **Camera**: Use high-speed cameras with minimal readout time
4. **Stage**: Ensure stable, vibration-free positioning

### Software Optimization

1. **Use ULTRA_FAST mode** for production scanning
2. **Enable all caching** for repeated operations
3. **Tune focus metrics** for your specific samples
4. **Regular calibration updates** for accuracy maintenance

### System Integration

1. **Temperature monitoring**: Enable real-time compensation
2. **Performance monitoring**: Set up continuous QA validation
3. **Error handling**: Implement robust recovery mechanisms
4. **Logging**: Enable detailed telemetry for troubleshooting

## 🚀 Next Steps

### Production Deployment

1. **Complete Integration**: Follow [Integration Guide](integration_guide.md)
2. **Performance Tuning**: See [Performance Guide](performance_guide.md)
3. **Advanced Features**: Review [Advanced Topics](advanced_topics.md)
4. **API Reference**: Explore [Complete API Documentation](api_reference.md)

### Code Examples

1. **Basic Examples**: [examples/basic/](examples/basic/)
2. **Advanced Examples**: [examples/advanced/](examples/advanced/)
3. **Integration Examples**: [examples/integration/](examples/integration/)

### Support Resources

- **Documentation**: Complete guides in `docs/` directory
- **Examples**: Working code in `examples/` directory
- **Demonstrations**: Performance demos in `scripts/` directory
- **Issues**: GitHub issues for bug reports and questions

---

**Congratulations!** 🎉 You've successfully set up the Dual-Lens Autofocus system. The system is now ready for production use with ultra-fast handoff performance and exceptional accuracy.

For advanced configuration and optimization, continue to the [Performance Guide](performance_guide.md) and [Integration Guide](integration_guide.md).