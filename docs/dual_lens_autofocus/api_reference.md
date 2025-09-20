# Dual-Lens Autofocus API Reference

## 📖 Overview

Complete API reference for the Dual-Lens Autofocus module. This document covers all public classes, methods, and interfaces with detailed parameter descriptions and usage examples.

## 📋 Table of Contents

- [Core Classes](#core-classes)
- [Optimization Classes](#optimization-classes)
- [Mapping Classes](#mapping-classes)
- [QA Framework](#qa-framework)
- [Data Types](#data-types)
- [Interfaces](#interfaces)
- [Enums](#enums)
- [Exceptions](#exceptions)

---

## 🏗️ Core Classes

### `LensProfile`

Represents the complete configuration for a single lens in the dual-lens system.

```python
@dataclass
class LensProfile:
    """Physical and optical properties of a lens."""
```

#### Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `lens_id` | `LensID` | Unique identifier for the lens | Required |
| `name` | `str` | Human-readable lens name | Required |
| `serial_number` | `Optional[str]` | Hardware serial number | `None` |
| `magnification` | `float` | Lens magnification (e.g., 20.0 for 20x) | `20.0` |
| `numerical_aperture` | `float` | Numerical aperture (e.g., 0.4) | `0.4` |
| `working_distance_mm` | `float` | Working distance in millimeters | `2.0` |
| `field_of_view_um` | `float` | Field of view in micrometers | `500.0` |
| `depth_of_field_um` | `float` | Depth of field in micrometers | `2.0` |
| `z_range_um` | `Tuple[float, float]` | Focus range (min, max) in micrometers | `(-50.0, 50.0)` |
| `z_resolution_um` | `float` | Focus resolution in micrometers | `0.1` |
| `parfocal_offset_um` | `float` | Offset from reference plane | `0.0` |
| `af_config` | `AutofocusConfig` | Autofocus algorithm configuration | Auto-generated |
| `metric_weights` | `Dict[str, float]` | Focus metric weights | `{"tenengrad": 0.4, "laplacian": 0.3, "brenner": 0.3}` |
| `preferred_illum_pattern` | `str` | Preferred illumination pattern | `"BRIGHTFIELD"` |
| `illum_intensity_factor` | `float` | Illumination intensity multiplier | `1.0` |
| `focus_speed_um_per_s` | `float` | Focus movement speed | `200.0` |
| `settle_time_ms` | `float` | Focus settling time in milliseconds | `10.0` |

#### Methods

##### `get_z_center() -> float`

Returns the center position of the focus range.

```python
lens = LensProfile(lens_id=LensID.LENS_A, name="20x Lens")
center = lens.get_z_center()  # Returns 0.0 for default range (-50, 50)
```

##### `is_z_in_range(z_um: float) -> bool`

Checks if a focus position is within the lens's valid range.

```python
lens = LensProfile(lens_id=LensID.LENS_A, name="20x Lens", z_range_um=(-25.0, 25.0))
valid = lens.is_z_in_range(10.0)  # Returns True
invalid = lens.is_z_in_range(30.0)  # Returns False
```

---

### `DualLensAutofocusManager`

Main controller for dual-lens autofocus operations.

```python
class DualLensAutofocusManager:
    """Main controller for dual-lens autofocus system."""
```

#### Constructor

```python
def __init__(self,
             camera: CameraInterface,
             stage_controller,
             illumination: IlluminationController,
             lens_a_profile: LensProfile,
             lens_b_profile: LensProfile,
             parfocal_mapping: Optional[ParfocalMapping] = None)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `camera` | `CameraInterface` | Camera hardware interface |
| `stage_controller` | `Any` | XYZ stage controller |
| `illumination` | `IlluminationController` | Illumination controller |
| `lens_a_profile` | `LensProfile` | Configuration for Lens-A (scanning) |
| `lens_b_profile` | `LensProfile` | Configuration for Lens-B (detailed) |
| `parfocal_mapping` | `Optional[ParfocalMapping]` | Cross-lens mapping (optional) |

#### Methods

##### `autofocus_scanning(x_um: float, y_um: float, z_guess_um: Optional[float] = None) -> float`

Performs autofocus optimized for scanning with Lens-A.

```python
# Move to position and focus with scanning lens
z_focused = system.autofocus_scanning(x_um=1000.0, y_um=2000.0, z_guess_um=0.0)
print(f"Focused at {z_focused:.2f} μm")
```

**Parameters:**
- `x_um`, `y_um`: Target XY position in micrometers
- `z_guess_um`: Initial focus guess (optional)

**Returns:** Final focus position in micrometers

##### `autofocus_detailed(x_um: float, y_um: float, z_guess_um: Optional[float] = None) -> float`

Performs high-precision autofocus with Lens-B.

```python
# High-precision focus with detailed lens
z_focused = system.autofocus_detailed(x_um=1000.0, y_um=2000.0, z_guess_um=2.5)
print(f"Precisely focused at {z_focused:.3f} μm")
```

##### `handoff_a_to_b(source_z_um: float) -> DualLensHandoffResult`

Fast handoff from Lens-A to Lens-B with parfocal mapping.

```python
# Perform A→B handoff
result = system.handoff_a_to_b(source_z_um=2.5)

if result.success:
    print(f"Handoff completed in {result.elapsed_ms:.0f}ms")
    print(f"Mapping error: {result.mapping_error_um:.2f}μm")
else:
    print(f"Handoff failed: {result.flags}")
```

**Parameters:**
- `source_z_um`: Focus position from Lens-A

**Returns:** `DualLensHandoffResult` with performance metrics

##### `handoff_b_to_a(source_z_um: float) -> DualLensHandoffResult`

Fast handoff from Lens-B to Lens-A.

```python
# Perform B→A handoff
result = system.handoff_b_to_a(source_z_um=-1.5)
```

##### `get_handoff_performance_stats() -> Dict[str, float]`

Gets handoff performance statistics.

```python
stats = system.get_handoff_performance_stats()
print(f"Average handoff time: {stats['avg_handoff_time_ms']:.0f}ms")
print(f"P95 handoff time: {stats['p95_handoff_time_ms']:.0f}ms")
print(f"Target achievement rate: {stats['target_met_rate']*100:.1f}%")
```

##### `update_temperature(temperature_c: float) -> None`

Updates system temperature for thermal compensation.

```python
# Update temperature for thermal drift compensation
system.update_temperature(25.5)  # 25.5°C
```

##### `close() -> None`

Clean up system resources.

```python
# Always close system when done
system.close()
```

---

## ⚡ Optimization Classes

### `OptimizedDualLensAutofocus`

Ultra-fast dual-lens autofocus system with advanced optimizations.

```python
class OptimizedDualLensAutofocus:
    """Ultra-fast dual-lens autofocus system optimized for ≤300ms handoffs."""
```

#### Constructor

```python
def __init__(self,
             camera: CameraInterface,
             stage_controller,
             illumination,
             lens_a_profile: LensProfile,
             lens_b_profile: LensProfile,
             parfocal_mapping: ParfocalMapping,
             optimization_level: OptimizationLevel = OptimizationLevel.FAST)
```

#### Methods

##### `handoff_a_to_b_optimized(source_z_um: float) -> OptimizedHandoffResult`

Ultra-fast async handoff with concurrent operations.

```python
# Ultra-fast optimized handoff
result = system.handoff_a_to_b_optimized(source_z_um=2.0)

print(f"Handoff time: {result.elapsed_ms:.0f}ms")
print(f"Timing breakdown:")
print(f"  Lens switch: {result.lens_switch_ms:.1f}ms")
print(f"  Focus move: {result.focus_move_ms:.1f}ms")
print(f"  Focus search: {result.focus_search_ms:.1f}ms")
print(f"  Validation: {result.validation_ms:.1f}ms")
print(f"Optimizations: {result.concurrent_operations}")
print(f"Cache hits: {result.cache_hits}")
```

##### `get_optimization_statistics() -> Dict[str, Any]`

Gets detailed optimization performance statistics.

```python
stats = system.get_optimization_statistics()

# Performance metrics
perf = stats['performance']
print(f"Average time: {perf['avg_total_time_ms']:.0f}ms")
print(f"P95 time: {perf['p95_total_time_ms']:.0f}ms")
print(f"Target met rate: {perf['target_met_rate']*100:.1f}%")

# Optimization features
opt = stats['optimization_features']
print(f"Cache hit rate: {opt['cache_hit_rate']*100:.1f}%")
print(f"Avg prediction confidence: {opt['avg_prediction_confidence']:.2f}")
```

---

### `OptimizedDualLensCameraController`

Ultra-fast camera controller with predictive operations and caching.

```python
class OptimizedDualLensCameraController:
    """Ultra-fast camera controller with predictive operations and caching."""
```

#### Constructor

```python
def __init__(self,
             lens_a_profile: LensProfile,
             lens_b_profile: LensProfile,
             enable_predictive_focus: bool = True,
             enable_concurrent_operations: bool = True)
```

#### Methods

##### `set_focus_concurrent(z_a_um: Optional[float] = None, z_b_um: Optional[float] = None) -> None`

Set focus positions for both lenses concurrently.

```python
# Set focus for both lenses simultaneously
camera.set_focus_concurrent(z_a_um=2.0, z_b_um=-1.5)
```

##### `get_dual_frame_optimized(...) -> DualLensFrame`

Ultra-fast dual frame capture with optimization.

```python
# Optimized dual frame capture
dual_frame = camera.get_dual_frame_optimized(
    lens_a_z_um=1.0,
    lens_b_z_um=0.5,
    mode=AcquisitionMode.ALTERNATING
)

print(f"Captured frames:")
print(f"  Lens-A: {dual_frame.lens_a_frame.shape}")
print(f"  Lens-B: {dual_frame.lens_b_frame.shape}")
```

##### `focus_bracketing_optimized(...) -> List[Tuple[float, np.ndarray]]`

Optimized focus bracketing with minimal settling.

```python
# Fast focus bracketing
sequence = camera.focus_bracketing_optimized(
    lens_id=LensID.LENS_B,
    center_z_um=0.0,
    range_um=4.0,
    num_steps=5
)

for i, (z_pos, frame) in enumerate(sequence):
    print(f"Frame {i+1}: z={z_pos:.2f}μm, shape={frame.shape}")
```

##### `predict_focus_with_surface(lens_id: LensID, x_um: float, y_um: float) -> Optional[float]`

Fast focus prediction using cached surface models.

```python
# Predictive focus for faster positioning
predicted_z = camera.predict_focus_with_surface(
    lens_id=LensID.LENS_A,
    x_um=1000.0,
    y_um=2000.0
)

if predicted_z is not None:
    print(f"Predicted focus: {predicted_z:.2f}μm")
```

##### `get_performance_stats() -> Dict[str, Any]`

Get detailed performance statistics.

```python
stats = camera.get_performance_stats()

# Timing statistics
for operation, timing in stats['timing_stats'].items():
    print(f"{operation}: avg={timing['avg_ms']:.1f}ms")

# Cache statistics
cache = stats['cache_stats']
print(f"Focus prediction cache: {cache['focus_prediction_cache_size']} entries")
```

---

## 🗺️ Mapping Classes

### `EnhancedParfocalMapping`

Enhanced parfocal mapping with adaptive accuracy and learning.

```python
@dataclass
class EnhancedParfocalMapping:
    """Enhanced parfocal mapping with adaptive accuracy and learning."""
```

#### Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `model_type` | `MappingModel` | Mapping model type | `MappingModel.ADAPTIVE` |
| `coefficients` | `List[float]` | Polynomial coefficients | `[0.0, 1.0, 0.0, 0.0]` |
| `calibration_timestamp` | `float` | Calibration time | `0.0` |
| `calibration_temperature_c` | `float` | Calibration temperature | `23.0` |
| `num_calibration_points` | `int` | Number of calibration points | `0` |
| `rms_error_um` | `float` | RMS mapping error | `0.0` |
| `temp_coeff_linear_um_per_c` | `float` | Temperature coefficient | `0.0` |
| `learning_rate` | `float` | Adaptive learning rate | `0.1` |
| `confidence_threshold` | `float` | Confidence threshold | `0.95` |

#### Methods

##### `map_lens_a_to_b(z_a_um: float, temperature_c: float = 23.0) -> float`

Enhanced A→B mapping with adaptive accuracy.

```python
# Map from Lens-A to Lens-B position
z_b = mapping.map_lens_a_to_b(z_a_um=2.5, temperature_c=25.0)
print(f"A→B mapping: {2.5:.1f} → {z_b:.2f} μm")
```

##### `map_lens_b_to_a(z_b_um: float, temperature_c: float = 23.0) -> float`

Enhanced B→A mapping with iterative solution.

```python
# Map from Lens-B to Lens-A position
z_a = mapping.map_lens_b_to_a(z_b_um=-1.5, temperature_c=25.0)
print(f"B→A mapping: {-1.5:.1f} → {z_a:.2f} μm")
```

##### `calibrate_enhanced(calibration_data: List[Tuple[float, float, float]]) -> Dict[str, Any]`

Enhanced calibration with model selection and validation.

```python
# Calibration data: (z_a, z_b, temperature)
calibration_data = [
    (0.0, 2.1, 23.0),
    (1.0, 3.0, 23.0),
    (2.0, 4.1, 23.0),
    # ... more points
]

result = mapping.calibrate_enhanced(calibration_data)
print(f"Best model: {result['model_type']}")
print(f"RMS error: {result['rms_error_um']:.3f}μm")
print(f"Max error: {result['max_error_um']:.3f}μm")
```

##### `add_validation_point(z_a_um: float, z_b_actual: float) -> None`

Add validation point for adaptive learning.

```python
# Add real measurement for learning
mapping.add_validation_point(z_a_um=1.5, z_b_actual=2.9)
```

##### `get_mapping_accuracy_report() -> Dict[str, Any]`

Get comprehensive accuracy report.

```python
report = mapping.get_mapping_accuracy_report()

# Calibration info
cal = report['calibration']
print(f"Model type: {cal['model_type']}")
print(f"RMS error: {cal['rms_error_um']:.3f}μm")
print(f"Calibration age: {cal['calibration_age_hours']:.1f} hours")

# Recent performance
perf = report['recent_performance']
print(f"Recent average error: {perf['recent_avg_error_um']:.3f}μm")
print(f"Accuracy trend: {perf['accuracy_trend']}")

# Confidence metrics
conf = report['confidence_metrics']
print(f"Overall confidence: {conf['overall_confidence']:.2f}")
```

##### `save_mapping(filepath: str) -> None`

Save mapping parameters to file.

```python
# Save calibrated mapping
mapping.save_mapping("production_mapping.json")
```

##### `load_mapping(filepath: str) -> EnhancedParfocalMapping`

Load mapping parameters from file.

```python
# Load saved mapping
mapping = EnhancedParfocalMapping.load_mapping("production_mapping.json")
```

---

## 🧪 QA Framework

### `DualLensQAHarness`

QA harness for dual-lens autofocus system validation.

```python
class DualLensQAHarness:
    """QA harness for dual-lens autofocus system validation."""
```

#### Constructor

```python
def __init__(self, config: DualLensQAConfig)
```

#### Methods

##### `run_full_validation(system: DualLensAutofocusManager) -> Dict[str, Any]`

Run complete dual-lens validation suite.

```python
# Configure QA testing
qa_config = DualLensQAConfig(
    num_handoff_tests=50,
    handoff_time_target_ms=300.0,
    mapping_accuracy_target_um=1.0
)

# Run validation
qa_harness = DualLensQAHarness(qa_config)
summary = qa_harness.run_full_validation(system)

print(f"Overall status: {summary['overall_status']}")
print(f"Total duration: {summary['total_duration_s']:.1f}s")

# Individual test results
for test_name, result in summary['test_summary'].items():
    print(f"{test_name}: {result['status']} ({result['duration_s']:.1f}s)")

# Key performance metrics
if 'key_metrics' in summary:
    metrics = summary['key_metrics']
    print(f"Average handoff time: {metrics['avg_handoff_time_ms']:.0f}ms")
    print(f"P95 handoff time: {metrics['p95_handoff_time_ms']:.0f}ms")
```

---

## 📊 Data Types

### `DualLensHandoffResult`

Result from a cross-lens handoff operation.

```python
@dataclass
class DualLensHandoffResult:
    """Result of cross-lens handoff operation."""

    # Handoff success
    success: bool
    elapsed_ms: float

    # Focus positions
    source_lens: LensID
    target_lens: LensID
    source_z_um: float
    target_z_um: float

    # Mapping validation
    predicted_z_um: float
    actual_z_um: float
    mapping_error_um: float

    # Performance metrics
    focus_metric_source: float
    focus_metric_target: float
    metric_consistency: float

    # Status flags
    flags: List[str] = field(default_factory=list)
```

### `OptimizedHandoffResult`

Enhanced handoff result with optimization metrics.

```python
@dataclass
class OptimizedHandoffResult(DualLensHandoffResult):
    """Enhanced handoff result with optimization metrics."""

    # Timing breakdown
    lens_switch_ms: float = 0.0
    focus_move_ms: float = 0.0
    focus_search_ms: float = 0.0
    illumination_setup_ms: float = 0.0
    validation_ms: float = 0.0

    # Optimization details
    optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
    concurrent_operations: List[str] = field(default_factory=list)
    cache_hits: int = 0
    prediction_confidence: float = 0.0
```

### `DualLensFrame`

Frame data from dual-lens system.

```python
@dataclass
class DualLensFrame:
    """Frame data from dual-lens system."""

    # Image data
    lens_a_frame: Optional[np.ndarray] = None
    lens_b_frame: Optional[np.ndarray] = None

    # Acquisition metadata
    timestamp: float = field(default_factory=time.time)
    active_lens: Optional[LensID] = None
    acquisition_mode: AcquisitionMode = AcquisitionMode.SINGLE_LENS

    # Focus positions
    lens_a_z_um: Optional[float] = None
    lens_b_z_um: Optional[float] = None

    # Quality metrics
    lens_a_metric: Optional[float] = None
    lens_b_metric: Optional[float] = None
```

---

## 🔌 Interfaces

### `CameraInterface`

Protocol for camera interface in dual-lens system.

```python
class CameraInterface(Protocol):
    """Protocol for camera interface in dual-lens system."""

    def get_frame(self) -> np.ndarray:
        """Capture frame from current active lens."""
        ...

    def set_active_lens(self, lens_id: LensID) -> None:
        """Switch to specified lens."""
        ...

    def get_active_lens(self) -> LensID:
        """Get currently active lens."""
        ...

    def set_focus(self, z_um: float) -> None:
        """Set focus position for active lens."""
        ...

    def get_focus(self) -> float:
        """Get focus position of active lens."""
        ...
```

#### Implementation Example

```python
class YourCameraController(CameraInterface):
    """Your camera implementation."""

    def __init__(self):
        self.hardware = your_camera_hardware
        self.active_lens = LensID.LENS_A

    def get_frame(self) -> np.ndarray:
        return self.hardware.capture_frame()

    def set_active_lens(self, lens_id: LensID) -> None:
        self.hardware.switch_lens(lens_id.value)
        self.active_lens = lens_id

    def get_active_lens(self) -> LensID:
        return self.active_lens

    def set_focus(self, z_um: float) -> None:
        self.hardware.set_focus_position(z_um)

    def get_focus(self) -> float:
        return self.hardware.get_focus_position()
```

---

## 🏷️ Enums

### `LensID`

Lens identifiers for dual-lens system.

```python
class LensID(Enum):
    """Lens identifiers for dual-lens system."""
    LENS_A = "lens_a"  # 10-20x low NA for scanning
    LENS_B = "lens_b"  # 40-60x high NA for detailed analysis
```

### `OptimizationLevel`

Performance optimization levels.

```python
class OptimizationLevel(Enum):
    """Performance optimization levels."""
    STANDARD = "standard"      # Balanced performance and accuracy
    FAST = "fast"             # Optimized for speed
    ULTRA_FAST = "ultra_fast" # Maximum speed, minimal accuracy trade-off
```

### `MappingModel`

Parfocal mapping model types.

```python
class MappingModel(Enum):
    """Parfocal mapping model types."""
    LINEAR = "linear"         # Linear mapping: z_b = a + b*z_a
    QUADRATIC = "quadratic"   # Quadratic: z_b = a + b*z_a + c*z_a²
    CUBIC = "cubic"          # Cubic: z_b = a + b*z_a + c*z_a² + d*z_a³
    ADAPTIVE = "adaptive"    # Auto-select best model
```

### `AcquisitionMode`

Camera acquisition modes for dual-lens system.

```python
class AcquisitionMode(Enum):
    """Camera acquisition modes for dual-lens system."""
    SINGLE_LENS = "single_lens"          # Use one lens at a time
    SIMULTANEOUS = "simultaneous"        # Capture from both lenses simultaneously
    ALTERNATING = "alternating"          # Rapid alternation between lenses
    FOCUS_BRACKETING = "focus_bracketing" # Multiple focus positions per lens
```

---

## ⚠️ Exceptions

### `ParfocalMappingError`

Raised when parfocal mapping operations fail.

```python
class ParfocalMappingError(Exception):
    """Raised when parfocal mapping operations fail."""
    pass

# Usage
try:
    result = mapping.calibrate_enhanced(insufficient_data)
except ParfocalMappingError as e:
    print(f"Mapping calibration failed: {e}")
```

### `HandoffTimeoutError`

Raised when handoff operations exceed time limits.

```python
class HandoffTimeoutError(Exception):
    """Raised when handoff operations exceed time limits."""
    pass

# Usage
try:
    result = system.handoff_a_to_b(source_z_um=2.0)
    if result.elapsed_ms > 500:  # Custom timeout
        raise HandoffTimeoutError(f"Handoff took {result.elapsed_ms}ms")
except HandoffTimeoutError as e:
    print(f"Handoff timeout: {e}")
```

### `FocusRangeError`

Raised when focus positions are outside valid range.

```python
class FocusRangeError(Exception):
    """Raised when focus positions are outside valid range."""
    pass

# Usage
try:
    if not lens_profile.is_z_in_range(z_position):
        raise FocusRangeError(f"Position {z_position}μm outside range {lens_profile.z_range_um}")
except FocusRangeError as e:
    print(f"Focus range error: {e}")
```

---

## 🏭 Factory Functions

### `create_optimized_dual_lens_system(...)`

Factory function to create optimized dual-lens system.

```python
def create_optimized_dual_lens_system(
    camera: CameraInterface,
    stage_controller,
    illumination,
    lens_a_profile: LensProfile,
    lens_b_profile: LensProfile,
    parfocal_mapping: ParfocalMapping,
    optimization_level: OptimizationLevel = OptimizationLevel.FAST
) -> OptimizedDualLensAutofocus
```

### `create_enhanced_parfocal_mapping(...)`

Factory function to create enhanced parfocal mapping from calibration data.

```python
def create_enhanced_parfocal_mapping(
    calibration_data: List[Tuple[float, float, float]]
) -> EnhancedParfocalMapping
```

---

## 📝 Usage Examples

### Complete System Setup

```python
from autofocus.dual_lens_optimized import create_optimized_dual_lens_system, OptimizationLevel
from autofocus.dual_lens import LensProfile, LensID
from autofocus.parfocal_mapping_optimized import create_enhanced_parfocal_mapping

# 1. Create lens profiles
lens_a = LensProfile(
    lens_id=LensID.LENS_A,
    name="Scanning 20x/0.4",
    magnification=20.0,
    numerical_aperture=0.4,
    focus_speed_um_per_s=500.0  # Fast for scanning
)

lens_b = LensProfile(
    lens_id=LensID.LENS_B,
    name="Detail 60x/0.8",
    magnification=60.0,
    numerical_aperture=0.8,
    focus_speed_um_per_s=300.0  # Precise for detail
)

# 2. Create parfocal mapping
calibration_data = [
    # (z_a, z_b, temperature)
    (-5.0, -2.5, 23.0),
    (0.0, 2.1, 23.0),
    (5.0, 7.2, 23.0),
    # ... more calibration points
]
mapping = create_enhanced_parfocal_mapping(calibration_data)

# 3. Create optimized system
system = create_optimized_dual_lens_system(
    camera=your_camera,
    stage_controller=your_stage,
    illumination=your_illumination,
    lens_a_profile=lens_a,
    lens_b_profile=lens_b,
    parfocal_mapping=mapping,
    optimization_level=OptimizationLevel.ULTRA_FAST
)

# 4. Perform operations
try:
    # Scanning autofocus
    z_a = system.autofocus_scanning(x_um=1000, y_um=2000)

    # Fast handoff to detailed lens
    result = system.handoff_a_to_b_optimized(z_a)

    if result.success:
        print(f"Handoff: {result.elapsed_ms:.0f}ms, error: {result.mapping_error_um:.2f}μm")

    # Detailed autofocus
    z_b = system.autofocus_detailed(x_um=1000, y_um=2000, z_guess_um=result.target_z_um)

finally:
    system.close()
```

### Performance Monitoring

```python
import time
import threading

def monitor_performance(system, interval_s=60):
    """Monitor system performance continuously."""
    while True:
        stats = system.get_optimization_statistics()

        if stats.get('performance'):
            perf = stats['performance']
            print(f"[{time.strftime('%H:%M:%S')}] Performance Update:")
            print(f"  Avg handoff: {perf['avg_total_time_ms']:.0f}ms")
            print(f"  Target met: {perf['target_met_rate']*100:.1f}%")

            # Alert if performance degrades
            if perf['avg_total_time_ms'] > 300:
                print("⚠️  WARNING: Handoff time exceeding target!")

            if perf['target_met_rate'] < 0.9:
                print("⚠️  WARNING: Target achievement rate below 90%!")

        time.sleep(interval_s)

# Start monitoring in background
monitor_thread = threading.Thread(target=monitor_performance, args=(system,), daemon=True)
monitor_thread.start()
```

---

This API reference provides complete documentation for all public interfaces in the Dual-Lens Autofocus module. For usage examples and tutorials, see the [examples directory](examples/) and [getting started guide](getting_started.md).