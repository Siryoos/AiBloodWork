from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Union, Tuple
from pathlib import Path


@dataclass
class MetricConfig:
    """Configuration for focus metrics."""

    primary_metric: str = "tenengrad"  # Primary metric to use
    fusion_weights: Dict[str, float] = field(default_factory=lambda: {
        'tenengrad': 0.4,
        'hf_energy': 0.3,
        'brenner': 0.2,
        'laplacian': 0.1
    })
    tenengrad_ksize: int = 3
    hf_cutoff_ratio: float = 0.25
    dct_cutoff_ratio: float = 0.25


@dataclass
class SearchConfig:
    """Configuration for autofocus search strategies."""

    # Coarse-to-fine parameters
    bracket_um: float = 8.0
    coarse_step_um: float = 2.0
    fine_step_um: float = 0.3
    max_micro_iters: int = 20

    # ROI configuration for multi-point sampling
    roi_fractions: Tuple[Tuple[float, float, float, float], ...] = (
        (0.35, 0.35, 0.30, 0.30),  # center
        (0.10, 0.10, 0.25, 0.25),  # top-left
        (0.65, 0.10, 0.25, 0.25),  # top-right
        (0.10, 0.65, 0.25, 0.25),  # bottom-left
    )

    # Contrast maximization parameters
    contrast_step_um: float = 5.0
    min_step_um: float = 0.5
    max_iters: int = 50


@dataclass
class HardwareConfig:
    """Configuration for hardware interfaces."""

    # Camera settings
    settle_time_s: float = 0.02
    focus_range_um: Tuple[float, float] = (-50.0, 50.0)

    # Stage settings
    stage_timeout_s: float = 5.0
    xy_tolerance_um: float = 1.0

    # Illumination settings
    led_channels: int = 8
    default_intensity: float = 0.5
    illumination_settle_time_s: float = 0.02


@dataclass
class SurfaceConfig:
    """Configuration for focus surface modeling."""

    model_type: str = "quad"  # "plane", "quad", or "rbf"
    grid_points: int = 4  # Grid size for surface sampling
    rbf_epsilon: float = 200.0
    rbf_regularization: float = 1e-8

    # Cross-validation parameters for RBF
    rbf_epsilon_candidates: Tuple[float, ...] = (50.0, 100.0, 200.0, 400.0, 800.0)


@dataclass
class ThermalConfig:
    """Configuration for thermal compensation."""

    enable_compensation: bool = True
    thermal_coeff_um_per_c: float = 0.5  # Initial estimate
    ref_temperature_c: float = 20.0
    update_interval_s: float = 300.0  # 5 minutes
    calibration_interval_s: float = 300.0
    max_history: int = 100


@dataclass
class ValidationConfig:
    """Configuration for validation and testing."""

    # Accuracy requirements
    max_focus_error_um: float = 1.0
    max_repeatability_um: float = 0.5
    min_mtf50_ratio: float = 0.95

    # Test parameters
    accuracy_test_positions: Tuple[float, ...] = (-10, -5, 0, 5, 10, 15)
    repeatability_tests: int = 10

    # Image quality validation
    enable_mtf_validation: bool = True
    mtf_edge_angle_deg: float = 5.0


@dataclass
class AutofocusConfig:
    """Complete autofocus system configuration."""

    metric: MetricConfig = field(default_factory=MetricConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    surface: SurfaceConfig = field(default_factory=SurfaceConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    # Global settings
    timeout_s: Optional[float] = 30.0
    enable_diagnostics: bool = True
    diagnostics_path: Optional[str] = None

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AutofocusConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        # Handle nested dataclass creation
        config = cls()

        if 'metric' in data:
            config.metric = MetricConfig(**data['metric'])
        if 'search' in data:
            config.search = SearchConfig(**data['search'])
        if 'hardware' in data:
            config.hardware = HardwareConfig(**data['hardware'])
        if 'surface' in data:
            config.surface = SurfaceConfig(**data['surface'])
        if 'thermal' in data:
            config.thermal = ThermalConfig(**data['thermal'])
        if 'validation' in data:
            config.validation = ValidationConfig(**data['validation'])

        # Set global settings
        for key in ['timeout_s', 'enable_diagnostics', 'diagnostics_path']:
            if key in data:
                setattr(config, key, data[key])

        return config

    @classmethod
    def create_blood_smear_config(cls) -> "AutofocusConfig":
        """Create configuration optimized for blood smear analysis."""
        config = cls()

        # Optimize metrics for blood smears
        config.metric.primary_metric = "metric_fusion"
        config.metric.fusion_weights = {
            'tenengrad': 0.5,  # High weight on edge contrast
            'hf_energy': 0.3,  # Good for RBC edges
            'brenner': 0.15,   # Rapid computation
            'laplacian': 0.05  # Backup metric
        }

        # Optimize search for thin specimens
        config.search.bracket_um = 6.0  # Smaller bracket for thin smears
        config.search.coarse_step_um = 1.5
        config.search.fine_step_um = 0.2

        # Tighter validation for clinical use
        config.validation.max_focus_error_um = 0.8
        config.validation.max_repeatability_um = 0.4

        return config

    @classmethod
    def create_development_config(cls) -> "AutofocusConfig":
        """Create configuration for development and testing."""
        config = cls()

        # Enable all diagnostics
        config.enable_diagnostics = True
        config.diagnostics_path = "./autofocus_diagnostics.csv"

        # More conservative search for testing
        config.search.bracket_um = 12.0
        config.search.max_iters = 100

        # Enable comprehensive validation
        config.validation.enable_mtf_validation = True
        config.validation.repeatability_tests = 20

        return config


@dataclass
class CalibrationData:
    """Stores calibration data for the autofocus system."""

    timestamp: float

    # Focus range calibration
    focus_min_um: float
    focus_max_um: float
    focus_home_um: float

    # Metric calibration
    metric_baseline: Dict[str, float] = field(default_factory=dict)
    optimal_weights: Dict[str, float] = field(default_factory=dict)

    # Surface model validation
    surface_accuracy_um: Optional[float] = None
    surface_model_rmse: Optional[float] = None

    # Thermal calibration
    thermal_coeff_um_per_c: Optional[float] = None
    thermal_r_squared: Optional[float] = None

    # MTF reference
    reference_mtf50: Optional[float] = None
    mtf_target_z: Optional[float] = None

    def save(self, path: Union[str, Path]) -> None:
        """Save calibration data to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CalibrationData":
        """Load calibration data from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class AutofocusCalibrator:
    """Handles calibration procedures for the autofocus system."""

    def __init__(self, config: AutofocusConfig):
        self.config = config
        self.calibration_data: Optional[CalibrationData] = None

    def calibrate_focus_range(self, camera_interface) -> Tuple[float, float, float]:
        """Calibrate the usable focus range.

        Returns:
            Tuple of (min_focus, max_focus, home_focus)
        """
        device_min, device_max = camera_interface.get_focus_range()

        # Test actual usable range by looking for reasonable image quality
        test_positions = np.linspace(device_min, device_max, 20)
        valid_positions = []

        from .metrics import tenengrad

        for z in test_positions:
            camera_interface.set_focus(z)
            time.sleep(self.config.hardware.settle_time_s)

            try:
                frame = camera_interface.get_frame()
                if frame is not None and frame.size > 0:
                    score = tenengrad(frame)
                    if score > 0:  # Basic sanity check
                        valid_positions.append(z)
            except Exception:
                continue

        if len(valid_positions) < 2:
            # Fallback to device limits
            return device_min, device_max, (device_min + device_max) / 2

        focus_min = min(valid_positions)
        focus_max = max(valid_positions)
        focus_home = (focus_min + focus_max) / 2

        return focus_min, focus_max, focus_home

    def calibrate_metrics(self, camera_interface, z_positions: Optional[List[float]] = None):
        """Calibrate focus metrics at different positions."""
        if z_positions is None:
            z_positions = np.linspace(-10, 10, 11).tolist()

        from .metrics import tenengrad, variance_of_laplacian, brenner_gradient, high_frequency_energy

        metrics = {
            'tenengrad': tenengrad,
            'laplacian': variance_of_laplacian,
            'brenner': brenner_gradient,
            'hf_energy': high_frequency_energy
        }

        baselines = {}

        for name, metric_fn in metrics.items():
            values = []
            for z in z_positions:
                camera_interface.set_focus(z)
                time.sleep(self.config.hardware.settle_time_s)

                frame = camera_interface.get_frame()
                value = metric_fn(frame)
                values.append(value)

            # Use median as baseline to avoid outliers
            baselines[name] = float(np.median(values))

        return baselines

    def run_full_calibration(self,
                           camera_interface,
                           stage_interface=None,
                           save_path: Optional[Union[str, Path]] = None) -> CalibrationData:
        """Run complete calibration procedure."""
        import time

        calibration = CalibrationData(timestamp=time.time())

        # Calibrate focus range
        print("Calibrating focus range...")
        focus_min, focus_max, focus_home = self.calibrate_focus_range(camera_interface)
        calibration.focus_min_um = focus_min
        calibration.focus_max_um = focus_max
        calibration.focus_home_um = focus_home

        # Set home position
        camera_interface.set_focus(focus_home)
        time.sleep(self.config.hardware.settle_time_s)

        # Calibrate metrics
        print("Calibrating focus metrics...")
        baselines = self.calibrate_metrics(camera_interface)
        calibration.metric_baseline = baselines

        # Save if path provided
        if save_path:
            calibration.save(save_path)
            print(f"Calibration saved to {save_path}")

        self.calibration_data = calibration
        return calibration