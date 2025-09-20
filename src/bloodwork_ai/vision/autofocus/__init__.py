"""Auto-focus module for camera control.

Provides a complete autofocus system optimized for blood smear analysis,
including hardware interfaces, focus metrics, search strategies, surface
prediction, thermal compensation, validation, and diagnostics.
"""

# Core interfaces
from .camera_interface import CameraInterface
from .stage_interface import StageInterface

# Focus metrics
from .metrics import (
    variance_of_laplacian,
    tenengrad,
    brenner_gradient,
    high_frequency_energy,
    normalized_dct_energy,
    metric_fusion,
)

# Search strategies
from .strategies.contrast import ContrastMaximizationStrategy
from .strategies.coarse_to_fine import CoarseToFineStrategy

# Surface modeling
from .surface import (
    FocusSurfaceModel,
    FocusSurfaceBuilder,
    fit_plane,
    fit_quad,
    fit_rbf,
    select_rbf_epsilon,
    load_surface,
    save_surface,
    save_surface_cache,
    load_surface_from_cache,
)

# Thermal compensation and drift tracking
from .thermal import (
    TemperatureSensor,
    ThermalCompensationModel,
    DriftTracker,
    GlobalFocusOffset,
)

# Illumination control
from .illumination import (
    IlluminationController,
    IlluminationPattern,
    IlluminationPatterns,
    IlluminationManager,
    MockIlluminationController,
)

# Configuration and calibration
from .config import (
    AutofocusConfig,
    MetricConfig,
    SearchConfig,
    HardwareConfig,
    SurfaceConfig,
    ThermalConfig,
    ValidationConfig,
    CalibrationData,
    AutofocusCalibrator,
)

# Validation and testing
from .validation import (
    FocusAccuracyTest,
    MTFMeasurement,
    ImageQualityValidator,
    AutofocusTestSuite,
)

# Diagnostics
from .diagnostics import DiagnosticsLogger

# Computational photography integration
from .comp_photo import (
    CompPhotoConfig,
    CompPhotoAutofocus,
    IlluminationMode,
    ReconstructionMethod,
    ChromaticFocusCompensation,
    FocusStackPlan,
)

# Production APIs
from .api import (
    AutofocusAPIServer,
    AutofocusRequest,
    AutofocusResponse,
    ROISpec,
    MetricResult,
    create_autofocus_api,
)

# Telemetry and logging
from .telemetry import (
    ProductionTelemetryLogger,
    RegulatoryLogger,
    TelemetryEvent,
)

# Outlier detection and failure handling
from .outlier_detection import (
    OutlierDetector,
    OutlierDetectionConfig,
    FailureHandler,
    FailureHandlingConfig,
    OutlierType,
    FailureMode,
)

# QA harness
from .qa_harness import (
    AutofocusQAHarness,
    KPIThresholds,
    TestConfiguration,
    TestReport,
    create_qa_harness,
)

# Main controllers
from .controller import AutoFocusController
from .blood_smear_autofocus import BloodSmearAutofocus, create_blood_smear_autofocus

__all__ = [
    # Core interfaces
    "CameraInterface",
    "StageInterface",

    # Focus metrics
    "variance_of_laplacian",
    "tenengrad",
    "brenner_gradient",
    "high_frequency_energy",
    "normalized_dct_energy",
    "metric_fusion",
    "block_dct_energy",

    # Search strategies
    "ContrastMaximizationStrategy",
    "CoarseToFineStrategy",

    # Surface modeling
    "FocusSurfaceModel",
    "FocusSurfaceBuilder",
    "fit_plane",
    "fit_quad",
    "fit_rbf",
    "select_rbf_epsilon",
    "load_surface",
    "save_surface",
    "save_surface_cache",
    "load_surface_from_cache",

    # Thermal compensation
    "TemperatureSensor",
    "ThermalCompensationModel",
    "DriftTracker",
    "GlobalFocusOffset",

    # Illumination
    "IlluminationController",
    "IlluminationPattern",
    "IlluminationPatterns",
    "IlluminationManager",
    "MockIlluminationController",

    # Configuration
    "AutofocusConfig",
    "MetricConfig",
    "SearchConfig",
    "HardwareConfig",
    "SurfaceConfig",
    "ThermalConfig",
    "ValidationConfig",
    "CalibrationData",
    "AutofocusCalibrator",

    # Validation
    "FocusAccuracyTest",
    "MTFMeasurement",
    "ImageQualityValidator",
    "AutofocusTestSuite",

    # Diagnostics
    "DiagnosticsLogger",

    # Computational photography
    "CompPhotoConfig",
    "CompPhotoAutofocus",
    "IlluminationMode",
    "ReconstructionMethod",
    "ChromaticFocusCompensation",
    "FocusStackPlan",

    # Production APIs
    "AutofocusAPIServer",
    "AutofocusRequest",
    "AutofocusResponse",
    "ROISpec",
    "MetricResult",
    "create_autofocus_api",

    # Telemetry and logging
    "ProductionTelemetryLogger",
    "RegulatoryLogger",
    "TelemetryEvent",

    # Outlier detection
    "OutlierDetector",
    "OutlierDetectionConfig",
    "FailureHandler",
    "FailureHandlingConfig",
    "OutlierType",
    "FailureMode",

    # QA harness
    "AutofocusQAHarness",
    "KPIThresholds",
    "TestConfiguration",
    "TestReport",
    "create_qa_harness",

    # Controllers
    "AutoFocusController",
    "BloodSmearAutofocus",
    "create_blood_smear_autofocus",
]
