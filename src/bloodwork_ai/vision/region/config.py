"""Configuration objects for the region detection module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class SurveyCaptureConfig:
    """Low-magnification survey acquisition parameters."""

    lens_id: str
    tile_size_um: float
    stride_um: float
    illumination_profile: str
    roi_size_px: Optional[int] = None
    max_exposure_ms: Optional[float] = None
    max_tiles: Optional[int] = None


@dataclass
class PreprocessConfig:
    """Pre-processing options applied before detection."""

    enable_flat_field: bool = True
    enable_stain_normalization: bool = True
    enable_glare_inpainting: bool = True
    flat_field_profile: Optional[Path] = None
    stain_reference_profile: Optional[Path] = None


@dataclass
class CoarseDetectionConfig:
    """Parameters for coarse smear detection."""

    model_path: Optional[Path] = None
    threshold: float = 0.5
    min_component_area_um2: float = 2.5e6
    smoothing_radius_um: float = 50.0


@dataclass
class SegmentationConfig:
    """Fine segmentation configuration."""

    cnn_model_path: Optional[Path] = None
    cnn_input_scale: float = 0.25
    contrast_weights: Tuple[float, float] = (0.6, 0.4)
    otsu_bias: float = 0.0
    graph_cut_lambda: float = 10.0
    remove_holes_area_um2: float = 3.0e4


@dataclass
class EdgeAnalysisConfig:
    """Feathered edge and quality map configuration."""

    gradient_window_um: float = 80.0
    feathered_edge_search_margin_um: float = 500.0
    quality_map_scale: float = 0.5
    monolayer_density_threshold: float = 0.4


@dataclass
class MaskExportConfig:
    """Mask pyramid export options."""

    output_dir: Path
    pyramid_scales: Tuple[int, ...] = (1, 4, 16)
    compression: str = "zstd"
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class CalibrationConfig:
    """Geometry calibration configuration."""

    fiducial_slide_id: str
    max_reprojection_error_um: float = 2.0
    stage_scale_prior_um_per_count: Optional[Tuple[float, float]] = None
    cross_lens_ransac_threshold_um: float = 3.0


@dataclass
class PlannerConfig:
    """Scan planning parameters."""

    tile_size_um: float
    stride_um: float
    keepout_margin_um: float = 200.0
    priority_policy: str = "monolayer_first"
    af_seeds_per_tile: int = 3


@dataclass
class RegionModuleConfig:
    """Aggregated configuration for the region detection module."""

    survey: SurveyCaptureConfig
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    coarse: CoarseDetectionConfig = field(default_factory=CoarseDetectionConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    edge: EdgeAnalysisConfig = field(default_factory=EdgeAnalysisConfig)
    mask_export: Optional[MaskExportConfig] = None
    planner: Optional[PlannerConfig] = None

    def with_defaults(self, **overrides: object) -> "RegionModuleConfig":
        """Return a copy with selected fields overridden."""
        data = {
            "survey": self.survey,
            "preprocess": self.preprocess,
            "coarse": self.coarse,
            "segmentation": self.segmentation,
            "edge": self.edge,
            "mask_export": self.mask_export,
            "planner": self.planner,
        }
        data.update(overrides)
        return RegionModuleConfig(**data)  # type: ignore[arg-type]

