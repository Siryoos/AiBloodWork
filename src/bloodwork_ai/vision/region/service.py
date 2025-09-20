"""Region detection service orchestrating the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import RegionModuleConfig
from .detect_coarse import CoarseSmearDetector
from .edge import EdgeAnalyzer
from .geometry.mapper import HomographyCoordinateMapper
from .mask_pyramid import MaskPyramidWriter
from .plan import ScanPlanner
from .preprocess import apply_preprocessing
from .segment import HybridSegmenter
from .types import (
    CalibrationPack,
    CoordinateMapper,
    RegionDetectionArtifacts,
    RegionDetectionResult,
    SurveyStream,
)


@dataclass
class RegionDetectionService:
    """High-level orchestrator for smear detection and planning."""

    config: RegionModuleConfig
    calibration: CalibrationPack
    mapper: CoordinateMapper

    @classmethod
    def from_calibration(
        cls,
        config: RegionModuleConfig,
        calibration: CalibrationPack,
        *,
        default_lens: Optional[str] = None,
    ) -> "RegionDetectionService":
        """Construct service using homography mapper derived from calibration."""
        mapper = HomographyCoordinateMapper.from_calibration(calibration, default_lens)
        return cls(config=config, calibration=calibration, mapper=mapper)

    def process_survey(self, survey: SurveyStream, slide_id: str) -> RegionDetectionResult:
        coarse_detector = CoarseSmearDetector(self.config.coarse)
        segmenter = HybridSegmenter(self.config.segmentation)
        edge_analyzer = EdgeAnalyzer(self.config.edge)

        survey_frames = list(survey.frames)
        if not survey_frames:
            raise ValueError("Survey stream is empty")
        image = survey_frames[0].image
        preprocessed = apply_preprocessing(image, self.config.preprocess)
        coarse_mask = coarse_detector.detect(preprocessed)
        fine_mask = segmenter.segment(preprocessed, coarse_mask)
        smear_polygon = self._extract_smear_polygon(fine_mask)
        feathered_edge, quality_map = edge_analyzer.analyze(fine_mask)

        artifacts = RegionDetectionArtifacts(
            smear_mask=fine_mask,
            smear_polygon=smear_polygon,
            feathered_edge=feathered_edge,
            quality_map=quality_map,
        )

        if self.config.mask_export:
            writer = MaskPyramidWriter(self.config.mask_export)
            artifacts.mask_pyramid_path = writer.write(fine_mask, slide_id)

        scan_plan = None
        if self.config.planner:
            planner = ScanPlanner(self.config.planner, self.mapper)
            scan_plan = planner.plan_tiles(fine_mask, survey.lens_id, quality_map=quality_map)

        return RegionDetectionResult(
            slide_id=slide_id,
            calibration=self.calibration,
            mapper=self.mapper,
            artifacts=artifacts,
            scan_plan=scan_plan,
        )

    def _extract_smear_polygon(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Extract outer boundary polygon from smear mask."""
        if mask.sum() == 0:
            return None

        mask_uint = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)

        return approx.reshape(-1, 2).astype(np.float32)
