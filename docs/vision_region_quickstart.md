# Vision Region Module Quickstart

This guide shows how to run the blood smear region detector, export mask pyramids, and generate scan plans.

```python
from pathlib import Path
import numpy as np

from bloodwork_ai.vision.region import (
    CalibrationEstimator,
    RegionDetectionService,
    RegionModuleConfig,
)
from bloodwork_ai.vision.region.config import (
    SurveyCaptureConfig,
    MaskExportConfig,
    PlannerConfig,
)
from bloodwork_ai.vision.region.geometry.calibration import GridObservation
from bloodwork_ai.vision.region.types import SurveyFrame, SurveyStream, StageCoordinate

# --- 1. Calibrate ---
observations: list[GridObservation] = load_calibration_images()  # user-defined helper
calibration_estimator = CalibrationEstimator()
calibration_pack, metrics = calibration_estimator.estimate_from_grid(observations)
print("Reprojection µm:", metrics.reprojection_error_um)

# --- 2. Configure ---
config = RegionModuleConfig(
    survey=SurveyCaptureConfig(
        lens_id="A_10x",
        tile_size_um=1200,
        stride_um=900,
        illumination_profile="BF_white",
    ),
    mask_export=MaskExportConfig(output_dir=Path("outputs/masks")),
    planner=PlannerConfig(tile_size_um=200,
                          stride_um=150,
                          keepout_margin_um=150,
                          af_seeds_per_tile=3),
)

# --- 3. Build service ---
service = RegionDetectionService.from_calibration(config, calibration_pack, default_lens="A_10x")

# --- 4. Acquire survey stream ---
frames = [
    SurveyFrame(
        image=np.load("survey_tile.npy"),
        stage_position=StageCoordinate(x_um=0, y_um=0),
        exposure_ms=10.0,
        timestamp=0.0,
    ),
]
survey_stream = SurveyStream(frames=frames, lens_id="A_10x", slide_id="slide_001")

# --- 5. Run detection ---
result = service.process_survey(survey_stream, slide_id="slide_001")
print("Tiles planned:", len(result.scan_plan or []))
print("Mask saved to:", result.artifacts.mask_pyramid_path)
```

See `src/bloodwork_ai/vision/region/qa.py` for evaluation helpers and `region/io.py` for serialization utilities.
