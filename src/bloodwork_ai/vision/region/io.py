"""Input/output helpers for region detection artifacts."""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import zarr
from zarr.storage import FSStore

from .types import CalibrationPack, RegionDetectionResult


class ZarrMaskWriter:
    """Write mask pyramids to Zarr/NGFF format."""

    def __init__(self, compression: str = "zstd", compression_level: int = 3):
        self.compression = compression
        self.compression_level = compression_level

    def write_mask_pyramid(self, mask: np.ndarray, output_path: Path,
                          pyramid_scales: List[int] = [1, 4, 16],
                          metadata: Optional[Dict[str, Any]] = None) -> Path:
        """Write mask pyramid to Zarr format with NGFF metadata."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create zarr store
        store = FSStore(str(output_path))
        root = zarr.group(store=store, overwrite=True)

        # Create pyramid levels
        for i, scale in enumerate(pyramid_scales):
            level_name = str(i)

            # Downsample mask
            if scale == 1:
                level_mask = mask
            else:
                level_mask = self._downsample_mask(mask, scale)

            # Create zarr array
            level_array = root.create_dataset(
                level_name,
                data=level_mask,
                chunks=(512, 512),
                dtype=np.uint8,
                compressor=zarr.Blosc(cname=self.compression, clevel=self.compression_level)
            )

            # Add level metadata
            level_array.attrs['scale'] = scale
            level_array.attrs['shape'] = level_mask.shape

        # Add NGFF metadata
        self._add_ngff_metadata(root, pyramid_scales, metadata or {})

        return output_path

    def _downsample_mask(self, mask: np.ndarray, scale: int) -> np.ndarray:
        """Downsample mask by given scale factor."""
        h, w = mask.shape
        new_h, new_w = h // scale, w // scale

        if new_h == 0 or new_w == 0:
            return np.zeros((1, 1), dtype=mask.dtype)

        # Simple downsampling by averaging
        downsampled = np.zeros((new_h, new_w), dtype=np.float32)

        for y in range(new_h):
            for x in range(new_w):
                y_start, y_end = y * scale, min((y + 1) * scale, h)
                x_start, x_end = x * scale, min((x + 1) * scale, w)

                region = mask[y_start:y_end, x_start:x_end]
                downsampled[y, x] = np.mean(region)

        # Convert back to binary
        return (downsampled > 0.5).astype(np.uint8)

    def _add_ngff_metadata(self, root: zarr.Group, scales: List[int], metadata: Dict[str, Any]) -> None:
        """Add NGFF metadata to zarr group."""
        # NGFF multiscales metadata
        multiscales = [{
            "version": "0.4",
            "name": "smear_mask",
            "axes": [
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"}
            ],
            "datasets": [
                {
                    "path": str(i),
                    "coordinateTransformations": [{
                        "type": "scale",
                        "scale": [float(scale), float(scale)]
                    }]
                }
                for i, scale in enumerate(scales)
            ]
        }]

        root.attrs['multiscales'] = multiscales
        root.attrs['metadata'] = metadata


class ZarrMaskReader:
    """Read mask pyramids from Zarr/NGFF format."""

    def __init__(self, zarr_path: Path):
        self.zarr_path = zarr_path
        self._store = None
        self._root = None

    def __enter__(self):
        self._store = FSStore(str(self.zarr_path))
        self._root = zarr.group(store=self._store, mode='r')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._store:
            self._store.close()

    def read_level(self, level: int = 0) -> np.ndarray:
        """Read mask at specified pyramid level."""
        if self._root is None:
            raise RuntimeError("Reader not opened. Use as context manager.")

        level_name = str(level)
        if level_name not in self._root:
            raise ValueError(f"Level {level} not found in pyramid")

        return np.array(self._root[level_name])

    def get_metadata(self) -> Dict[str, Any]:
        """Get NGFF metadata."""
        if self._root is None:
            raise RuntimeError("Reader not opened. Use as context manager.")

        return dict(self._root.attrs.get('metadata', {}))

    def get_scales(self) -> List[int]:
        """Get available pyramid scales."""
        if self._root is None:
            raise RuntimeError("Reader not opened. Use as context manager.")

        multiscales = self._root.attrs.get('multiscales', [])
        if not multiscales:
            return [1]

        datasets = multiscales[0].get('datasets', [])
        scales = []

        for dataset in datasets:
            transforms = dataset.get('coordinateTransformations', [])
            for transform in transforms:
                if transform.get('type') == 'scale':
                    scale = transform.get('scale', [1, 1])
                    scales.append(int(scale[0]))
                    break

        return scales if scales else [1]


def save_scan_plan(plan: List[Dict[str, Any]], output_path: Path) -> None:
    """Save scan plan as JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in plan:
            # Convert numpy types to native Python types
            serializable_item = _make_serializable(item)
            f.write(json.dumps(serializable_item) + "\n")


def load_scan_plan(path: Path) -> List[Dict[str, Any]]:
    """Load scan plan from JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_polygon_geojson(
    polygon: np.ndarray,
    output_path: Path,
    *,
    properties: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist polygon vertices as a single-feature GeoJSON file."""
    if polygon.size == 0:
        raise ValueError("Polygon array is empty")

    coords = polygon.astype(float).tolist()
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    feature = {
        "type": "Feature",
        "properties": properties or {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [coords],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": [feature]}, f, indent=2)


def save_calibration_pack(calibration: CalibrationPack, output_path: Path) -> None:
    """Save calibration pack to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    data = {
        'intrinsics': {k: v.tolist() for k, v in calibration.intrinsics.items()},
        'distortion': {k: v.tolist() for k, v in calibration.distortion.items()},
        'extrinsics': {
            k: [R.tolist(), t.tolist()]
            for k, (R, t) in calibration.extrinsics.items()
        },
        'homographies': {k: v.tolist() for k, v in calibration.homographies.items()},
        'stage_scale_um_per_count': calibration.stage_scale_um_per_count,
        'cross_lens_transform': calibration.cross_lens_transform.tolist() if calibration.cross_lens_transform is not None else None,
        'metadata': calibration.metadata
    }

    with output_path.open('w') as f:
        json.dump(data, f, indent=2)


def load_calibration_pack(path: Path) -> CalibrationPack:
    """Load calibration pack from file."""
    with path.open('r') as f:
        data = json.load(f)

    return CalibrationPack(
        intrinsics={k: np.array(v) for k, v in data['intrinsics'].items()},
        distortion={k: np.array(v) for k, v in data['distortion'].items()},
        extrinsics={
            k: (np.array(R), np.array(t))
            for k, (R, t) in data['extrinsics'].items()
        },
        homographies={k: np.array(v) for k, v in data.get('homographies', {}).items()},
        stage_scale_um_per_count=tuple(data['stage_scale_um_per_count']),
        cross_lens_transform=np.array(data['cross_lens_transform']) if data.get('cross_lens_transform') is not None else None,
        metadata=data.get('metadata', {})
    )


def save_region_result(result: RegionDetectionResult, output_dir: Path) -> None:
    """Save complete region detection result."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    meta_path = output_dir / "region_result.json"
    with meta_path.open("w", encoding="utf-8") as f:
        metadata = {
            "slide_id": result.slide_id,
            "artifacts": {
                "smear_mask": bool(result.artifacts.smear_mask is not None),
                "smear_polygon": bool(result.artifacts.smear_polygon is not None),
                "feathered_edge": bool(result.artifacts.feathered_edge is not None),
                "quality_map": bool(result.artifacts.quality_map is not None),
            },
            "scan_plan_tiles": len(result.scan_plan) if result.scan_plan else 0,
        }
        json.dump(metadata, f, indent=2)

    # Save arrays
    if result.artifacts.smear_mask is not None:
        np.save(output_dir / "smear_mask.npy", result.artifacts.smear_mask)

    if result.artifacts.smear_polygon is not None:
        np.save(output_dir / "smear_polygon.npy", result.artifacts.smear_polygon)

    if result.artifacts.feathered_edge is not None:
        np.save(output_dir / "feathered_edge.npy", result.artifacts.feathered_edge)

    if result.artifacts.quality_map is not None:
        np.save(output_dir / "quality_map.npy", result.artifacts.quality_map)

    # Save scan plan
    if result.scan_plan:
        save_scan_plan(result.scan_plan, output_dir / "scan_plan.jsonl")

    # Save calibration
    save_calibration_pack(result.calibration, output_dir / "calibration.json")


def load_region_result(input_dir: Path) -> RegionDetectionResult:
    """Load complete region detection result."""
    # Load metadata
    meta_path = input_dir / "region_result.json"
    with meta_path.open("r") as f:
        metadata = json.load(f)

    # Load arrays
    artifacts = {}

    mask_path = input_dir / "smear_mask.npy"
    if mask_path.exists():
        artifacts['smear_mask'] = np.load(mask_path)

    polygon_path = input_dir / "smear_polygon.npy"
    if polygon_path.exists():
        artifacts['smear_polygon'] = np.load(polygon_path)

    edge_path = input_dir / "feathered_edge.npy"
    if edge_path.exists():
        artifacts['feathered_edge'] = np.load(edge_path)

    quality_path = input_dir / "quality_map.npy"
    if quality_path.exists():
        artifacts['quality_map'] = np.load(quality_path)

    # Load scan plan
    scan_plan = None
    scan_plan_path = input_dir / "scan_plan.jsonl"
    if scan_plan_path.exists():
        scan_plan = load_scan_plan(scan_plan_path)

    # Load calibration
    calibration = load_calibration_pack(input_dir / "calibration.json")

    # Create result object (simplified - would need proper reconstruction)
    from .types import RegionDetectionArtifacts
    from .geometry import HomographyCoordinateMapper

    detection_artifacts = RegionDetectionArtifacts(**artifacts)

    default_lens = next(iter(calibration.homographies)) if calibration.homographies else None
    mapper = HomographyCoordinateMapper.from_calibration(calibration, default_lens)

    return RegionDetectionResult(
        slide_id=metadata['slide_id'],
        calibration=calibration,
        mapper=mapper,
        artifacts=detection_artifacts,
        scan_plan=scan_plan
    )


def export_masks_to_geojson(mask: np.ndarray, output_path: Path,
                           pixel_to_world_transform: Optional[np.ndarray] = None) -> None:
    """Export mask contours to GeoJSON format."""
    import cv2

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find contours
    mask_uint8 = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features = []

    for i, contour in enumerate(contours):
        # Simplify contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, epsilon, True)

        # Convert to coordinate list
        coords = simplified.reshape(-1, 2).tolist()

        # Apply coordinate transform if provided
        if pixel_to_world_transform is not None:
            transformed_coords = []
            for x, y in coords:
                point = np.array([x, y, 1.0])
                world_point = pixel_to_world_transform @ point
                transformed_coords.append([world_point[0], world_point[1]])
            coords = transformed_coords

        # Close polygon
        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])

        feature = {
            "type": "Feature",
            "properties": {
                "id": i,
                "area": float(cv2.contourArea(contour)),
                "perimeter": float(cv2.arcLength(contour, True))
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords]
            }
        }
        features.append(feature)

    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }

    with output_path.open('w') as f:
        json.dump(geojson_data, f, indent=2)


def _make_serializable(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    else:
        return obj


def create_ome_zarr_metadata(shape: tuple, scales: List[int],
                           pixel_size_um: float = 1.0,
                           name: str = "blood_smear_mask") -> Dict[str, Any]:
    """Create OME-Zarr metadata for mask pyramids."""
    metadata = {
        "multiscales": [{
            "version": "0.4",
            "name": name,
            "axes": [
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"}
            ],
            "datasets": [
                {
                    "path": str(i),
                    "coordinateTransformations": [{
                        "type": "scale",
                        "scale": [pixel_size_um * scale, pixel_size_um * scale]
                    }]
                }
                for i, scale in enumerate(scales)
            ]
        }],
        "omero": {
            "channels": [{
                "label": "mask",
                "color": "00FF00"
            }]
        }
    }

    return metadata


class RegionResultExporter:
    """Export region detection results in various formats."""

    def __init__(self, result: RegionDetectionResult):
        self.result = result

    def export_summary_report(self, output_path: Path) -> None:
        """Export summary report as JSON."""
        mask = self.result.artifacts.smear_mask
        quality_map = self.result.artifacts.quality_map

        summary = {
            "slide_id": self.result.slide_id,
            "detection_summary": {
                "smear_detected": mask is not None and mask.sum() > 0,
                "smear_area_px": int(mask.sum()) if mask is not None else 0,
                "image_coverage_percent": float(mask.mean() * 100) if mask is not None else 0.0,
            },
            "quality_metrics": {},
            "scan_plan_summary": {
                "total_tiles": len(self.result.scan_plan) if self.result.scan_plan else 0,
            }
        }

        if quality_map is not None:
            summary["quality_metrics"] = {
                "mean_quality": float(quality_map[mask > 0].mean()) if mask is not None and mask.sum() > 0 else 0.0,
                "high_quality_area_percent": float((quality_map > 0.7).sum() / max(1, mask.sum()) * 100) if mask is not None else 0.0,
            }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w') as f:
            json.dump(summary, f, indent=2)

    def export_all_formats(self, output_dir: Path) -> None:
        """Export in all supported formats."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Standard format
        save_region_result(self.result, output_dir / "standard")

        # Zarr pyramid
        if self.result.artifacts.smear_mask is not None:
            writer = ZarrMaskWriter()
            writer.write_mask_pyramid(
                self.result.artifacts.smear_mask,
                output_dir / "zarr" / "smear_mask.zarr",
                pyramid_scales=[1, 2, 4, 8, 16]
            )

        # GeoJSON
        if self.result.artifacts.smear_mask is not None:
            export_masks_to_geojson(
                self.result.artifacts.smear_mask,
                output_dir / "geojson" / "smear_contours.geojson"
            )

        # Summary report
        self.export_summary_report(output_dir / "summary_report.json")
