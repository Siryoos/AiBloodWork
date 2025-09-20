"""Scan planning utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from skimage import morphology
from skimage.feature import peak_local_max

from .config import PlannerConfig
from .types import CoordinateMapper, PixelCoordinate


@dataclass
class ScanPlanner:
    """Generate comprehensive scan plans from detected smear regions."""

    config: PlannerConfig
    mapper: CoordinateMapper
    _pixel_size_cache: Dict[str, float] = field(default_factory=dict)

    def plan_tiles(self, mask: np.ndarray, lens_id: str, quality_map: Optional[np.ndarray] = None) -> List[Dict[str, object]]:
        """Generate optimized scan plan with priorities and AF seeds."""
        if mask.sum() == 0:
            return []

        pixel_size_um = self._pixel_size_um(lens_id)

        # Apply keepout margins
        working_mask = self._apply_keepout_margins(mask, pixel_size_um)

        # Generate tile grid
        tile_positions = self._generate_tile_grid(working_mask, pixel_size_um)

        # Generate tiles with priorities and AF seeds
        tiles = []
        for tile_pos in tile_positions:
            tile_data = self._create_tile(tile_pos, mask, quality_map, lens_id, pixel_size_um)
            if tile_data is not None:
                tiles.append(tile_data)

        # Sort by priority
        tiles = self._sort_by_priority(tiles)

        return tiles

    def _apply_keepout_margins(self, mask: np.ndarray, pixel_size_um: float) -> np.ndarray:
        """Apply keepout margins to mask."""
        if self.config.keepout_margin_um <= 0:
            return mask

        margin_px = max(1, int(round(self.config.keepout_margin_um / pixel_size_um)))

        # Erode mask to create keepout
        kernel = morphology.disk(margin_px)
        eroded_mask = morphology.erosion(mask > 0.5, kernel)

        return eroded_mask.astype(np.float32)

    def _generate_tile_grid(self, mask: np.ndarray, pixel_size_um: float) -> List[Tuple[int, int]]:
        """Generate grid of tile positions with optimal coverage."""
        h, w = mask.shape

        tile_size_px = max(1, int(round(self.config.tile_size_um / pixel_size_um)))
        stride_px = max(1, int(round(self.config.stride_um / pixel_size_um)))

        if h < tile_size_px or w < tile_size_px:
            return []

        positions = []

        for y in range(0, h - tile_size_px + 1, stride_px):
            for x in range(0, w - tile_size_px + 1, stride_px):
                # Check if tile overlaps with mask
                tile_region = mask[y:y+tile_size_px, x:x+tile_size_px]

                if tile_region.size > 0 and np.mean(tile_region) > 0.1:
                    positions.append((y, x))

        return positions

    def _create_tile(
        self,
        position: Tuple[int, int],
        mask: np.ndarray,
        quality_map: Optional[np.ndarray],
        lens_id: str,
        pixel_size_um: float,
    ) -> Optional[Dict[str, object]]:
        """Create tile data structure with stage coordinates, priority, and AF seeds."""
        y, x = position
        tile_size_px = max(1, int(round(self.config.tile_size_um / pixel_size_um)))

        # Extract tile region
        tile_mask = mask[y:y+tile_size_px, x:x+tile_size_px]
        tile_quality = quality_map[y:y+tile_size_px, x:x+tile_size_px] if quality_map is not None else tile_mask

        # Skip if insufficient coverage
        if np.mean(tile_mask) < 0.1:
            return None

        # Convert to stage coordinates
        pixel_coord = PixelCoordinate(u=float(x + tile_size_px / 2), v=float(y + tile_size_px / 2), lens_id=lens_id)
        stage_coord = self.mapper.to_stage(pixel_coord)

        # Compute priority
        priority = self._compute_tile_priority(tile_mask, tile_quality)

        # Generate AF seeds
        af_seeds = self._generate_af_seeds(tile_mask, tile_quality, (y, x), lens_id, pixel_size_um)

        # Estimate Z position (placeholder)
        z_pred = 0.0

        return {
            "x_um": stage_coord.x_um,
            "y_um": stage_coord.y_um,
            "z_pred_um": z_pred,
            "lens_id": lens_id,
            "priority": priority,
            "af_seeds": af_seeds,
            "tile_size_um": self.config.tile_size_um,
            "coverage": float(np.mean(tile_mask)),
            "quality_score": float(np.mean(tile_quality)) if tile_quality.size > 0 else 0.0,
        }

    def _compute_tile_priority(self, tile_mask: np.ndarray, tile_quality: np.ndarray) -> float:
        """Compute tile priority based on quality map and policy."""
        if tile_mask.size == 0:
            return 0.0

        coverage = np.mean(tile_mask)
        quality_score = np.mean(tile_quality) if tile_quality.size > 0 else coverage

        if self.config.priority_policy == "monolayer_first":
            # High priority for good monolayer regions
            priority = 0.7 * quality_score + 0.3 * coverage
        elif self.config.priority_policy == "coverage_first":
            # High priority for high coverage
            priority = 0.8 * coverage + 0.2 * quality_score
        elif self.config.priority_policy == "balanced":
            # Balanced approach
            priority = 0.5 * coverage + 0.5 * quality_score
        else:
            # Default: quality-first
            priority = quality_score

        return float(np.clip(priority, 0.0, 1.0))

    def _generate_af_seeds(
        self,
        tile_mask: np.ndarray,
        tile_quality: np.ndarray,
        offset: Tuple[int, int],
        lens_id: str,
        pixel_size_um: float,
    ) -> List[Dict[str, object]]:
        """Generate autofocus seed points within the tile."""
        if tile_mask.size == 0:
            return []

        seeds = []
        y_offset, x_offset = offset

        # Find best regions for AF within tile
        if tile_quality.size > 0:
            quality_for_af = tile_quality * tile_mask
        else:
            quality_for_af = tile_mask.copy()

        # Generate seeds based on local maxima
        seed_positions = self._find_af_seed_positions(quality_for_af, self.config.af_seeds_per_tile)

        for seed_pos in seed_positions:
            seed_y, seed_x = seed_pos
            global_y = y_offset + seed_y
            global_x = x_offset + seed_x

            # Convert to stage coordinates
            pixel_coord = PixelCoordinate(u=float(global_x + 0.5), v=float(global_y + 0.5), lens_id=lens_id)
            stage_coord = self.mapper.to_stage(pixel_coord)

            roi_size_px = max(16, int(round(80.0 / pixel_size_um)))

            seed = {
                "x_um": stage_coord.x_um,
                "y_um": stage_coord.y_um,
                "z_pred_um": 0.0,  # Placeholder
                "roi_size_px": roi_size_px,
                "quality_score": float(quality_for_af[seed_y, seed_x]) if quality_for_af.size > 0 else 0.0,
            }
            seeds.append(seed)

        return seeds

    def _find_af_seed_positions(self, quality_map: np.ndarray, num_seeds: int) -> List[Tuple[int, int]]:
        """Find optimal positions for AF seeds using local maxima detection."""
        if quality_map.size == 0 or num_seeds <= 0:
            return []

        h, w = quality_map.shape

        min_distance = max(3, min(h, w) // 20)

        peaks = peak_local_max(
            quality_map,
            num_peaks=num_seeds,
            min_distance=min_distance,
            threshold_abs=0.1,
            exclude_border=False,
        )

        if peaks.size == 0:
            flat_indices = np.argsort(quality_map.ravel())[::-1]
            positions: List[Tuple[int, int]] = []
            for idx in flat_indices:
                if quality_map.flat[idx] <= 0.1:
                    break
                y, x = np.unravel_index(idx, quality_map.shape)
                positions.append((int(y), int(x)))
                if len(positions) >= num_seeds:
                    break
            return positions

        return [(int(y), int(x)) for y, x in peaks[:num_seeds]]

    def _pixel_size_um(self, lens_id: str) -> float:
        """Estimate pixel size in microns using mapper differentials."""
        if lens_id in self._pixel_size_cache:
            return self._pixel_size_cache[lens_id]

        base = PixelCoordinate(u=0.0, v=0.0, lens_id=lens_id)
        try:
            base_stage = self.mapper.to_stage(base)
            dx_stage = self.mapper.to_stage(PixelCoordinate(u=1.0, v=0.0, lens_id=lens_id))
            dy_stage = self.mapper.to_stage(PixelCoordinate(u=0.0, v=1.0, lens_id=lens_id))
        except Exception:
            pixel_size = 1.0
        else:
            delta_x = np.hypot(dx_stage.x_um - base_stage.x_um, dx_stage.y_um - base_stage.y_um)
            delta_y = np.hypot(dy_stage.x_um - base_stage.x_um, dy_stage.y_um - base_stage.y_um)
            candidates = [d for d in (delta_x, delta_y) if d > 1e-6]
            pixel_size = float(np.mean(candidates)) if candidates else 1.0

        pixel_size = max(pixel_size, 1e-3)
        self._pixel_size_cache[lens_id] = pixel_size
        return pixel_size

    def _sort_by_priority(self, tiles: List[Dict[str, object]]) -> List[Dict[str, object]]:
        """Sort tiles by priority and add sequence numbers."""
        sorted_tiles = sorted(tiles, key=lambda t: t["priority"], reverse=True)

        # Add sequence numbers
        for i, tile in enumerate(sorted_tiles):
            tile["sequence"] = i

        return sorted_tiles

    def generate_raster_scan(self, mask: np.ndarray, lens_id: str) -> List[Dict[str, object]]:
        """Generate simple raster scan pattern."""
        tiles = []
        h, w = mask.shape

        pixel_size_um = self._pixel_size_um(lens_id)

        tile_size_px = max(1, int(round(self.config.tile_size_um / pixel_size_um)))
        stride_px = max(1, int(round(self.config.stride_um / pixel_size_um)))

        sequence = 0
        for y in range(0, h - tile_size_px + 1, stride_px):
            for x in range(0, w - tile_size_px + 1, stride_px):
                tile_region = mask[y:y+tile_size_px, x:x+tile_size_px]

                if tile_region.size > 0 and np.mean(tile_region) > 0.05:
                    pixel_coord = PixelCoordinate(u=float(x + tile_size_px / 2), v=float(y + tile_size_px / 2), lens_id=lens_id)
                    stage_coord = self.mapper.to_stage(pixel_coord)

                    tile = {
                        "x_um": stage_coord.x_um,
                        "y_um": stage_coord.y_um,
                        "z_pred_um": 0.0,
                        "lens_id": lens_id,
                        "sequence": sequence,
                        "priority": 1.0,
                        "af_seeds": [],
                    }
                    tiles.append(tile)
                    sequence += 1

        return tiles

    def optimize_scan_path(self, tiles: List[Dict[str, object]]) -> List[Dict[str, object]]:
        """Optimize scan path to minimize stage movement."""
        if len(tiles) <= 1:
            return tiles

        # Simple optimization: sort by Y first, then X (raster pattern)
        optimized = sorted(tiles, key=lambda t: (t["y_um"], t["x_um"]))

        # Update sequence numbers
        for i, tile in enumerate(optimized):
            tile["sequence"] = i

        return optimized

    def estimate_scan_time(self, tiles: List[Dict[str, object]],
                          time_per_tile_ms: float = 500.0) -> Dict[str, float]:
        """Estimate total scan time."""
        if not tiles:
            return {"total_time_s": 0.0, "movement_time_s": 0.0, "acquisition_time_s": 0.0}

        # Acquisition time
        acquisition_time_s = len(tiles) * time_per_tile_ms / 1000.0

        # Movement time estimation (simple)
        movement_time_s = 0.0
        if len(tiles) > 1:
            total_distance_um = 0.0
            for i in range(1, len(tiles)):
                prev_tile = tiles[i-1]
                curr_tile = tiles[i]
                dx = curr_tile["x_um"] - prev_tile["x_um"]
                dy = curr_tile["y_um"] - prev_tile["y_um"]
                distance = np.sqrt(dx**2 + dy**2)
                total_distance_um += distance

            # Assume 10 mm/s average speed
            movement_time_s = total_distance_um / 10000.0

        total_time_s = acquisition_time_s + movement_time_s

        return {
            "total_time_s": total_time_s,
            "movement_time_s": movement_time_s,
            "acquisition_time_s": acquisition_time_s,
            "num_tiles": len(tiles),
        }


class PriorityPolicy:
    """Priority policies for scan planning."""

    @staticmethod
    def monolayer_first(coverage: float, quality: float, edge_distance: float) -> float:
        """High priority for monolayer regions."""
        return 0.6 * quality + 0.3 * coverage + 0.1 * edge_distance

    @staticmethod
    def coverage_first(coverage: float, quality: float, edge_distance: float) -> float:
        """High priority for high coverage areas."""
        return 0.7 * coverage + 0.2 * quality + 0.1 * edge_distance

    @staticmethod
    def edge_first(coverage: float, quality: float, edge_distance: float) -> float:
        """High priority for feathered edge regions."""
        return 0.1 * coverage + 0.3 * quality + 0.6 * (1.0 - edge_distance)  # Invert edge distance

    @staticmethod
    def balanced(coverage: float, quality: float, edge_distance: float) -> float:
        """Balanced priority scoring."""
        return 0.4 * coverage + 0.4 * quality + 0.2 * edge_distance


class ScanPathOptimizer:
    """Optimize scan paths for minimal stage movement."""

    @staticmethod
    def traveling_salesman_simple(tiles: List[Dict[str, object]]) -> List[Dict[str, object]]:
        """Simple greedy TSP approximation."""
        if len(tiles) <= 1:
            return tiles

        # Start with first tile
        remaining = tiles[1:]
        path = [tiles[0]]
        current_pos = (tiles[0]["x_um"], tiles[0]["y_um"])

        while remaining:
            # Find nearest tile
            distances = []
            for tile in remaining:
                dx = tile["x_um"] - current_pos[0]
                dy = tile["y_um"] - current_pos[1]
                distance = np.sqrt(dx**2 + dy**2)
                distances.append(distance)

            nearest_idx = np.argmin(distances)
            nearest_tile = remaining.pop(nearest_idx)
            path.append(nearest_tile)
            current_pos = (nearest_tile["x_um"], nearest_tile["y_um"])

        # Update sequence numbers
        for i, tile in enumerate(path):
            tile["sequence"] = i

        return path

    @staticmethod
    def snake_pattern(tiles: List[Dict[str, object]]) -> List[Dict[str, object]]:
        """Generate snake/serpentine scan pattern."""
        if not tiles:
            return tiles

        # Group tiles by Y coordinate (rows)
        tiles_by_y = {}
        for tile in tiles:
            y_key = round(tile["y_um"], 1)  # Group by 0.1µm precision
            if y_key not in tiles_by_y:
                tiles_by_y[y_key] = []
            tiles_by_y[y_key].append(tile)

        # Sort rows by Y coordinate
        sorted_y_keys = sorted(tiles_by_y.keys())

        # Create snake pattern
        snake_path = []
        for i, y_key in enumerate(sorted_y_keys):
            row_tiles = tiles_by_y[y_key]

            # Sort by X coordinate
            row_tiles.sort(key=lambda t: t["x_um"])

            # Reverse every other row for snake pattern
            if i % 2 == 1:
                row_tiles.reverse()

            snake_path.extend(row_tiles)

        # Update sequence numbers
        for i, tile in enumerate(snake_path):
            tile["sequence"] = i

        return snake_path
