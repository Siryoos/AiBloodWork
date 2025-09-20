"""Utilities for exporting mask pyramids."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
from skimage.transform import resize

from .config import MaskExportConfig
from .io import ZarrMaskWriter, create_ome_zarr_metadata


@dataclass
class MaskPyramidWriter:
    """Persist smear masks in multiscale formats (NumPy and Zarr)."""

    config: MaskExportConfig

    def write(self, mask: np.ndarray, slide_id: str) -> Path:
        """Write mask pyramid to disk, returning the output directory."""
        output_dir = self.config.output_dir / slide_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save NumPy pyramid
        self._write_numpy_pyramid(mask, output_dir)

        # Optionally write Zarr/NGFF pyramid for interoperability
        try:
            writer = ZarrMaskWriter()
            zarr_path = output_dir / "smear_mask.zarr"
            writer.write_mask_pyramid(
                mask,
                zarr_path,
                pyramid_scales=list(self.config.pyramid_scales),
                metadata=self.config.metadata,
            )
        except Exception:
            # Fallback silently; NumPy pyramid already stored
            pass

        return output_dir

    def _write_numpy_pyramid(self, mask: np.ndarray, output_dir: Path) -> None:
        np.save(output_dir / "smear_mask_level1.npy", mask.astype(np.uint8))
        for scale in self.config.pyramid_scales[1:]:
            level = self._downsample(mask, scale)
            np.save(output_dir / f"smear_mask_level{scale}.npy", level.astype(np.uint8))

    @staticmethod
    def _downsample(mask: np.ndarray, scale: int) -> np.ndarray:
        if scale <= 1:
            return mask
        new_shape = (max(1, mask.shape[0] // scale), max(1, mask.shape[1] // scale))
        if new_shape == mask.shape:
            return mask
        resized = resize(mask, new_shape, order=1, mode="reflect", anti_aliasing=False)
        return (resized > 0.5).astype(mask.dtype)
