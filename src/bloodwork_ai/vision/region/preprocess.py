"""Image preprocessing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage
from skimage import color, exposure

from .config import PreprocessConfig


def apply_preprocessing(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """Apply comprehensive preprocessing pipeline for blood smear analysis."""
    processed = image.astype(np.float32)

    if config.enable_flat_field:
        processed = apply_flat_field_correction(processed, config.flat_field_profile)

    if config.enable_stain_normalization:
        processed = apply_stain_normalization(processed, config.stain_reference_profile)

    if config.enable_glare_inpainting:
        processed = remove_specular_highlights(processed)

    return processed


def apply_flat_field_correction(image: np.ndarray, flat_field_path: Optional[Path] = None) -> np.ndarray:
    """Apply flat-field / shading correction: I' = (I - dark) / (flat - dark)."""
    if flat_field_path and flat_field_path.exists():
        flat_field = np.load(flat_field_path).astype(np.float32)
        dark_field = np.zeros_like(flat_field)  # Assume dark current is negligible

        corrected = (image - dark_field) / (flat_field - dark_field + 1e-6)
        corrected = np.clip(corrected, 0.0, np.inf)
    else:
        corrected = estimate_and_correct_shading(image)

    return corrected


def estimate_and_correct_shading(image: np.ndarray) -> np.ndarray:
    """Estimate and correct illumination shading using morphological operations."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[2] == 3 else image[:, :, 0]
    else:
        gray = image

    kernel_size = int(min(gray.shape) * 0.1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    background = cv2.GaussianBlur(background, (kernel_size, kernel_size), 0)

    corrected = gray / (background + 1e-6)
    corrected = np.clip(corrected, 0.0, 3.0)

    if image.ndim == 3:
        correction_factor = corrected / (gray + 1e-6)
        corrected_rgb = image * correction_factor[:, :, np.newaxis]
        return np.clip(corrected_rgb, 0.0, 255.0)

    return corrected * 255.0


def apply_stain_normalization(image: np.ndarray, reference_path: Optional[Path] = None) -> np.ndarray:
    """Apply stain normalization using Macenko or Reinhard method."""
    if reference_path and reference_path.exists():
        reference_stats = np.load(reference_path)
        return reinhard_normalization(image, reference_stats)
    else:
        return macenko_normalization(image)


def macenko_normalization(image: np.ndarray) -> np.ndarray:
    """Macenko stain normalization for H&E stained images."""
    if image.ndim != 3 or image.shape[2] != 3:
        return image

    image = image.astype(np.float64)
    image = np.maximum(image, 1)

    log_rgb = -np.log(image / 255.0)

    mask = (image.sum(axis=2) > 50) & (image.sum(axis=2) < 700)
    log_rgb_vec = log_rgb[mask].reshape(-1, 3)

    if log_rgb_vec.shape[0] < 100:
        return image.astype(np.float32)

    cov_matrix = np.cov(log_rgb_vec.T)
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)

    idx = np.argsort(eigenvals)[::-1]
    eigenvecs = eigenvecs[:, idx]

    proj = log_rgb_vec @ eigenvecs[:, :2]

    angles = np.arctan2(proj[:, 1], proj[:, 0])

    percentile_low, percentile_high = 1, 99
    angle_low = np.percentile(angles, percentile_low)
    angle_high = np.percentile(angles, percentile_high)

    stain_matrix = np.array([
        [np.cos(angle_low), np.cos(angle_high)],
        [np.sin(angle_low), np.sin(angle_high)]
    ])

    stain_matrix_3d = eigenvecs[:, :2] @ stain_matrix

    concentrations = np.linalg.lstsq(stain_matrix_3d.T, log_rgb_vec.T, rcond=None)[0]

    max_conc = np.percentile(concentrations, 99, axis=1)

    target_he = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])
    target_max = np.array([1.9705, 1.0308])

    normalized_conc = concentrations * (target_max[:, np.newaxis] / max_conc[:, np.newaxis])

    normalized_log = target_he.T @ normalized_conc
    normalized_rgb = np.exp(-normalized_log.T) * 255

    output = image.copy()
    output[mask] = normalized_rgb

    return np.clip(output, 0, 255).astype(np.float32)


def reinhard_normalization(image: np.ndarray, reference_stats: dict) -> np.ndarray:
    """Reinhard color normalization using reference statistics."""
    if image.ndim != 3:
        return image

    lab_image = color.rgb2lab(image / 255.0)

    for i in range(3):
        channel = lab_image[:, :, i]
        mean_val = np.mean(channel)
        std_val = np.std(channel)

        if std_val > 0:
            normalized = (channel - mean_val) / std_val
            normalized = normalized * reference_stats[f'std_{i}'] + reference_stats[f'mean_{i}']
            lab_image[:, :, i] = normalized

    rgb_normalized = color.lab2rgb(lab_image) * 255
    return np.clip(rgb_normalized, 0, 255).astype(np.float32)


def remove_specular_highlights(image: np.ndarray) -> np.ndarray:
    """Remove specular highlights using HSV saturation threshold and inpainting."""
    if image.ndim != 3:
        return image

    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2HSV)

    brightness = np.max(image_uint8, axis=2)
    saturation = hsv[:, :, 1]

    bright_mask = brightness > 220
    low_sat_mask = saturation < 30
    highlight_mask = bright_mask & low_sat_mask

    highlight_mask = cv2.morphologyEx(
        highlight_mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ).astype(bool)

    if np.any(highlight_mask):
        inpainted = cv2.inpaint(
            image_uint8,
            highlight_mask.astype(np.uint8),
            inpaintRadius=3,
            flags=cv2.INPAINT_TELEA
        )
        result = image.copy()
        result[highlight_mask] = inpainted[highlight_mask].astype(np.float32)
        return result

    return image


def romanowsky_deconvolution(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Deconvolve Romanowsky stains into eosin and methylene blue channels."""
    if image.ndim != 3:
        gray = image if image.ndim == 2 else image[:, :, 0]
        return gray, gray

    image_norm = image.astype(np.float64) / 255.0
    image_norm = np.maximum(image_norm, 1e-6)

    optical_density = -np.log(image_norm)

    eosin_vector = np.array([0.2743, 0.5794, 0.7660])
    methylene_vector = np.array([0.5626, 0.2744, 0.7804])

    stain_matrix = np.column_stack([eosin_vector, methylene_vector])

    od_reshaped = optical_density.reshape(-1, 3)
    concentrations = np.linalg.lstsq(stain_matrix, od_reshaped.T, rcond=None)[0]

    eosin_conc = concentrations[0].reshape(image.shape[:2])
    methylene_conc = concentrations[1].reshape(image.shape[:2])

    eosin_normalized = exposure.rescale_intensity(eosin_conc, out_range=(0, 1))
    methylene_normalized = exposure.rescale_intensity(methylene_conc, out_range=(0, 1))

    return eosin_normalized.astype(np.float32), methylene_normalized.astype(np.float32)


def create_reference_statistics(images: list[np.ndarray]) -> dict:
    """Create reference statistics for Reinhard normalization from a set of images."""
    all_lab = []

    for img in images:
        if img.ndim == 3:
            lab = color.rgb2lab(img / 255.0)
            all_lab.append(lab.reshape(-1, 3))

    if not all_lab:
        return {'mean_0': 0, 'mean_1': 0, 'mean_2': 0, 'std_0': 1, 'std_1': 1, 'std_2': 1}

    combined_lab = np.vstack(all_lab)

    stats = {}
    for i in range(3):
        stats[f'mean_{i}'] = np.mean(combined_lab[:, i])
        stats[f'std_{i}'] = np.std(combined_lab[:, i])

    return stats

