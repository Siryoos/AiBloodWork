from __future__ import annotations

from typing import Final, Optional, Tuple, Union
import time

import cv2
import numpy as np

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback decorator that does nothing
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range

_EPS: Final[float] = 1e-12


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale with optimized path."""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        # Assume BGR (OpenCV convention). If using RGB, adjust as needed.
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("Unsupported image shape for grayscale conversion: " + str(image.shape))


def _normalize_roi(image: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Production-grade ROI normalization for consistent metrics."""
    gray = _to_grayscale(image).astype(np.float64)

    if roi_mask is not None:
        # Only use valid ROI pixels for normalization
        valid_pixels = gray[roi_mask]
        if len(valid_pixels) == 0:
            return gray
        mean_val = float(np.mean(valid_pixels))
        std_val = float(np.std(valid_pixels))
    else:
        mean_val = float(gray.mean())
        std_val = float(gray.std())

    if std_val < _EPS:
        return gray - mean_val

    # Z-score normalization
    normalized = (gray - mean_val) / std_val
    return normalized


def _create_roi_mask(image_shape: Tuple[int, int],
                    roi_fracs: Optional[Tuple[float, float, float, float]] = None) -> np.ndarray:
    """Create ROI mask from fractional coordinates (x, y, w, h)."""
    h, w = image_shape
    if roi_fracs is None:
        # Default center ROI
        roi_fracs = (0.25, 0.25, 0.5, 0.5)

    fx, fy, fw, fh = roi_fracs
    x = int(fx * w)
    y = int(fy * h)
    rw = int(fw * w)
    rh = int(fh * h)

    # Ensure bounds
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    x2 = min(w, x + rw)
    y2 = min(h, y + rh)

    mask = np.zeros((h, w), dtype=bool)
    mask[y:y2, x:x2] = True
    return mask


@njit
def _sobel_magnitude_numba(gray: np.ndarray) -> float:
    """Optimized Sobel gradient computation using Numba."""
    h, w = gray.shape
    result = 0.0

    for y in prange(1, h - 1):
        for x in range(1, w - 1):
            # Sobel X kernel: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
            gx = (-1 * gray[y-1, x-1] + 1 * gray[y-1, x+1] +
                  -2 * gray[y, x-1]   + 2 * gray[y, x+1] +
                  -1 * gray[y+1, x-1] + 1 * gray[y+1, x+1])

            # Sobel Y kernel: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
            gy = (-1 * gray[y-1, x-1] + -2 * gray[y-1, x] + -1 * gray[y-1, x+1] +
                   1 * gray[y+1, x-1] +  2 * gray[y+1, x] +  1 * gray[y+1, x+1])

            result += gx * gx + gy * gy

    return result


@njit
def _brenner_gradient_numba(gray: np.ndarray) -> float:
    """Optimized Brenner gradient using Numba."""
    h, w = gray.shape
    result = 0.0

    for y in prange(h):
        for x in range(w - 2):
            diff = gray[y, x] - gray[y, x + 2]
            result += diff * diff

    return result


@njit
def _block_dct_energy_numba(gray: np.ndarray, block_size: int = 8) -> float:
    """Block DCT energy computation using Numba (simplified)."""
    h, w = gray.shape
    energy = 0.0

    # Simple block variance as proxy for DCT energy (fast approximation)
    for y in prange(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = gray[y:y+block_size, x:x+block_size]
            mean_val = 0.0
            for by in range(block_size):
                for bx in range(block_size):
                    mean_val += block[by, bx]
            mean_val /= (block_size * block_size)

            var_val = 0.0
            for by in range(block_size):
                for bx in range(block_size):
                    diff = block[by, bx] - mean_val
                    var_val += diff * diff

            energy += var_val

    return energy


def variance_of_laplacian(image: np.ndarray) -> float:
    """Compute Variance of Laplacian as a focus measure.

    Higher values indicate sharper focus.
    """
    gray = _to_grayscale(image)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = float(lap.var())
    # Guard against NaNs or tiny values
    if not np.isfinite(var):
        return 0.0
    return max(0.0, var)


def tenengrad(image: np.ndarray,
              ksize: int = 3,
              roi_fracs: Optional[Tuple[float, float, float, float]] = None,
              normalize: bool = True,
              use_numba: bool = True) -> float:
    """Production-grade Tenengrad (Sobel gradient magnitude) focus measure.

    Args:
        image: Input image (BGR or grayscale).
        ksize: Sobel kernel size (1, 3, 5, 7). Use 3 for speed.
        roi_fracs: ROI as (x, y, w, h) fractions. None for full image.
        normalize: Whether to apply ROI normalization.
        use_numba: Use optimized Numba implementation for ksize=3.
    """
    gray = _to_grayscale(image).astype(np.float64)

    # Apply ROI mask if specified
    roi_mask = None
    if roi_fracs is not None:
        roi_mask = _create_roi_mask(gray.shape, roi_fracs)
        if normalize:
            gray = _normalize_roi(image, roi_mask)

    # Use optimized path for default case
    if use_numba and HAS_NUMBA and ksize == 3 and roi_mask is None:
        score = _sobel_magnitude_numba(gray)
        # Normalize by pixel count
        score = score / (gray.shape[0] * gray.shape[1])
    else:
        # Fallback to OpenCV
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        mag2 = gx * gx + gy * gy

        if roi_mask is not None:
            mag2 = mag2[roi_mask]

        score = float(np.mean(mag2))

    if not np.isfinite(score):
        return 0.0
    return max(0.0, score)


def brenner_gradient(image: np.ndarray,
                    roi_fracs: Optional[Tuple[float, float, float, float]] = None,
                    normalize: bool = True,
                    use_numba: bool = True) -> float:
    """Production-grade Brenner focus metric using forward finite differences.

    Computes sum((I(x,y) - I(x+2,y))^2). Cheap and effective for PBS.

    Args:
        image: Input image (BGR or grayscale).
        roi_fracs: ROI as (x, y, w, h) fractions. None for full image.
        normalize: Whether to apply ROI normalization.
        use_numba: Use optimized Numba implementation.
    """
    gray = _to_grayscale(image).astype(np.float64)

    # Apply ROI mask if specified
    roi_mask = None
    if roi_fracs is not None:
        roi_mask = _create_roi_mask(gray.shape, roi_fracs)
        if normalize:
            gray = _normalize_roi(image, roi_mask)

    # Use optimized path
    if use_numba and HAS_NUMBA and roi_mask is None:
        score = _brenner_gradient_numba(gray)
        # Normalize by valid pixel count (width - 2)
        score = score / (gray.shape[0] * (gray.shape[1] - 2))
    else:
        # Fallback implementation
        diff = gray[:, 2:] - gray[:, :-2]
        if roi_mask is not None:
            # Apply mask to diff array (adjusted for smaller width)
            roi_diff = roi_mask[:, :-2]  # Adjust mask for diff array
            diff = diff[roi_diff]
        score = float(np.mean(diff * diff))

    if not np.isfinite(score):
        return 0.0
    return max(0.0, score)


def block_dct_energy(image: np.ndarray,
                    block_size: int = 8,
                    roi_fracs: Optional[Tuple[float, float, float, float]] = None,
                    normalize: bool = True,
                    use_numba: bool = True) -> float:
    """Block DCT energy metric for focus assessment.

    Faster than FFT for small blocks. Good for PBS edge detection.

    Args:
        image: Input image (BGR or grayscale).
        block_size: DCT block size (typically 8).
        roi_fracs: ROI as (x, y, w, h) fractions.
        normalize: Whether to apply ROI normalization.
        use_numba: Use optimized Numba approximation.
    """
    gray = _to_grayscale(image).astype(np.float64)

    # Apply ROI mask if specified
    if roi_fracs is not None:
        roi_mask = _create_roi_mask(gray.shape, roi_fracs)
        if normalize:
            gray = _normalize_roi(image, roi_mask)
        # Extract ROI region
        y_indices, x_indices = np.where(roi_mask)
        if len(y_indices) == 0:
            return 0.0
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        gray = gray[y_min:y_max+1, x_min:x_max+1]

    # Use optimized approximation
    if use_numba and HAS_NUMBA:
        score = _block_dct_energy_numba(gray, block_size)
        # Normalize by number of blocks
        num_blocks = ((gray.shape[0] // block_size) *
                     (gray.shape[1] // block_size))
        if num_blocks > 0:
            score = score / num_blocks
    else:
        # Fallback: use block variance as proxy
        h, w = gray.shape
        energy = 0.0
        num_blocks = 0

        for y in range(0, h - block_size + 1, block_size):
            for x in range(0, w - block_size + 1, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                energy += float(np.var(block))
                num_blocks += 1

        score = energy / num_blocks if num_blocks > 0 else 0.0

    if not np.isfinite(score):
        return 0.0
    return max(0.0, score)


def high_frequency_energy(image: np.ndarray, cutoff_ratio: float = 0.25) -> float:
    """High-frequency energy via 2D FFT magnitude above a cutoff.

    Args:
        image: Input image (BGR or grayscale).
        cutoff_ratio: Fraction of Nyquist below which to zero-out (0..0.49).
    """
    gray = _to_grayscale(image).astype(np.float64)
    # Remove DC bias
    gray = gray - float(gray.mean())
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    ry = int(cutoff_ratio * h / 2)
    rx = int(cutoff_ratio * w / 2)
    # Zero a centered low-pass square region
    y0, y1 = max(0, cy - ry), min(h, cy + ry + 1)
    x0, x1 = max(0, cx - rx), min(w, cx + rx + 1)
    fshift[y0:y1, x0:x1] = 0
    mag = np.abs(fshift)
    score = float(np.mean(mag))
    if not np.isfinite(score):
        return 0.0
    return max(0.0, score)


def normalized_dct_energy(image: np.ndarray, cutoff_ratio: float = 0.25) -> float:
    """Normalized DCT energy metric for focus assessment.

    Computes high-frequency energy using DCT, often faster than FFT for small blocks.

    Args:
        image: Input image (BGR or grayscale).
        cutoff_ratio: Fraction below which to zero-out DCT coefficients.
    """
    gray = _to_grayscale(image).astype(np.float64)
    # Normalize intensity
    gray = gray - float(gray.mean())

    # Use DCT2
    from scipy.fftpack import dct
    dct_coeffs = dct(dct(gray, norm='ortho', axis=0), norm='ortho', axis=1)

    h, w = gray.shape
    # Zero out low-frequency components
    cutoff_h = int(cutoff_ratio * h)
    cutoff_w = int(cutoff_ratio * w)

    # Create mask to keep only high-frequency components
    mask = np.ones_like(dct_coeffs)
    mask[:cutoff_h, :cutoff_w] = 0

    hf_energy = float(np.mean(np.abs(dct_coeffs * mask)))

    if not np.isfinite(hf_energy):
        return 0.0
    return max(0.0, hf_energy)


def metric_fusion(image: np.ndarray, weights: dict | None = None) -> float:
    """Fused focus metric combining multiple measures for robustness.

    Args:
        image: Input image (BGR or grayscale).
        weights: Dict with keys 'tenengrad', 'hf_energy', 'brenner', 'laplacian'.
                 If None, uses balanced weights optimized for blood smears.
    """
    if weights is None:
        weights = {
            'tenengrad': 0.4,
            'hf_energy': 0.3,
            'brenner': 0.2,
            'laplacian': 0.1
        }

    # Normalize image for consistent metrics
    gray = _to_grayscale(image)
    mean_intensity = float(gray.mean())
    if mean_intensity < _EPS:
        return 0.0

    # Compute individual metrics
    metrics = {}
    if weights.get('tenengrad', 0) > 0:
        metrics['tenengrad'] = tenengrad(image) / (mean_intensity + _EPS)
    if weights.get('hf_energy', 0) > 0:
        metrics['hf_energy'] = high_frequency_energy(image) / (mean_intensity + _EPS)
    if weights.get('brenner', 0) > 0:
        metrics['brenner'] = brenner_gradient(image) / (mean_intensity + _EPS)
    if weights.get('laplacian', 0) > 0:
        metrics['laplacian'] = variance_of_laplacian(image) / (mean_intensity + _EPS)

    # Weighted combination
    fused_score = sum(weights.get(k, 0) * v for k, v in metrics.items())

    if not np.isfinite(fused_score):
        return 0.0
    return max(0.0, float(fused_score))
