"""Stain normalization for microscopy images."""

from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import albumentations as A

from ..utils.log import get_logger


class StainNormalizer:
    """Stain normalization for microscopy images."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the stain normalizer.
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = get_logger(__name__)
        self.config = self._load_config(config_path)
        self.target_stain = None
        self.stain_matrix = None
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file."""
        if config_path is None:
            config_path = "configs/data/images.yaml"
        
        try:
            from ..utils.io import load_data
            return load_data(config_path, file_type="yaml")
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            return {}
    
    def normalize_stain(
        self,
        image: np.ndarray,
        method: str = "macenko",
        target_stain: str = "he"
    ) -> np.ndarray:
        """
        Normalize stain in microscopy image.
        
        Args:
            image: Input image (H, W, C)
            method: Normalization method (macenko, reinhard, vahadane)
            target_stain: Target stain type (he, ihc, pas)
            
        Returns:
            Normalized image
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Image must be 3-channel RGB image")
        
        self.logger.debug(f"Normalizing stain using {method} method")
        
        if method == "macenko":
            return self._macenko_normalization(image, target_stain)
        elif method == "reinhard":
            return self._reinhard_normalization(image, target_stain)
        elif method == "vahadane":
            return self._vahadane_normalization(image, target_stain)
        else:
            self.logger.warning(f"Unknown normalization method: {method}")
            return image
    
    def _macenko_normalization(
        self,
        image: np.ndarray,
        target_stain: str = "he"
    ) -> np.ndarray:
        """
        Macenko stain normalization.
        
        Args:
            image: Input RGB image
            target_stain: Target stain type
            
        Returns:
            Normalized image
        """
        try:
            # Convert RGB to OD (Optical Density)
            od = self._rgb_to_od(image)
            
            # Remove white pixels
            od = self._remove_white_pixels(od)
            
            # Get stain matrix
            stain_matrix = self._get_stain_matrix(od)
            
            # Normalize to target stain
            normalized_od = self._normalize_to_target_stain(od, stain_matrix, target_stain)
            
            # Convert back to RGB
            normalized_rgb = self._od_to_rgb(normalized_od)
            
            return normalized_rgb
            
        except Exception as e:
            self.logger.error(f"Macenko normalization failed: {e}")
            return image
    
    def _reinhard_normalization(
        self,
        image: np.ndarray,
        target_stain: str = "he"
    ) -> np.ndarray:
        """
        Reinhard stain normalization.
        
        Args:
            image: Input RGB image
            target_stain: Target stain type
            
        Returns:
            Normalized image
        """
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Get target statistics
            target_mean, target_std = self._get_target_statistics(target_stain)
            
            # Normalize each channel
            for i in range(3):
                lab[:, :, i] = (lab[:, :, i] - lab[:, :, i].mean()) / lab[:, :, i].std()
                lab[:, :, i] = lab[:, :, i] * target_std[i] + target_mean[i]
            
            # Convert back to RGB
            normalized_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return normalized_rgb
            
        except Exception as e:
            self.logger.error(f"Reinhard normalization failed: {e}")
            return image
    
    def _vahadane_normalization(
        self,
        image: np.ndarray,
        target_stain: str = "he"
    ) -> np.ndarray:
        """
        Vahadane stain normalization.
        
        Args:
            image: Input RGB image
            target_stain: Target stain type
            
        Returns:
            Normalized image
        """
        try:
            # Convert RGB to OD
            od = self._rgb_to_od(image)
            
            # Remove white pixels
            od = self._remove_white_pixels(od)
            
            # Get stain matrix using SVD
            stain_matrix = self._get_stain_matrix_svd(od)
            
            # Normalize to target stain
            normalized_od = self._normalize_to_target_stain(od, stain_matrix, target_stain)
            
            # Convert back to RGB
            normalized_rgb = self._od_to_rgb(normalized_od)
            
            return normalized_rgb
            
        except Exception as e:
            self.logger.error(f"Vahadane normalization failed: {e}")
            return image
    
    def _rgb_to_od(self, image: np.ndarray) -> np.ndarray:
        """Convert RGB to Optical Density."""
        # Add small epsilon to avoid log(0)
        od = -np.log(image.astype(np.float32) / 255.0 + 1e-8)
        return od
    
    def _od_to_rgb(self, od: np.ndarray) -> np.ndarray:
        """Convert Optical Density to RGB."""
        rgb = np.exp(-od) * 255.0
        return np.clip(rgb, 0, 255).astype(np.uint8)
    
    def _remove_white_pixels(self, od: np.ndarray, threshold: float = 0.8) -> np.ndarray:
        """Remove white pixels from OD image."""
        # White pixels have low OD values
        white_mask = np.all(od < threshold, axis=2)
        return od[~white_mask]
    
    def _get_stain_matrix(self, od: np.ndarray) -> np.ndarray:
        """Get stain matrix using Macenko method."""
        # Reshape to 2D
        od_flat = od.reshape(-1, 3)
        
        # Remove zero rows
        od_flat = od_flat[np.any(od_flat > 0, axis=1)]
        
        # SVD decomposition
        U, S, V = np.linalg.svd(od_flat, full_matrices=False)
        
        # Get first two principal components
        stain_matrix = V[:2, :].T
        
        return stain_matrix
    
    def _get_stain_matrix_svd(self, od: np.ndarray) -> np.ndarray:
        """Get stain matrix using SVD method."""
        # Reshape to 2D
        od_flat = od.reshape(-1, 3)
        
        # Remove zero rows
        od_flat = od_flat[np.any(od_flat > 0, axis=1)]
        
        # SVD decomposition
        U, S, V = np.linalg.svd(od_flat, full_matrices=False)
        
        # Get first two principal components
        stain_matrix = V[:2, :].T
        
        return stain_matrix
    
    def _normalize_to_target_stain(
        self,
        od: np.ndarray,
        stain_matrix: np.ndarray,
        target_stain: str
    ) -> np.ndarray:
        """Normalize to target stain."""
        # Get target stain matrix
        target_matrix = self._get_target_stain_matrix(target_stain)
        
        # Project onto stain space
        od_flat = od.reshape(-1, 3)
        stain_concentrations = np.linalg.lstsq(stain_matrix, od_flat.T, rcond=None)[0]
        
        # Normalize to target stain
        normalized_od = target_matrix @ stain_concentrations
        normalized_od = normalized_od.T.reshape(od.shape)
        
        return normalized_od
    
    def _get_target_stain_matrix(self, target_stain: str) -> np.ndarray:
        """Get target stain matrix for different stain types."""
        if target_stain == "he":
            # H&E stain matrix
            return np.array([
                [0.65, 0.70, 0.29],
                [0.07, 0.99, 0.11]
            ]).T
        elif target_stain == "ihc":
            # IHC stain matrix
            return np.array([
                [0.65, 0.70, 0.29],
                [0.07, 0.99, 0.11]
            ]).T
        elif target_stain == "pas":
            # PAS stain matrix
            return np.array([
                [0.65, 0.70, 0.29],
                [0.07, 0.99, 0.11]
            ]).T
        else:
            # Default H&E matrix
            return np.array([
                [0.65, 0.70, 0.29],
                [0.07, 0.99, 0.11]
            ]).T
    
    def _get_target_statistics(self, target_stain: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get target statistics for Reinhard normalization."""
        if target_stain == "he":
            # H&E target statistics
            mean = np.array([8.7414, -0.1249, 0.3394])
            std = np.array([0.6136, 0.1096, 0.2901])
        elif target_stain == "ihc":
            # IHC target statistics
            mean = np.array([8.7414, -0.1249, 0.3394])
            std = np.array([0.6136, 0.1096, 0.2901])
        elif target_stain == "pas":
            # PAS target statistics
            mean = np.array([8.7414, -0.1249, 0.3394])
            std = np.array([0.6136, 0.1096, 0.2901])
        else:
            # Default H&E statistics
            mean = np.array([8.7414, -0.1249, 0.3394])
            std = np.array([0.6136, 0.1096, 0.2901])
        
        return mean, std
    
    def normalize_batch(
        self,
        images: List[np.ndarray],
        method: str = "macenko",
        target_stain: str = "he"
    ) -> List[np.ndarray]:
        """
        Normalize a batch of images.
        
        Args:
            images: List of input images
            method: Normalization method
            target_stain: Target stain type
            
        Returns:
            List of normalized images
        """
        normalized_images = []
        
        for i, image in enumerate(images):
            try:
                normalized_image = self.normalize_stain(image, method, target_stain)
                normalized_images.append(normalized_image)
            except Exception as e:
                self.logger.error(f"Failed to normalize image {i}: {e}")
                normalized_images.append(image)  # Return original if normalization fails
        
        return normalized_images
    
    def get_normalization_report(self, original_image: np.ndarray, normalized_image: np.ndarray) -> Dict[str, Any]:
        """
        Generate a normalization report.
        
        Args:
            original_image: Original image
            normalized_image: Normalized image
            
        Returns:
            Normalization report dictionary
        """
        report = {
            "original_shape": original_image.shape,
            "normalized_shape": normalized_image.shape,
            "original_mean": original_image.mean(axis=(0, 1)).tolist(),
            "normalized_mean": normalized_image.mean(axis=(0, 1)).tolist(),
            "original_std": original_image.std(axis=(0, 1)).tolist(),
            "normalized_std": normalized_image.std(axis=(0, 1)).tolist(),
            "original_min": original_image.min(),
            "normalized_min": normalized_image.min(),
            "original_max": original_image.max(),
            "normalized_max": normalized_image.max(),
        }
        
        # Calculate color distribution changes
        original_hist = cv2.calcHist([original_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        normalized_hist = cv2.calcHist([normalized_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        # Calculate histogram correlation
        correlation = cv2.compareHist(original_hist, normalized_hist, cv2.HISTCMP_CORREL)
        report["histogram_correlation"] = float(correlation)
        
        return report
