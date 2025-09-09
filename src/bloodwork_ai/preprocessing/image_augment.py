"""Image augmentation for microscopy images."""

from typing import Any, Dict, List, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

from ..utils.log import get_logger


class ImageAugmenter:
    """Image augmentation for microscopy images."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the image augmenter.
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = get_logger(__name__)
        self.config = self._load_config(config_path)
        self.augmentation_pipeline = None
        self._build_augmentation_pipeline()
        
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
    
    def _build_augmentation_pipeline(self) -> None:
        """Build the augmentation pipeline based on configuration."""
        augmentation_config = self.config.get("augmentation", {})
        
        if not augmentation_config.get("enabled", True):
            self.augmentation_pipeline = A.Compose([])
            return
        
        transforms = []
        probability = augmentation_config.get("probability", 0.5)
        
        # Geometric transformations
        geometric_config = augmentation_config.get("geometric", {})
        
        if geometric_config.get("rotation", {}).get("enabled", True):
            rotation_config = geometric_config["rotation"]
            transforms.append(
                A.Rotate(
                    limit=rotation_config.get("limit", 15),
                    p=rotation_config.get("probability", 0.3)
                )
            )
        
        if geometric_config.get("flip", {}).get("horizontal", True):
            flip_config = geometric_config["flip"]
            transforms.append(
                A.HorizontalFlip(p=flip_config.get("probability", 0.5))
            )
        
        if geometric_config.get("flip", {}).get("vertical", False):
            flip_config = geometric_config["flip"]
            transforms.append(
                A.VerticalFlip(p=flip_config.get("probability", 0.5))
            )
        
        if geometric_config.get("shift_scale_rotate", {}).get("enabled", True):
            ssr_config = geometric_config["shift_scale_rotate"]
            transforms.append(
                A.ShiftScaleRotate(
                    shift_limit=ssr_config.get("shift_limit", 0.1),
                    scale_limit=ssr_config.get("scale_limit", 0.1),
                    rotate_limit=ssr_config.get("rotate_limit", 15),
                    p=ssr_config.get("probability", 0.3)
                )
            )
        
        if geometric_config.get("elastic_transform", {}).get("enabled", False):
            elastic_config = geometric_config["elastic_transform"]
            transforms.append(
                A.ElasticTransform(
                    alpha=elastic_config.get("alpha", 1),
                    sigma=elastic_config.get("sigma", 50),
                    alpha_affine=elastic_config.get("alpha_affine", 50),
                    p=elastic_config.get("probability", 0.2)
                )
            )
        
        # Color transformations
        color_config = augmentation_config.get("color", {})
        
        if color_config.get("brightness_contrast", {}).get("enabled", True):
            bc_config = color_config["brightness_contrast"]
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=bc_config.get("brightness_limit", 0.2),
                    contrast_limit=bc_config.get("contrast_limit", 0.2),
                    p=bc_config.get("probability", 0.3)
                )
            )
        
        if color_config.get("hue_saturation_value", {}).get("enabled", True):
            hsv_config = color_config["hue_saturation_value"]
            transforms.append(
                A.HueSaturationValue(
                    hue_shift_limit=hsv_config.get("hue_shift_limit", 20),
                    sat_shift_limit=hsv_config.get("sat_shift_limit", 30),
                    val_shift_limit=hsv_config.get("val_shift_limit", 20),
                    p=hsv_config.get("probability", 0.3)
                )
            )
        
        if color_config.get("rgb_shift", {}).get("enabled", True):
            rgb_config = color_config["rgb_shift"]
            transforms.append(
                A.RGBShift(
                    r_shift_limit=rgb_config.get("r_shift_limit", 20),
                    g_shift_limit=rgb_config.get("g_shift_limit", 20),
                    b_shift_limit=rgb_config.get("b_shift_limit", 20),
                    p=rgb_config.get("probability", 0.3)
                )
            )
        
        if color_config.get("random_gamma", {}).get("enabled", True):
            gamma_config = color_config["random_gamma"]
            transforms.append(
                A.RandomGamma(
                    gamma_limit=gamma_config.get("gamma_limit", [80, 120]),
                    p=gamma_config.get("probability", 0.2)
                )
            )
        
        # Noise and blur
        noise_config = augmentation_config.get("noise", {})
        
        if noise_config.get("gauss_noise", {}).get("enabled", True):
            gauss_config = noise_config["gauss_noise"]
            transforms.append(
                A.GaussNoise(
                    var_limit=gauss_config.get("var_limit", [10, 50]),
                    p=gauss_config.get("probability", 0.2)
                )
            )
        
        if noise_config.get("gaussian_blur", {}).get("enabled", True):
            blur_config = noise_config["gaussian_blur"]
            transforms.append(
                A.GaussianBlur(
                    blur_limit=blur_config.get("blur_limit", [3, 7]),
                    p=blur_config.get("probability", 0.2)
                )
            )
        
        if noise_config.get("motion_blur", {}).get("enabled", False):
            motion_config = noise_config["motion_blur"]
            transforms.append(
                A.MotionBlur(
                    blur_limit=motion_config.get("blur_limit", 7),
                    p=motion_config.get("probability", 0.2)
                )
            )
        
        # Advanced augmentations
        advanced_config = augmentation_config.get("advanced", {})
        
        if advanced_config.get("cutout", {}).get("enabled", False):
            cutout_config = advanced_config["cutout"]
            transforms.append(
                A.CoarseDropout(
                    max_holes=cutout_config.get("num_holes", 8),
                    max_height=cutout_config.get("max_h_size", 32),
                    max_width=cutout_config.get("max_w_size", 32),
                    p=cutout_config.get("probability", 0.2)
                )
            )
        
        if advanced_config.get("grid_distortion", {}).get("enabled", False):
            grid_config = advanced_config["grid_distortion"]
            transforms.append(
                A.GridDistortion(
                    num_steps=grid_config.get("num_steps", 5),
                    distort_limit=grid_config.get("distort_limit", 0.3),
                    p=grid_config.get("probability", 0.2)
                )
            )
        
        if advanced_config.get("optical_distortion", {}).get("enabled", False):
            optical_config = advanced_config["optical_distortion"]
            transforms.append(
                A.OpticalDistortion(
                    distort_limit=optical_config.get("distort_limit", 0.05),
                    shift_limit=optical_config.get("shift_limit", 0.05),
                    p=optical_config.get("probability", 0.2)
                )
            )
        
        # Build the pipeline
        self.augmentation_pipeline = A.Compose(transforms)
        
        self.logger.info(f"Built augmentation pipeline with {len(transforms)} transforms")
    
    def augment_image(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        bboxes: Optional[List[List[float]]] = None,
        keypoints: Optional[List[List[float]]] = None
    ) -> Dict[str, Any]:
        """
        Augment a single image.
        
        Args:
            image: Input image (H, W, C)
            mask: Optional mask (H, W)
            bboxes: Optional bounding boxes [[x, y, w, h], ...]
            keypoints: Optional keypoints [[x, y], ...]
            
        Returns:
            Dictionary with augmented data
        """
        if self.augmentation_pipeline is None:
            self._build_augmentation_pipeline()
        
        # Prepare data for augmentation
        data = {"image": image}
        
        if mask is not None:
            data["mask"] = mask
        
        if bboxes is not None:
            data["bboxes"] = bboxes
        
        if keypoints is not None:
            data["keypoints"] = keypoints
        
        try:
            # Apply augmentation
            augmented = self.augmentation_pipeline(**data)
            
            return augmented
            
        except Exception as e:
            self.logger.error(f"Augmentation failed: {e}")
            # Return original data if augmentation fails
            return data
    
    def augment_batch(
        self,
        images: List[np.ndarray],
        masks: Optional[List[np.ndarray]] = None,
        bboxes_list: Optional[List[List[List[float]]]] = None,
        keypoints_list: Optional[List[List[List[float]]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Augment a batch of images.
        
        Args:
            images: List of input images
            masks: Optional list of masks
            bboxes_list: Optional list of bounding box lists
            keypoints_list: Optional list of keypoint lists
            
        Returns:
            List of augmented data dictionaries
        """
        augmented_batch = []
        
        for i, image in enumerate(images):
            mask = masks[i] if masks is not None else None
            bboxes = bboxes_list[i] if bboxes_list is not None else None
            keypoints = keypoints_list[i] if keypoints_list is not None else None
            
            try:
                augmented = self.augment_image(image, mask, bboxes, keypoints)
                augmented_batch.append(augmented)
            except Exception as e:
                self.logger.error(f"Failed to augment image {i}: {e}")
                # Return original data if augmentation fails
                augmented_batch.append({
                    "image": image,
                    "mask": mask,
                    "bboxes": bboxes,
                    "keypoints": keypoints
                })
        
        return augmented_batch
    
    def get_training_augmentation_pipeline(self) -> A.Compose:
        """
        Get augmentation pipeline for training.
        
        Returns:
            Albumentations pipeline for training
        """
        if self.augmentation_pipeline is None:
            self._build_augmentation_pipeline()
        
        return self.augmentation_pipeline
    
    def get_validation_augmentation_pipeline(self) -> A.Compose:
        """
        Get augmentation pipeline for validation (minimal augmentation).
        
        Returns:
            Albumentations pipeline for validation
        """
        # Minimal augmentation for validation
        transforms = [
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
        
        return A.Compose(transforms)
    
    def get_inference_augmentation_pipeline(self) -> A.Compose:
        """
        Get augmentation pipeline for inference (no augmentation).
        
        Returns:
            Albumentations pipeline for inference
        """
        # No augmentation for inference
        transforms = [
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
        
        return A.Compose(transforms)
    
    def visualize_augmentation(
        self,
        image: np.ndarray,
        num_samples: int = 4,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize augmentation results.
        
        Args:
            image: Input image
            num_samples: Number of augmented samples to generate
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        augmented_images = []
        
        for _ in range(num_samples):
            augmented = self.augment_image(image)
            augmented_images.append(augmented["image"])
        
        # Create visualization grid
        if len(augmented_images) > 0:
            # Resize images to same size
            target_size = (224, 224)
            resized_images = []
            
            for img in [image] + augmented_images:
                resized = cv2.resize(img, target_size)
                resized_images.append(resized)
            
            # Create grid
            rows = 2
            cols = (len(resized_images) + 1) // 2
            
            grid_height = rows * target_size[0]
            grid_width = cols * target_size[1]
            
            grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
            
            for i, img in enumerate(resized_images):
                row = i // cols
                col = i % cols
                
                y_start = row * target_size[0]
                y_end = y_start + target_size[0]
                x_start = col * target_size[1]
                x_end = x_start + target_size[1]
                
                grid[y_start:y_end, x_start:x_end] = img
            
            if save_path:
                cv2.imwrite(save_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
                self.logger.info(f"Saved augmentation visualization to {save_path}")
            
            return grid
        
        return image
    
    def get_augmentation_report(self) -> Dict[str, Any]:
        """
        Get augmentation configuration report.
        
        Returns:
            Augmentation report dictionary
        """
        augmentation_config = self.config.get("augmentation", {})
        
        report = {
            "enabled": augmentation_config.get("enabled", True),
            "probability": augmentation_config.get("probability", 0.5),
            "transforms": []
        }
        
        # Count enabled transforms
        geometric_config = augmentation_config.get("geometric", {})
        color_config = augmentation_config.get("color", {})
        noise_config = augmentation_config.get("noise", {})
        advanced_config = augmentation_config.get("advanced", {})
        
        # Geometric transforms
        if geometric_config.get("rotation", {}).get("enabled", True):
            report["transforms"].append("rotation")
        if geometric_config.get("flip", {}).get("horizontal", True):
            report["transforms"].append("horizontal_flip")
        if geometric_config.get("flip", {}).get("vertical", False):
            report["transforms"].append("vertical_flip")
        if geometric_config.get("shift_scale_rotate", {}).get("enabled", True):
            report["transforms"].append("shift_scale_rotate")
        if geometric_config.get("elastic_transform", {}).get("enabled", False):
            report["transforms"].append("elastic_transform")
        
        # Color transforms
        if color_config.get("brightness_contrast", {}).get("enabled", True):
            report["transforms"].append("brightness_contrast")
        if color_config.get("hue_saturation_value", {}).get("enabled", True):
            report["transforms"].append("hue_saturation_value")
        if color_config.get("rgb_shift", {}).get("enabled", True):
            report["transforms"].append("rgb_shift")
        if color_config.get("random_gamma", {}).get("enabled", True):
            report["transforms"].append("random_gamma")
        
        # Noise transforms
        if noise_config.get("gauss_noise", {}).get("enabled", True):
            report["transforms"].append("gauss_noise")
        if noise_config.get("gaussian_blur", {}).get("enabled", True):
            report["transforms"].append("gaussian_blur")
        if noise_config.get("motion_blur", {}).get("enabled", False):
            report["transforms"].append("motion_blur")
        
        # Advanced transforms
        if advanced_config.get("cutout", {}).get("enabled", False):
            report["transforms"].append("cutout")
        if advanced_config.get("grid_distortion", {}).get("enabled", False):
            report["transforms"].append("grid_distortion")
        if advanced_config.get("optical_distortion", {}).get("enabled", False):
            report["transforms"].append("optical_distortion")
        
        report["total_transforms"] = len(report["transforms"])
        
        return report
