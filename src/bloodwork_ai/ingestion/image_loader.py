"""Image data loader for blood smear microscopy images."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from ..utils.io import load_data, save_data
from ..utils.log import get_logger


class ImageLoader:
    """Loader for microscopy images with COCO/YOLO format support."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the image loader.
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = get_logger(__name__)
        self.config = self._load_config(config_path)
        self.supported_formats = self.config.get("formats", {}).get("supported_extensions", [".jpg", ".jpeg", ".png"])
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file."""
        if config_path is None:
            config_path = "configs/data/images.yaml"
        
        try:
            return load_data(config_path, file_type="yaml")
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            return {}
    
    def load_image(
        self,
        image_path: Union[str, Path],
        target_size: Optional[Tuple[int, int]] = None,
        color_space: str = "RGB"
    ) -> np.ndarray:
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to the image file
            target_size: Target size (width, height) for resizing
            color_space: Color space conversion (RGB, BGR, GRAY)
            
        Returns:
            Loaded image as numpy array
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if image_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported image format: {image_path.suffix}")
        
        try:
            # Load image using PIL
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != "RGB" and color_space == "RGB":
                    img = img.convert("RGB")
                elif img.mode != "L" and color_space == "GRAY":
                    img = img.convert("L")
                
                # Convert to numpy array
                image = np.array(img)
                
                # Resize if target size is specified
                if target_size is not None:
                    image = self._resize_image(image, target_size)
                
                self.logger.debug(f"Loaded image: {image_path}, shape: {image.shape}")
                return image
                
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {e}")
            raise
    
    def _resize_image(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int],
        maintain_aspect_ratio: bool = True,
        padding: bool = True
    ) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            target_size: Target size (width, height)
            maintain_aspect_ratio: Whether to maintain aspect ratio
            padding: Whether to pad with zeros if aspect ratio is maintained
            
        Returns:
            Resized image
        """
        target_width, target_height = target_size
        current_height, current_width = image.shape[:2]
        
        if maintain_aspect_ratio:
            # Calculate scaling factor
            scale = min(target_width / current_width, target_height / current_height)
            new_width = int(current_width * scale)
            new_height = int(current_height * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            if padding:
                # Create padded image
                padded = np.zeros((target_height, target_width, image.shape[2] if len(image.shape) == 3 else 1), dtype=image.dtype)
                
                # Calculate padding offsets
                y_offset = (target_height - new_height) // 2
                x_offset = (target_width - new_width) // 2
                
                # Place resized image in center
                if len(image.shape) == 3:
                    padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
                else:
                    padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width, 0] = resized
                
                return padded
            else:
                return resized
        else:
            # Direct resize without maintaining aspect ratio
            return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    
    def load_coco_annotations(
        self,
        annotation_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Load COCO format annotations.
        
        Args:
            annotation_path: Path to COCO annotation file
            
        Returns:
            COCO annotation dictionary
        """
        try:
            annotations = load_data(annotation_path, file_type="json")
            self.logger.info(f"Loaded COCO annotations from {annotation_path}")
            return annotations
        except Exception as e:
            self.logger.error(f"Failed to load COCO annotations from {annotation_path}: {e}")
            raise
    
    def load_yolo_annotations(
        self,
        annotation_path: Union[str, Path],
        image_size: Tuple[int, int] = (640, 640)
    ) -> List[Dict[str, Any]]:
        """
        Load YOLO format annotations.
        
        Args:
            annotation_path: Path to YOLO annotation file
            image_size: Image size (width, height) for denormalization
            
        Returns:
            List of annotation dictionaries
        """
        try:
            annotations = []
            image_width, image_height = image_size
            
            with open(annotation_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    center_x = float(parts[1])
                    center_y = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert normalized coordinates to absolute coordinates
                    abs_center_x = center_x * image_width
                    abs_center_y = center_y * image_height
                    abs_width = width * image_width
                    abs_height = height * image_height
                    
                    # Convert to COCO format (x, y, width, height)
                    x = abs_center_x - abs_width / 2
                    y = abs_center_y - abs_height / 2
                    
                    annotation = {
                        "class_id": class_id,
                        "bbox": [x, y, abs_width, abs_height],
                        "area": abs_width * abs_height,
                        "iscrowd": 0
                    }
                    annotations.append(annotation)
            
            self.logger.info(f"Loaded {len(annotations)} YOLO annotations from {annotation_path}")
            return annotations
            
        except Exception as e:
            self.logger.error(f"Failed to load YOLO annotations from {annotation_path}: {e}")
            raise
    
    def convert_yolo_to_coco(
        self,
        yolo_annotations: List[Dict[str, Any]],
        image_id: int,
        category_id: int
    ) -> List[Dict[str, Any]]:
        """
        Convert YOLO annotations to COCO format.
        
        Args:
            yolo_annotations: List of YOLO annotation dictionaries
            image_id: Image ID for COCO format
            category_id: Category ID for COCO format
            
        Returns:
            List of COCO annotation dictionaries
        """
        coco_annotations = []
        
        for idx, yolo_ann in enumerate(yolo_annotations):
            coco_ann = {
                "id": idx + 1,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": yolo_ann["bbox"],
                "area": yolo_ann["area"],
                "iscrowd": yolo_ann["iscrowd"]
            }
            coco_annotations.append(coco_ann)
        
        return coco_annotations
    
    def load_dataset(
        self,
        data_dir: Union[str, Path],
        annotation_format: str = "coco",
        image_size: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """
        Load a complete dataset with images and annotations.
        
        Args:
            data_dir: Directory containing images and annotations
            annotation_format: Format of annotations (coco, yolo)
            image_size: Target image size for resizing
            
        Returns:
            Dataset dictionary with images and annotations
        """
        data_dir = Path(data_dir)
        
        # Find image files
        image_files = []
        for ext in self.supported_formats:
            image_files.extend(data_dir.glob(f"*{ext}"))
        
        if not image_files:
            raise ValueError(f"No image files found in {data_dir}")
        
        # Load annotations
        if annotation_format == "coco":
            annotation_file = data_dir / "annotations.json"
            if annotation_file.exists():
                annotations = self.load_coco_annotations(annotation_file)
            else:
                annotations = {"images": [], "annotations": [], "categories": []}
        elif annotation_format == "yolo":
            # Load YOLO annotations for each image
            annotations = {"images": [], "annotations": [], "categories": []}
        else:
            raise ValueError(f"Unsupported annotation format: {annotation_format}")
        
        # Load images
        images = []
        for idx, image_path in enumerate(image_files):
            try:
                image = self.load_image(image_path, target_size=image_size)
                
                image_info = {
                    "id": idx + 1,
                    "file_name": image_path.name,
                    "width": image.shape[1],
                    "height": image.shape[0],
                    "path": str(image_path)
                }
                images.append(image_info)
                
                # Load YOLO annotations if needed
                if annotation_format == "yolo":
                    yolo_ann_path = image_path.with_suffix(".txt")
                    if yolo_ann_path.exists():
                        yolo_annotations = self.load_yolo_annotations(yolo_ann_path, (image.shape[1], image.shape[0]))
                        coco_annotations = self.convert_yolo_to_coco(yolo_annotations, idx + 1, 1)
                        annotations["annotations"].extend(coco_annotations)
                
            except Exception as e:
                self.logger.warning(f"Failed to load image {image_path}: {e}")
                continue
        
        annotations["images"] = images
        
        self.logger.info(f"Loaded dataset with {len(images)} images and {len(annotations['annotations'])} annotations")
        return {
            "images": images,
            "annotations": annotations,
            "format": annotation_format
        }
    
    def save_dataset(
        self,
        dataset: Dict[str, Any],
        output_dir: Union[str, Path],
        format: str = "coco"
    ) -> None:
        """
        Save dataset to disk.
        
        Args:
            dataset: Dataset dictionary
            output_dir: Output directory
            format: Output format (coco, yolo)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if format == "coco":
            # Save COCO format
            coco_file = output_dir / "annotations.json"
            save_data(dataset["annotations"], coco_file, file_type="json")
            self.logger.info(f"Saved COCO annotations to {coco_file}")
            
        elif format == "yolo":
            # Save YOLO format
            for image_info in dataset["images"]:
                image_id = image_info["id"]
                image_annotations = [
                    ann for ann in dataset["annotations"]["annotations"] 
                    if ann["image_id"] == image_id
                ]
                
                # Convert to YOLO format
                yolo_annotations = []
                for ann in image_annotations:
                    bbox = ann["bbox"]
                    x, y, w, h = bbox
                    
                    # Convert to normalized coordinates
                    center_x = (x + w / 2) / image_info["width"]
                    center_y = (y + h / 2) / image_info["height"]
                    norm_width = w / image_info["width"]
                    norm_height = h / image_info["height"]
                    
                    yolo_line = f"{ann['category_id'] - 1} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
                    yolo_annotations.append(yolo_line)
                
                # Save YOLO annotation file
                yolo_file = output_dir / f"{Path(image_info['file_name']).stem}.txt"
                with open(yolo_file, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
            
            self.logger.info(f"Saved YOLO annotations to {output_dir}")
        
        else:
            raise ValueError(f"Unsupported output format: {format}")
    
    def get_image_metadata(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get metadata for an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image metadata dictionary
        """
        image_path = Path(image_path)
        
        try:
            with Image.open(image_path) as img:
                metadata = {
                    "file_name": image_path.name,
                    "file_size": image_path.stat().st_size,
                    "width": img.width,
                    "height": img.height,
                    "mode": img.mode,
                    "format": img.format,
                    "has_transparency": img.mode in ("RGBA", "LA") or "transparency" in img.info
                }
                
                # Add EXIF data if available
                if hasattr(img, '_getexif') and img._getexif() is not None:
                    metadata["exif"] = dict(img._getexif())
                
                return metadata
                
        except Exception as e:
            self.logger.error(f"Failed to get metadata for {image_path}: {e}")
            return {}
    
    def validate_dataset(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate dataset quality and consistency.
        
        Args:
            dataset: Dataset dictionary
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            "total_images": len(dataset["images"]),
            "total_annotations": len(dataset["annotations"]["annotations"]),
            "validation_errors": [],
            "warnings": []
        }
        
        # Check for missing images
        for image_info in dataset["images"]:
            image_path = Path(image_info["path"])
            if not image_path.exists():
                validation_results["validation_errors"].append(f"Image not found: {image_path}")
        
        # Check annotation consistency
        image_ids = {img["id"] for img in dataset["images"]}
        for ann in dataset["annotations"]["annotations"]:
            if ann["image_id"] not in image_ids:
                validation_results["validation_errors"].append(f"Annotation references non-existent image: {ann['image_id']}")
        
        # Check for empty images
        for image_info in dataset["images"]:
            image_annotations = [
                ann for ann in dataset["annotations"]["annotations"] 
                if ann["image_id"] == image_info["id"]
            ]
            if not image_annotations:
                validation_results["warnings"].append(f"No annotations for image: {image_info['file_name']}")
        
        validation_results["is_valid"] = len(validation_results["validation_errors"]) == 0
        
        return validation_results
