"""Hybrid smear segmentation implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from skimage import filters, morphology, segmentation

from .config import SegmentationConfig


class LightweightUNet(nn.Module):
    """Lightweight U-Net for blood smear segmentation refinement."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_filters: int = 32):
        super().__init__()

        # Encoder
        self.enc1 = self._conv_block(in_channels, base_filters)
        self.enc2 = self._conv_block(base_filters, base_filters * 2)
        self.enc3 = self._conv_block(base_filters * 2, base_filters * 4)

        # Bottleneck
        self.bottleneck = self._conv_block(base_filters * 4, base_filters * 8)

        # Decoder
        self.dec3 = self._conv_block(base_filters * 8 + base_filters * 4, base_filters * 4)
        self.dec2 = self._conv_block(base_filters * 4 + base_filters * 2, base_filters * 2)
        self.dec1 = self._conv_block(base_filters * 2 + base_filters, base_filters)

        # Output
        self.final = nn.Conv2d(base_filters, out_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder
        d3 = self.dec3(torch.cat([self.upsample(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))

        return torch.sigmoid(self.final(d1))


@dataclass
class HybridSegmenter:
    """Segment smear region using classical methods and CNN refinement."""

    config: SegmentationConfig
    _cnn_model: Optional[nn.Module] = None
    _device: Optional[torch.device] = None

    def __post_init__(self):
        """Initialize CNN model if available."""
        if self.config.cnn_model_path and Path(self.config.cnn_model_path).exists():
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._cnn_model = LightweightUNet(in_channels=3, out_channels=1)
            self._cnn_model.load_state_dict(torch.load(self.config.cnn_model_path, map_location=self._device))
            self._cnn_model.to(self._device)
            self._cnn_model.eval()

    def segment(self, image: np.ndarray, coarse_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Perform hybrid segmentation using classical methods and CNN refinement."""
        # Stage 1: Classical segmentation
        classical_mask = self._classical_segmentation(image, coarse_mask)

        # Stage 2: CNN refinement (if model available)
        if self._cnn_model is not None:
            refined_mask = self._cnn_refinement(image, classical_mask)
        else:
            refined_mask = classical_mask

        # Stage 3: Post-processing
        final_mask = self._postprocess_mask(refined_mask)

        return final_mask

    def _classical_segmentation(self, image: np.ndarray, coarse_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Classical segmentation using morphological operations and contrast metrics."""
        # Convert to grayscale
        gray = self._to_grayscale(image)

        # Background estimation
        background = self._estimate_background(gray)
        contrast_image = gray - background

        # Compute contrast metrics
        tenengrad = self._compute_tenengrad(contrast_image)
        hf_energy = self._compute_hf_energy(contrast_image)

        # Weighted combination
        w1, w2 = self.config.contrast_weights
        contrast_metric = w1 * tenengrad + w2 * hf_energy

        # Otsu thresholding with bias
        threshold = filters.threshold_otsu(contrast_metric) + self.config.otsu_bias
        binary_mask = contrast_metric > threshold

        # Graph cut refinement
        if self.config.graph_cut_lambda > 0:
            binary_mask = self._graph_cut_refinement(gray, binary_mask)

        # Apply coarse mask constraint
        if coarse_mask is not None:
            binary_mask = binary_mask & (coarse_mask > 0.5)

        return binary_mask.astype(np.float32)

    def _cnn_refinement(self, image: np.ndarray, classical_mask: np.ndarray) -> np.ndarray:
        """Refine segmentation using CNN."""
        if self._cnn_model is None or self._device is None:
            return classical_mask

        # Prepare input at CNN scale
        scale = self.config.cnn_input_scale
        h, w = image.shape[:2]
        cnn_h, cnn_w = int(h * scale), int(w * scale)

        # Resize image and mask
        image_resized = cv2.resize(image, (cnn_w, cnn_h), interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(classical_mask, (cnn_w, cnn_h), interpolation=cv2.INTER_NEAREST)

        # Normalize and convert to tensor
        if image_resized.ndim == 2:
            image_resized = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)

        image_tensor = torch.from_numpy(image_resized.transpose(2, 0, 1)).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self._device)

        # CNN inference
        with torch.no_grad():
            cnn_output = self._cnn_model(image_tensor)
            cnn_mask = cnn_output.squeeze().cpu().numpy()

        # Resize back to original size
        cnn_mask_full = cv2.resize(cnn_mask, (w, h), interpolation=cv2.INTER_LINEAR)

        # Combine CNN output with classical mask
        # Use CNN where it's confident, fall back to classical elsewhere
        confidence_threshold = 0.3
        high_confidence = (cnn_mask_full > confidence_threshold) | (cnn_mask_full < (1 - confidence_threshold))

        refined_mask = np.where(high_confidence, cnn_mask_full, classical_mask)

        return refined_mask

    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Post-process mask with morphological operations."""
        # Convert to binary
        binary_mask = mask > 0.5

        # Remove small holes
        if self.config.remove_holes_area_um2 > 0:
            min_hole_area_px = max(1, int(self.config.remove_holes_area_um2 / 100))  # Assume ~10µm/px
            binary_mask = morphology.remove_small_holes(binary_mask, area_threshold=min_hole_area_px)

        # Remove small objects
        min_object_area_px = max(1, int(self.config.remove_holes_area_um2 / 50))
        binary_mask = morphology.remove_small_objects(binary_mask, min_size=min_object_area_px)

        # Morphological closing to smooth boundaries
        kernel_size = 3
        kernel = morphology.disk(kernel_size)
        binary_mask = morphology.closing(binary_mask, kernel)

        return binary_mask.astype(np.float32)

    def _estimate_background(self, gray: np.ndarray) -> np.ndarray:
        """Estimate background using morphological opening."""
        # Adaptive kernel size based on image dimensions
        kernel_size = max(15, min(gray.shape) // 20)
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

        # Smooth background
        background = cv2.GaussianBlur(background, (kernel_size//2*2+1, kernel_size//2*2+1), 0)

        return background

    def _compute_tenengrad(self, image: np.ndarray) -> np.ndarray:
        """Compute Tenengrad focus measure."""
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = sobel_x**2 + sobel_y**2

        # Normalize
        tenengrad = (tenengrad - tenengrad.min()) / (tenengrad.ptp() + 1e-8)

        return tenengrad.astype(np.float32)

    def _compute_hf_energy(self, image: np.ndarray) -> np.ndarray:
        """Compute high-frequency energy."""
        # Laplacian of Gaussian
        blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

        # High-frequency energy
        hf_energy = np.abs(laplacian)

        # Normalize
        hf_energy = (hf_energy - hf_energy.min()) / (hf_energy.ptp() + 1e-8)

        return hf_energy.astype(np.float32)

    def _graph_cut_refinement(self, image: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
        """Refine segmentation using graph cuts."""
        # Create markers for watershed/graph cut
        markers = np.zeros_like(binary_mask, dtype=np.int32)

        # Sure foreground (eroded binary mask)
        kernel = morphology.disk(2)
        sure_fg = morphology.erosion(binary_mask, kernel)
        markers[sure_fg] = 2

        # Sure background (dilated inverse)
        sure_bg = morphology.dilation(~binary_mask, morphology.disk(5))
        markers[sure_bg] = 1

        # Apply watershed
        if image.ndim == 2:
            watershed_input = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            watershed_input = (image * 255).astype(np.uint8)

        labels = cv2.watershed(watershed_input, markers)

        # Convert back to binary mask
        refined_mask = (labels == 2)

        return refined_mask

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        if image.ndim == 2:
            return image.astype(np.float32)
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

    @staticmethod
    def _remove_small_holes(mask: np.ndarray, min_area_px: int = 128) -> np.ndarray:
        """Legacy method for removing small holes."""
        mask_uint = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = np.zeros_like(mask_uint)
        for contour in contours:
            if cv2.contourArea(contour) >= min_area_px:
                cv2.drawContours(filtered, [contour], -1, 1, thickness=-1)
        return filtered.astype(np.float32)


class CNNTrainer:
    """Trainer for the lightweight U-Net model."""

    def __init__(self, model: LightweightUNet, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.BCELoss()

    def train_epoch(self, dataloader, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item:.4f}')

        return total_loss / len(dataloader)

    def validate(self, dataloader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0

        with torch.no_grad():
            for images, masks in dataloader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()

                # Compute IoU
                pred_binary = (outputs > 0.5).float()
                intersection = (pred_binary * masks).sum()
                union = pred_binary.sum() + masks.sum() - intersection
                iou = intersection / (union + 1e-8)
                total_iou += iou.item()

        avg_loss = total_loss / len(dataloader)
        avg_iou = total_iou / len(dataloader)

        return avg_loss, avg_iou

    def save_model(self, path: Path) -> None:
        """Save model state dict."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")


class SegmentationDataset(torch.utils.data.Dataset):
    """Dataset for training segmentation models."""

    def __init__(self, images: list, masks: list, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Convert to tensors
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        return image, mask

