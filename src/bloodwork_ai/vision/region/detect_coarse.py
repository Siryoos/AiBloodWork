"""Coarse smear detection module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import joblib
from scipy import ndimage
from skimage import feature, filters, morphology
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from .config import CoarseDetectionConfig
from .preprocess import romanowsky_deconvolution


@dataclass
class CoarseSmearDetector:
    """Perform coarse smear presence detection using ML classifier."""

    config: CoarseDetectionConfig
    _model: Optional[object] = None

    def __post_init__(self):
        """Load or initialize the classification model."""
        if self.config.model_path and Path(self.config.model_path).exists():
            self._model = joblib.load(self.config.model_path)
        else:
            self._model = self._create_default_model()

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Return a coarse probability mask using feature-based classification."""
        downsampled = self._downsample_image(image, scale=4)

        features = self._extract_features(downsampled)

        if self._model is not None:
            probabilities = self._classify_patches(features, downsampled.shape[:2])
        else:
            probabilities = self._fallback_detection(downsampled)

        mask = self._postprocess_mask(probabilities)

        return cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    def _downsample_image(self, image: np.ndarray, scale: int = 4) -> np.ndarray:
        """Downsample image for faster processing."""
        h, w = image.shape[:2]
        new_h, new_w = h // scale, w // scale
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features for smear detection classification."""
        if image.ndim == 3:
            eosin, methylene = romanowsky_deconvolution(image)
        else:
            eosin = methylene = image.astype(np.float32) / 255.0

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        gray = gray.astype(np.float32) / 255.0

        features = []

        features.extend(self._color_features(image, eosin, methylene))
        features.extend(self._texture_features(gray))
        features.extend(self._gradient_features(gray))

        return np.array(features, dtype=np.float32)

    def _color_features(self, image: np.ndarray, eosin: np.ndarray, methylene: np.ndarray) -> list:
        """Extract color-based features."""
        features = []

        if image.ndim == 3:
            rgb_means = np.mean(image, axis=(0, 1))
            rgb_stds = np.std(image, axis=(0, 1))
            features.extend(rgb_means)
            features.extend(rgb_stds)

            hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)
            hsv_means = np.mean(hsv, axis=(0, 1))
            hsv_stds = np.std(hsv, axis=(0, 1))
            features.extend(hsv_means)
            features.extend(hsv_stds)
        else:
            gray_mean = np.mean(image)
            gray_std = np.std(image)
            features.extend([gray_mean, gray_std] * 6)

        eosin_mean = np.mean(eosin)
        eosin_std = np.std(eosin)
        methylene_mean = np.mean(methylene)
        methylene_std = np.std(methylene)

        features.extend([eosin_mean, eosin_std, methylene_mean, methylene_std])

        return features

    def _texture_features(self, gray: np.ndarray) -> list:
        """Extract texture-based features using LBP and other descriptors."""
        features = []

        lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 9), density=True)
        features.extend(lbp_hist)

        glcm = feature.graycomatrix(
            (gray * 255).astype(np.uint8),
            distances=[1],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=256,
            symmetric=True,
            normed=True
        )

        contrast = feature.graycoprops(glcm, 'contrast').mean()
        dissimilarity = feature.graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = feature.graycoprops(glcm, 'homogeneity').mean()
        energy = feature.graycoprops(glcm, 'energy').mean()

        features.extend([contrast, dissimilarity, homogeneity, energy])

        return features

    def _gradient_features(self, gray: np.ndarray) -> list:
        """Extract gradient-based features."""
        features = []

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        features.append(np.mean(gradient_magnitude))
        features.append(np.std(gradient_magnitude))

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features.append(np.mean(np.abs(laplacian)))
        features.append(np.std(laplacian))

        tenengrad = cv2.filter2D(gray, cv2.CV_64F, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))
        features.append(np.mean(tenengrad**2))

        hf_energy = np.mean(filters.gaussian(gray, sigma=1) - filters.gaussian(gray, sigma=2))
        features.append(hf_energy)

        return features

    def _classify_patches(self, features: np.ndarray, image_shape: tuple) -> np.ndarray:
        """Classify image patches using the trained model."""
        patch_size = 32
        h, w = image_shape

        n_patches_y = max(1, h // patch_size)
        n_patches_x = max(1, w // patch_size)

        probabilities = np.zeros((n_patches_y, n_patches_x), dtype=np.float32)

        if hasattr(self._model, 'predict_proba'):
            prob = self._model.predict_proba(features.reshape(1, -1))[0]
            if len(prob) > 1:
                prob_smear = prob[1]
            else:
                prob_smear = prob[0]
        else:
            prob_smear = max(0.0, min(1.0, self._model.predict(features.reshape(1, -1))[0]))

        probabilities.fill(prob_smear)

        return probabilities

    def _fallback_detection(self, image: np.ndarray) -> np.ndarray:
        """Fallback detection when no model is available."""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        gray = gray.astype(np.float32) / 255.0

        background = morphology.white_tophat(gray, morphology.disk(50))

        contrast = filters.rank.variance(
            (gray * 255).astype(np.uint8),
            morphology.disk(5)
        ).astype(np.float32) / 255.0

        binary = gray > filters.threshold_otsu(gray)

        combined_score = 0.4 * background + 0.4 * contrast + 0.2 * binary.astype(np.float32)

        return combined_score

    def _postprocess_mask(self, probabilities: np.ndarray) -> np.ndarray:
        """Post-process probability map to create final mask."""
        mask = probabilities > self.config.threshold

        if self.config.min_component_area_um2 > 0:
            min_area_px = int(self.config.min_component_area_um2 / (10.0 ** 2))  # Assume ~10µm/px
            mask = morphology.remove_small_objects(mask, min_size=min_area_px)

        if self.config.smoothing_radius_um > 0:
            radius_px = max(1, int(self.config.smoothing_radius_um / 10.0))
            mask = morphology.closing(mask, morphology.disk(radius_px))

        return mask.astype(np.float32)

    def _create_default_model(self) -> object:
        """Create a default XGBoost model for smear detection."""
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )

        return model

    def train_model(self, training_data: list, labels: list, model_save_path: Optional[Path] = None) -> None:
        """Train the coarse detection model on provided data."""
        if not training_data or not labels:
            raise ValueError("Training data and labels cannot be empty")

        features_list = []
        for image in training_data:
            downsampled = self._downsample_image(image, scale=4)
            features = self._extract_features(downsampled)
            features_list.append(features)

        X = np.array(features_list)
        y = np.array(labels)

        if self.config.model_path and 'xgb' in str(self.config.model_path).lower():
            self._model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                random_state=42,
                eval_metric='logloss'
            )
        elif self.config.model_path and 'rf' in str(self.config.model_path).lower():
            self._model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:
            self._model = LogisticRegression(random_state=42, max_iter=1000)

        self._model.fit(X, y)

        if model_save_path:
            joblib.dump(self._model, model_save_path)
            print(f"Model saved to {model_save_path}")


class SmearPresenceClassifier:
    """Binary classifier for smear presence at tile level."""

    def __init__(self, feature_dim: int = 32):
        self.feature_dim = feature_dim
        self.model = LogisticRegression(random_state=42)

    def extract_tile_features(self, tile: np.ndarray) -> np.ndarray:
        """Extract features from a single tile for presence classification."""
        if tile.ndim == 3:
            gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
            eosin, methylene = romanowsky_deconvolution(tile)
        else:
            gray = tile
            eosin = methylene = gray

        gray = gray.astype(np.float32) / 255.0

        features = []

        features.extend([np.mean(gray), np.std(gray), np.min(gray), np.max(gray)])

        if tile.ndim == 3:
            for channel in range(3):
                ch = tile[:, :, channel] / 255.0
                features.extend([np.mean(ch), np.std(ch)])

        features.extend([np.mean(eosin), np.std(eosin), np.mean(methylene), np.std(methylene)])

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        features.extend([np.mean(gradient_mag), np.std(gradient_mag)])

        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        features.append(laplacian_var)

        lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, density=True)
        features.extend(lbp_hist)

        return np.array(features[:self.feature_dim], dtype=np.float32)

    def fit(self, tiles: list, labels: list) -> None:
        """Train the classifier on tiles and binary labels."""
        X = np.array([self.extract_tile_features(tile) for tile in tiles])
        y = np.array(labels)
        self.model.fit(X, y)

    def predict_proba(self, tile: np.ndarray) -> float:
        """Predict probability that tile contains smear."""
        features = self.extract_tile_features(tile).reshape(1, -1)
        return self.model.predict_proba(features)[0, 1]

