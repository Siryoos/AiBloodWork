"""Anemia classifier task implementation."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from ...models.tabular import XGBoostModel, LightGBMModel, MLPModel
from ...preprocessing import TabularCleaner, FeatureEngineer
from ...ingestion import TabularLoader
from ...utils.log import get_logger
from ...utils.metrics import calculate_metrics, calculate_clinical_metrics


class AnemiaClassifier:
    """Anemia classification task."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the anemia classifier.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = get_logger(__name__)
        self.config = config or {}
        self.model = None
        self.cleaner = None
        self.engineer = None
        self.is_trained = False
        
    def prepare_data(
        self,
        data_path: str,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Prepare data for training.
        
        Args:
            data_path: Path to the data file
            test_size: Test set size
            val_size: Validation set size
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        self.logger.info("Preparing data for anemia classification")
        
        # Load data
        loader = TabularLoader()
        data = loader.load_panel_data(data_path, "cbc")
        
        # Clean data
        self.cleaner = TabularCleaner()
        clean_data = self.cleaner.clean_data(data, "cbc")
        
        # Engineer features
        self.engineer = FeatureEngineer()
        features = self.engineer.engineer_features(clean_data)
        
        # Prepare features and target
        feature_cols = [col for col in features.columns if col not in ["patient_id", "date", "anemia_type"]]
        X = features[feature_cols]
        y = features["anemia_type"]
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=random_state, stratify=y_temp
        )
        
        self.logger.info(f"Data prepared: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        algorithm: str = "xgboost"
    ) -> Dict[str, Any]:
        """
        Train the anemia classifier.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            algorithm: Algorithm to use (xgboost, lightgbm, mlp)
            
        Returns:
            Training results dictionary
        """
        self.logger.info(f"Training anemia classifier using {algorithm}")
        
        # Initialize model
        if algorithm == "xgboost":
            self.model = XGBoostModel(self.config.get("xgboost", {}))
        elif algorithm == "lightgbm":
            self.model = LightGBMModel(self.config.get("lightgbm", {}))
        elif algorithm == "mlp":
            self.model = MLPModel(self.config.get("mlp", {}))
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Train model
        self.model.fit(X_train, y_train, X_val, y_val, task_type="classification")
        
        self.is_trained = True
        
        # Evaluate on validation set
        if X_val is not None and y_val is not None:
            val_metrics = self.model.evaluate(X_val, y_val)
            self.logger.info(f"Validation metrics: {val_metrics}")
        
        return {
            "algorithm": algorithm,
            "is_trained": True,
            "feature_count": len(X_train.columns),
            "training_samples": len(X_train)
        }
    
    def predict(
        self,
        X: pd.DataFrame,
        return_proba: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions.
        
        Args:
            X: Features to predict
            return_proba: Whether to return probabilities
            
        Returns:
            Predictions or (predictions, probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X, return_proba=return_proba)
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Evaluate the model.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Evaluation results dictionary
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, predictions, probabilities, task_type="multiclass")
        clinical_metrics = calculate_clinical_metrics(y_test, predictions, probabilities)
        
        # Classification report
        class_report = classification_report(y_test, predictions, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        results = {
            "metrics": metrics,
            "clinical_metrics": clinical_metrics,
            "classification_report": class_report,
            "confusion_matrix": cm.tolist(),
            "feature_importance": self.model.get_feature_importance() if hasattr(self.model, 'get_feature_importance') else {}
        }
        
        self.logger.info(f"Evaluation completed. Accuracy: {metrics.get('accuracy', 0):.3f}")
        
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        if hasattr(self.model, 'get_feature_importance'):
            return self.model.get_feature_importance()
        else:
            return {}
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        self.model.save_model(filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        # This is a simplified implementation
        # In practice, you would need to know the algorithm type
        self.model = XGBoostModel()
        self.model.load_model(filepath)
        self.is_trained = True
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if not self.is_trained:
            return {"is_trained": False}
        
        info = {
            "is_trained": True,
            "algorithm": type(self.model).__name__,
            "feature_count": len(self.model.feature_names) if hasattr(self.model, 'feature_names') else 0,
            "config": self.config
        }
        
        if hasattr(self.model, 'get_model_info'):
            info.update(self.model.get_model_info())
        
        return info
