"""Feature engineering for tabular lab data."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    SelectFromModel,
    RFE,
    mutual_info_classif,
    mutual_info_regression,
    f_classif,
    f_regression,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, ElasticNetCV

from ..utils.log import get_logger


class FeatureEngineer:
    """Feature engineering for tabular lab data."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the feature engineer.
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = get_logger(__name__)
        self.config = self._load_config(config_path)
        self.feature_selector = None
        self.feature_names = None
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file."""
        if config_path is None:
            config_path = "configs/data/tabular.yaml"
        
        try:
            from ..utils.io import load_data
            return load_data(config_path, file_type="yaml")
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            return {}
    
    def engineer_features(
        self,
        data: pd.DataFrame,
        target: Optional[pd.Series] = None,
        task_type: str = "classification"
    ) -> pd.DataFrame:
        """
        Engineer features from raw lab data.
        
        Args:
            data: Raw DataFrame
            target: Target variable for supervised feature selection
            task_type: Type of task (classification, regression)
            
        Returns:
            DataFrame with engineered features
        """
        self.logger.info(f"Engineering features for {len(data)} records")
        
        # Create a copy to avoid modifying original data
        engineered_data = data.copy()
        
        # Create ratio features
        if self.config.get("feature_engineering", {}).get("enable_ratios", True):
            engineered_data = self._create_ratio_features(engineered_data)
        
        # Create trend features (if time series data available)
        if self.config.get("feature_engineering", {}).get("enable_trends", True):
            engineered_data = self._create_trend_features(engineered_data)
        
        # Create interaction features
        if self.config.get("feature_engineering", {}).get("enable_interactions", True):
            engineered_data = self._create_interaction_features(engineered_data)
        
        # Create derived features
        engineered_data = self._create_derived_features(engineered_data)
        
        # Select features if target is provided
        if target is not None:
            engineered_data = self._select_features(engineered_data, target, task_type)
        
        self.logger.info(f"Feature engineering completed. Final shape: {engineered_data.shape}")
        return engineered_data
    
    def _create_ratio_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create ratio features from existing columns.
        
        Args:
            data: DataFrame to add ratio features to
            
        Returns:
            DataFrame with ratio features added
        """
        ratio_configs = self.config.get("feature_engineering", {}).get("ratios", {})
        
        for panel_type, ratios in ratio_configs.items():
            for ratio_config in ratios:
                name = ratio_config["name"]
                numerator = ratio_config["numerator"]
                denominator = ratio_config["denominator"]
                
                if numerator in data.columns and denominator in data.columns:
                    # Avoid division by zero
                    data[name] = np.where(
                        data[denominator] != 0,
                        data[numerator] / data[denominator],
                        0
                    )
                    self.logger.debug(f"Created ratio feature: {name}")
                else:
                    self.logger.warning(f"Could not create ratio {name}: missing columns")
        
        return data
    
    def _create_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create trend features for time series data.
        
        Args:
            data: DataFrame to add trend features to
            
        Returns:
            DataFrame with trend features added
        """
        # This is a simplified implementation
        # In practice, you would need time series data with timestamps
        
        # Create rolling statistics for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if data[col].isnull().all():
                continue
            
            # Create rolling mean and std (if we had time series data)
            # For now, create simple statistical features
            data[f"{col}_mean"] = data[col].mean()
            data[f"{col}_std"] = data[col].std()
            data[f"{col}_zscore"] = (data[col] - data[col].mean()) / data[col].std()
        
        return data
    
    def _create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between columns.
        
        Args:
            data: DataFrame to add interaction features to
            
        Returns:
            DataFrame with interaction features added
        """
        # Create polynomial features for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Limit to avoid too many features
        max_features = self.config.get("feature_engineering", {}).get("max_features", 100)
        if len(numeric_cols) > max_features:
            numeric_cols = numeric_cols[:max_features]
        
        # Create squared features
        for col in numeric_cols:
            if data[col].isnull().all():
                continue
            
            data[f"{col}_squared"] = data[col] ** 2
            data[f"{col}_sqrt"] = np.sqrt(np.abs(data[col]))
            data[f"{col}_log"] = np.log1p(np.abs(data[col]))
        
        # Create interaction features between highly correlated pairs
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr().abs()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.7:  # High correlation threshold
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            # Create interaction features for highly correlated pairs
            for col1, col2 in high_corr_pairs[:10]:  # Limit to 10 interactions
                if col1 in data.columns and col2 in data.columns:
                    data[f"{col1}_x_{col2}"] = data[col1] * data[col2]
                    data[f"{col1}_div_{col2}"] = np.where(
                        data[col2] != 0,
                        data[col1] / data[col2],
                        0
                    )
        
        return data
    
    def _create_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features specific to lab data.
        
        Args:
            data: DataFrame to add derived features to
            
        Returns:
            DataFrame with derived features added
        """
        # CBC derived features
        if "hb" in data.columns and "hct" in data.columns:
            data["mchc_calculated"] = np.where(
                data["hct"] != 0,
                data["hb"] / data["hct"] * 100,
                0
            )
        
        if "rbc" in data.columns and "hct" in data.columns:
            data["mcv_calculated"] = np.where(
                data["rbc"] != 0,
                data["hct"] / data["rbc"] * 10,
                0
            )
        
        if "hb" in data.columns and "rbc" in data.columns:
            data["mch_calculated"] = np.where(
                data["rbc"] != 0,
                data["hb"] / data["rbc"] * 10,
                0
            )
        
        # CMP derived features
        if "bun" in data.columns and "cr" in data.columns:
            data["bun_cr_ratio"] = np.where(
                data["cr"] != 0,
                data["bun"] / data["cr"],
                0
            )
        
        if "ast" in data.columns and "alt" in data.columns:
            data["ast_alt_ratio"] = np.where(
                data["alt"] != 0,
                data["ast"] / data["alt"],
                0
            )
        
        # Lipid derived features
        if "total_chol" in data.columns and "hdl" in data.columns:
            data["non_hdl_chol"] = data["total_chol"] - data["hdl"]
        
        if "ldl" in data.columns and "hdl" in data.columns:
            data["ldl_hdl_ratio"] = np.where(
                data["hdl"] != 0,
                data["ldl"] / data["hdl"],
                0
            )
        
        # Thyroid derived features
        if "tsh" in data.columns and "ft4" in data.columns:
            data["tsh_ft4_ratio"] = np.where(
                data["ft4"] != 0,
                data["tsh"] / data["ft4"],
                0
            )
        
        # Vitamins/Iron derived features
        if "iron" in data.columns and "tibc" in data.columns:
            data["transferrin_sat_calculated"] = np.where(
                data["tibc"] != 0,
                data["iron"] / data["tibc"] * 100,
                0
            )
        
        return data
    
    def _select_features(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        task_type: str = "classification"
    ) -> pd.DataFrame:
        """
        Select the most important features.
        
        Args:
            data: DataFrame with features
            target: Target variable
            task_type: Type of task (classification, regression)
            
        Returns:
            DataFrame with selected features
        """
        feature_selection_config = self.config.get("feature_selection", {})
        
        if not feature_selection_config.get("enabled", True):
            return data
        
        method = feature_selection_config.get("method", "mutual_info")
        max_features = feature_selection_config.get("max_features", 50)
        min_features = feature_selection_config.get("min_features", 5)
        
        # Only select from numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < min_features:
            self.logger.warning("Not enough numeric features for selection")
            return data
        
        self.logger.info(f"Selecting features using {method} method")
        
        try:
            if method == "mutual_info":
                if task_type == "classification":
                    selector = SelectKBest(
                        score_func=mutual_info_classif,
                        k=min(max_features, len(numeric_cols))
                    )
                else:
                    selector = SelectKBest(
                        score_func=mutual_info_regression,
                        k=min(max_features, len(numeric_cols))
                    )
                
                selector.fit(data[numeric_cols], target)
                selected_features = numeric_cols[selector.get_support()]
                
            elif method == "f_score":
                if task_type == "classification":
                    selector = SelectKBest(
                        score_func=f_classif,
                        k=min(max_features, len(numeric_cols))
                    )
                else:
                    selector = SelectKBest(
                        score_func=f_regression,
                        k=min(max_features, len(numeric_cols))
                    )
                
                selector.fit(data[numeric_cols], target)
                selected_features = numeric_cols[selector.get_support()]
                
            elif method == "l1":
                if task_type == "classification":
                    model = LassoCV(cv=5, random_state=42)
                else:
                    model = LassoCV(cv=5, random_state=42)
                
                model.fit(data[numeric_cols], target)
                selector = SelectFromModel(model, prefit=True)
                selected_features = numeric_cols[selector.get_support()]
                
            elif method == "rfe":
                if task_type == "classification":
                    estimator = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    estimator = RandomForestRegressor(n_estimators=100, random_state=42)
                
                selector = RFE(
                    estimator,
                    n_features_to_select=min(max_features, len(numeric_cols))
                )
                selector.fit(data[numeric_cols], target)
                selected_features = numeric_cols[selector.get_support()]
                
            else:
                self.logger.warning(f"Unknown feature selection method: {method}")
                return data
            
            # Store selector for later use
            self.feature_selector = selector
            self.feature_names = selected_features
            
            # Select features
            selected_data = data[selected_features].copy()
            
            # Add non-numeric columns back
            non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric_cols) > 0:
                selected_data = pd.concat([selected_data, data[non_numeric_cols]], axis=1)
            
            self.logger.info(f"Selected {len(selected_features)} features out of {len(numeric_cols)}")
            return selected_data
            
        except Exception as e:
            self.logger.error(f"Feature selection failed: {e}")
            return data
    
    def get_feature_importance(self, data: pd.DataFrame, target: pd.Series) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            data: DataFrame with features
            target: Target variable
            
        Returns:
            Dictionary of feature importance scores
        """
        if self.feature_selector is None:
            self.logger.warning("No feature selector found, fitting new one")
            self._select_features(data, target)
        
        if self.feature_selector is None:
            return {}
        
        # Get feature scores
        if hasattr(self.feature_selector, 'scores_'):
            scores = self.feature_selector.scores_
        elif hasattr(self.feature_selector, 'feature_importances_'):
            scores = self.feature_selector.feature_importances_
        else:
            return {}
        
        # Create feature importance dictionary
        if self.feature_names is not None:
            importance_dict = dict(zip(self.feature_names, scores))
        else:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            importance_dict = dict(zip(numeric_cols, scores))
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return importance_dict
    
    def transform_new_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using the fitted feature selector.
        
        Args:
            data: New data to transform
            
        Returns:
            Transformed data
        """
        if self.feature_selector is None:
            self.logger.warning("No fitted feature selector found")
            return data
        
        if self.feature_names is None:
            self.logger.warning("No feature names found")
            return data
        
        # Select the same features
        selected_data = data[self.feature_names].copy()
        
        # Add non-numeric columns back
        non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            selected_data = pd.concat([selected_data, data[non_numeric_cols]], axis=1)
        
        return selected_data
    
    def get_feature_engineering_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a feature engineering report.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Feature engineering report dictionary
        """
        report = {
            "total_features": len(data.columns),
            "numeric_features": len(data.select_dtypes(include=[np.number]).columns),
            "categorical_features": len(data.select_dtypes(include=['object', 'category']).columns),
            "feature_names": data.columns.tolist(),
            "memory_usage": data.memory_usage(deep=True).sum(),
        }
        
        # Add feature statistics
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            report["numeric_feature_statistics"] = data[numeric_cols].describe().to_dict()
        
        # Add correlation matrix for highly correlated features
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr().abs()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.8:  # High correlation threshold
                        high_corr_pairs.append({
                            "feature1": corr_matrix.columns[i],
                            "feature2": corr_matrix.columns[j],
                            "correlation": corr_matrix.iloc[i, j]
                        })
            
            report["high_correlation_pairs"] = high_corr_pairs
        
        return report
