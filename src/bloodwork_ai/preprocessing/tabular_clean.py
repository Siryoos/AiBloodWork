"""Tabular data cleaning and preprocessing."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from ..utils.log import get_logger


class TabularCleaner:
    """Clean and preprocess tabular lab data."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the tabular cleaner.
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = get_logger(__name__)
        self.config = self._load_config(config_path)
        self.imputer = None
        self.scaler = None
        self.outlier_handler = None
        
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
    
    def clean_data(
        self,
        data: pd.DataFrame,
        panel_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Clean tabular data with imputation, outlier handling, and scaling.
        
        Args:
            data: Raw DataFrame
            panel_type: Type of lab panel for panel-specific cleaning
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info(f"Cleaning data with {len(data)} records and {len(data.columns)} columns")
        
        # Create a copy to avoid modifying original data
        cleaned_data = data.copy()
        
        # Handle missing values
        cleaned_data = self._handle_missing_values(cleaned_data, panel_type)
        
        # Handle outliers
        cleaned_data = self._handle_outliers(cleaned_data, panel_type)
        
        # Scale features
        cleaned_data = self._scale_features(cleaned_data, panel_type)
        
        self.logger.info("Data cleaning completed")
        return cleaned_data
    
    def _handle_missing_values(
        self,
        data: pd.DataFrame,
        panel_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in the data.
        
        Args:
            data: DataFrame with potential missing values
            panel_type: Type of lab panel for panel-specific imputation
            
        Returns:
            DataFrame with imputed values
        """
        imputation_config = self.config.get("imputation", {})
        strategy = imputation_config.get("strategy", "median")
        
        # Get panel-specific imputation strategy
        if panel_type and panel_type in imputation_config.get("panel_specific", {}):
            panel_config = imputation_config["panel_specific"][panel_type]
            strategy = panel_config.get("strategy", strategy)
        
        self.logger.info(f"Handling missing values using {strategy} strategy")
        
        # Separate numeric and categorical columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        # Handle numeric columns
        if len(numeric_cols) > 0:
            if strategy == "knn":
                self.imputer = KNNImputer(n_neighbors=imputation_config.get("knn_neighbors", 5))
                data[numeric_cols] = self.imputer.fit_transform(data[numeric_cols])
            else:
                imputer = SimpleImputer(strategy=strategy)
                data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
        
        # Handle categorical columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                if data[col].isnull().any():
                    if strategy == "most_frequent":
                        mode_value = data[col].mode()
                        if len(mode_value) > 0:
                            data[col].fillna(mode_value[0], inplace=True)
                        else:
                            data[col].fillna("unknown", inplace=True)
                    else:
                        data[col].fillna("unknown", inplace=True)
        
        # Log missing value statistics
        missing_counts = data.isnull().sum()
        if missing_counts.sum() > 0:
            self.logger.warning(f"Still have missing values after imputation: {missing_counts[missing_counts > 0].to_dict()}")
        else:
            self.logger.info("All missing values have been handled")
        
        return data
    
    def _handle_outliers(
        self,
        data: pd.DataFrame,
        panel_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Handle outliers in the data.
        
        Args:
            data: DataFrame with potential outliers
            panel_type: Type of lab panel for panel-specific outlier handling
            
        Returns:
            DataFrame with outliers handled
        """
        outlier_config = self.config.get("outlier_detection", {})
        method = outlier_config.get("method", "winsorize")
        
        # Get panel-specific outlier handling
        if panel_type and panel_type in outlier_config.get("panel_specific", {}):
            panel_config = outlier_config["panel_specific"][panel_type]
            method = panel_config.get("method", method)
        
        self.logger.info(f"Handling outliers using {method} method")
        
        # Only handle numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if data[col].isnull().all():
                continue
                
            original_values = data[col].copy()
            
            if method == "winsorize":
                data[col] = self._winsorize(data[col], outlier_config.get("winsorize_limits", [0.05, 0.95]))
            elif method == "iqr":
                data[col] = self._iqr_filter(data[col], outlier_config.get("iqr_factor", 1.5))
            elif method == "zscore":
                data[col] = self._zscore_filter(data[col], outlier_config.get("zscore_threshold", 3.0))
            
            # Log outlier statistics
            outliers_removed = (original_values != data[col]).sum()
            if outliers_removed > 0:
                self.logger.info(f"Handled {outliers_removed} outliers in column {col}")
        
        return data
    
    def _winsorize(self, series: pd.Series, limits: List[float]) -> pd.Series:
        """Winsorize a series to the specified limits."""
        lower_limit = series.quantile(limits[0])
        upper_limit = series.quantile(limits[1])
        return series.clip(lower=lower_limit, upper=upper_limit)
    
    def _iqr_filter(self, series: pd.Series, factor: float = 1.5) -> pd.Series:
        """Filter outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        return series.clip(lower=lower_bound, upper=upper_bound)
    
    def _zscore_filter(self, series: pd.Series, threshold: float = 3.0) -> pd.Series:
        """Filter outliers using Z-score method."""
        z_scores = np.abs((series - series.mean()) / series.std())
        return series.where(z_scores <= threshold, series.median())
    
    def _scale_features(
        self,
        data: pd.DataFrame,
        panel_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Scale features in the data.
        
        Args:
            data: DataFrame to scale
            panel_type: Type of lab panel for panel-specific scaling
            
        Returns:
            Scaled DataFrame
        """
        scaling_config = self.config.get("preprocessing", {}).get("tabular", {}).get("scaling_method", "standard")
        
        if scaling_config == "none":
            return data
        
        self.logger.info(f"Scaling features using {scaling_config} method")
        
        # Only scale numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            self.logger.warning("No numeric columns found for scaling")
            return data
        
        # Initialize scaler
        if scaling_config == "standard":
            self.scaler = StandardScaler()
        elif scaling_config == "minmax":
            self.scaler = MinMaxScaler()
        elif scaling_config == "robust":
            self.scaler = RobustScaler()
        else:
            self.logger.warning(f"Unknown scaling method: {scaling_config}")
            return data
        
        # Fit and transform
        data[numeric_cols] = self.scaler.fit_transform(data[numeric_cols])
        
        self.logger.info(f"Scaled {len(numeric_cols)} numeric columns")
        return data
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform scaled data back to original scale.
        
        Args:
            data: Scaled DataFrame
            
        Returns:
            DataFrame in original scale
        """
        if self.scaler is None:
            self.logger.warning("No scaler found, returning data as-is")
            return data
        
        # Only inverse transform numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return data
        
        data[numeric_cols] = self.scaler.inverse_transform(data[numeric_cols])
        
        self.logger.info("Inverse transformed scaled data")
        return data
    
    def get_data_quality_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a data quality report.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Data quality report dictionary
        """
        report = {
            "total_records": len(data),
            "total_columns": len(data.columns),
            "missing_values": data.isnull().sum().to_dict(),
            "missing_percentage": (data.isnull().sum() / len(data) * 100).to_dict(),
            "data_types": data.dtypes.to_dict(),
            "memory_usage": data.memory_usage(deep=True).sum(),
            "duplicate_records": data.duplicated().sum(),
            "numeric_columns": data.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": data.select_dtypes(include=['object', 'category']).columns.tolist(),
        }
        
        # Add statistics for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            report["numeric_statistics"] = data[numeric_cols].describe().to_dict()
            
            # Check for potential outliers using IQR method
            outlier_counts = {}
            for col in numeric_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                outlier_counts[col] = outliers
            
            report["outlier_counts"] = outlier_counts
        
        # Add value counts for categorical columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            report["categorical_value_counts"] = {}
            for col in categorical_cols:
                report["categorical_value_counts"][col] = data[col].value_counts().to_dict()
        
        return report
    
    def validate_data_quality(
        self,
        data: pd.DataFrame,
        max_missing_percentage: float = 0.5,
        min_samples_per_class: int = 10
    ) -> Dict[str, Any]:
        """
        Validate data quality against specified thresholds.
        
        Args:
            data: DataFrame to validate
            max_missing_percentage: Maximum allowed missing percentage
            min_samples_per_class: Minimum samples per class for categorical variables
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check missing values
        missing_percentages = data.isnull().sum() / len(data) * 100
        high_missing_cols = missing_percentages[missing_percentages > max_missing_percentage * 100]
        
        if len(high_missing_cols) > 0:
            validation_results["errors"].append(
                f"Columns with high missing values: {high_missing_cols.to_dict()}"
            )
            validation_results["is_valid"] = False
        
        # Check categorical class distribution
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            value_counts = data[col].value_counts()
            small_classes = value_counts[value_counts < min_samples_per_class]
            
            if len(small_classes) > 0:
                validation_results["warnings"].append(
                    f"Column {col} has classes with < {min_samples_per_class} samples: {small_classes.to_dict()}"
                )
        
        # Check for constant columns
        constant_cols = data.columns[data.nunique() <= 1].tolist()
        if constant_cols:
            validation_results["warnings"].append(f"Constant columns found: {constant_cols}")
        
        return validation_results
