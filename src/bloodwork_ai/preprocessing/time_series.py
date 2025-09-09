"""Time series processing for longitudinal lab data."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..utils.log import get_logger


class TimeSeriesProcessor:
    """Time series processing for longitudinal lab data."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the time series processor.
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = get_logger(__name__)
        self.config = self._load_config(config_path)
        self.scaler = StandardScaler()
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
    
    def process_longitudinal_data(
        self,
        data: pd.DataFrame,
        patient_id_col: str = "patient_id",
        date_col: str = "date",
        value_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Process longitudinal lab data to extract time series features.
        
        Args:
            data: DataFrame with longitudinal data
            patient_id_col: Column name for patient ID
            date_col: Column name for date
            value_cols: List of columns to process as time series
            
        Returns:
            DataFrame with time series features
        """
        self.logger.info(f"Processing longitudinal data for {data[patient_id_col].nunique()} patients")
        
        # Convert date column to datetime
        data[date_col] = pd.to_datetime(data[date_col])
        
        # Sort by patient and date
        data = data.sort_values([patient_id_col, date_col])
        
        # Get value columns if not specified
        if value_cols is None:
            value_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            # Remove patient_id and date columns
            value_cols = [col for col in value_cols if col not in [patient_id_col, date_col]]
        
        # Process each patient
        processed_data = []
        
        for patient_id in data[patient_id_col].unique():
            patient_data = data[data[patient_id_col] == patient_id].copy()
            
            if len(patient_data) < 2:
                # Skip patients with only one measurement
                continue
            
            # Extract time series features
            ts_features = self._extract_time_series_features(
                patient_data, value_cols, date_col
            )
            
            # Add patient ID
            ts_features[patient_id_col] = patient_id
            
            processed_data.append(ts_features)
        
        if not processed_data:
            self.logger.warning("No patients with sufficient data for time series processing")
            return pd.DataFrame()
        
        # Combine all processed data
        result = pd.concat(processed_data, ignore_index=True)
        
        self.logger.info(f"Processed longitudinal data. Final shape: {result.shape}")
        return result
    
    def _extract_time_series_features(
        self,
        patient_data: pd.DataFrame,
        value_cols: List[str],
        date_col: str
    ) -> pd.DataFrame:
        """
        Extract time series features for a single patient.
        
        Args:
            patient_data: DataFrame with patient's longitudinal data
            value_cols: List of columns to process
            date_col: Column name for date
            
        Returns:
            DataFrame with time series features
        """
        features = {}
        
        # Calculate time intervals
        time_diffs = patient_data[date_col].diff().dt.days
        features["avg_time_interval"] = time_diffs.mean()
        features["std_time_interval"] = time_diffs.std()
        features["min_time_interval"] = time_diffs.min()
        features["max_time_interval"] = time_diffs.max()
        
        # Process each value column
        for col in value_cols:
            if col not in patient_data.columns:
                continue
            
            values = patient_data[col].dropna()
            
            if len(values) < 2:
                continue
            
            # Basic statistics
            features[f"{col}_mean"] = values.mean()
            features[f"{col}_std"] = values.std()
            features[f"{col}_min"] = values.min()
            features[f"{col}_max"] = values.max()
            features[f"{col}_range"] = values.max() - values.min()
            features[f"{col}_cv"] = values.std() / values.mean() if values.mean() != 0 else 0
            
            # Trend features
            features[f"{col}_slope"] = self._calculate_slope(values)
            features[f"{col}_trend"] = self._calculate_trend(values)
            features[f"{col}_acceleration"] = self._calculate_acceleration(values)
            
            # Change features
            features[f"{col}_first_last_diff"] = values.iloc[-1] - values.iloc[0]
            features[f"{col}_first_last_pct_change"] = (
                (values.iloc[-1] - values.iloc[0]) / values.iloc[0] * 100
                if values.iloc[0] != 0 else 0
            )
            
            # Volatility features
            features[f"{col}_volatility"] = self._calculate_volatility(values)
            features[f"{col}_stability"] = self._calculate_stability(values)
            
            # Peak features
            features[f"{col}_peaks"] = self._count_peaks(values)
            features[f"{col}_valleys"] = self._count_valleys(values)
            
            # Recent vs historical
            if len(values) >= 4:
                recent_values = values.iloc[-2:]
                historical_values = values.iloc[:-2]
                
                features[f"{col}_recent_mean"] = recent_values.mean()
                features[f"{col}_historical_mean"] = historical_values.mean()
                features[f"{col}_recent_vs_historical"] = (
                    recent_values.mean() - historical_values.mean()
                )
                features[f"{col}_recent_vs_historical_pct"] = (
                    (recent_values.mean() - historical_values.mean()) / historical_values.mean() * 100
                    if historical_values.mean() != 0 else 0
                )
        
        return pd.DataFrame([features])
    
    def _calculate_slope(self, values: pd.Series) -> float:
        """Calculate linear slope using least squares."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = values.values
        
        # Calculate slope using least squares
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def _calculate_trend(self, values: pd.Series) -> str:
        """Calculate trend direction."""
        if len(values) < 2:
            return "stable"
        
        slope = self._calculate_slope(values)
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_acceleration(self, values: pd.Series) -> float:
        """Calculate acceleration (second derivative)."""
        if len(values) < 3:
            return 0.0
        
        # Calculate first differences
        first_diff = values.diff().dropna()
        
        if len(first_diff) < 2:
            return 0.0
        
        # Calculate second differences
        second_diff = first_diff.diff().dropna()
        
        return second_diff.mean()
    
    def _calculate_volatility(self, values: pd.Series) -> float:
        """Calculate volatility (standard deviation of changes)."""
        if len(values) < 2:
            return 0.0
        
        changes = values.diff().dropna()
        return changes.std()
    
    def _calculate_stability(self, values: pd.Series) -> float:
        """Calculate stability (inverse of volatility)."""
        volatility = self._calculate_volatility(values)
        return 1.0 / (1.0 + volatility) if volatility > 0 else 1.0
    
    def _count_peaks(self, values: pd.Series) -> int:
        """Count peaks in the time series."""
        if len(values) < 3:
            return 0
        
        peaks = 0
        for i in range(1, len(values) - 1):
            if values.iloc[i] > values.iloc[i-1] and values.iloc[i] > values.iloc[i+1]:
                peaks += 1
        
        return peaks
    
    def _count_valleys(self, values: pd.Series) -> int:
        """Count valleys in the time series."""
        if len(values) < 3:
            return 0
        
        valleys = 0
        for i in range(1, len(values) - 1):
            if values.iloc[i] < values.iloc[i-1] and values.iloc[i] < values.iloc[i+1]:
                valleys += 1
        
        return valleys
    
    def create_rolling_features(
        self,
        data: pd.DataFrame,
        patient_id_col: str = "patient_id",
        date_col: str = "date",
        value_cols: List[str] = None,
        window_sizes: List[int] = [3, 6, 12]
    ) -> pd.DataFrame:
        """
        Create rolling window features.
        
        Args:
            data: DataFrame with longitudinal data
            patient_id_col: Column name for patient ID
            date_col: Column name for date
            value_cols: List of columns to process
            window_sizes: List of window sizes for rolling features
            
        Returns:
            DataFrame with rolling features
        """
        self.logger.info("Creating rolling window features")
        
        # Convert date column to datetime
        data[date_col] = pd.to_datetime(data[date_col])
        
        # Sort by patient and date
        data = data.sort_values([patient_id_col, date_col])
        
        # Get value columns if not specified
        if value_cols is None:
            value_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            value_cols = [col for col in value_cols if col not in [patient_id_col, date_col]]
        
        # Create rolling features for each patient
        rolling_data = []
        
        for patient_id in data[patient_id_col].unique():
            patient_data = data[data[patient_id_col] == patient_id].copy()
            
            if len(patient_data) < max(window_sizes):
                continue
            
            # Create rolling features
            for window_size in window_sizes:
                if len(patient_data) < window_size:
                    continue
                
                for col in value_cols:
                    if col not in patient_data.columns:
                        continue
                    
                    # Rolling statistics
                    rolling_mean = patient_data[col].rolling(window=window_size).mean()
                    rolling_std = patient_data[col].rolling(window=window_size).std()
                    rolling_min = patient_data[col].rolling(window=window_size).min()
                    rolling_max = patient_data[col].rolling(window=window_size).max()
                    
                    # Add rolling features
                    patient_data[f"{col}_rolling_mean_{window_size}"] = rolling_mean
                    patient_data[f"{col}_rolling_std_{window_size}"] = rolling_std
                    patient_data[f"{col}_rolling_min_{window_size}"] = rolling_min
                    patient_data[f"{col}_rolling_max_{window_size}"] = rolling_max
                    
                    # Rolling trend
                    rolling_slope = patient_data[col].rolling(window=window_size).apply(
                        lambda x: self._calculate_slope(x) if len(x) >= 2 else 0
                    )
                    patient_data[f"{col}_rolling_slope_{window_size}"] = rolling_slope
            
            rolling_data.append(patient_data)
        
        if not rolling_data:
            self.logger.warning("No patients with sufficient data for rolling features")
            return pd.DataFrame()
        
        # Combine all data
        result = pd.concat(rolling_data, ignore_index=True)
        
        self.logger.info(f"Created rolling features. Final shape: {result.shape}")
        return result
    
    def detect_anomalies(
        self,
        data: pd.DataFrame,
        patient_id_col: str = "patient_id",
        value_cols: List[str] = None,
        method: str = "zscore",
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Detect anomalies in time series data.
        
        Args:
            data: DataFrame with time series data
            patient_id_col: Column name for patient ID
            value_cols: List of columns to check for anomalies
            method: Anomaly detection method (zscore, iqr, isolation_forest)
            threshold: Threshold for anomaly detection
            
        Returns:
            DataFrame with anomaly flags
        """
        self.logger.info(f"Detecting anomalies using {method} method")
        
        # Get value columns if not specified
        if value_cols is None:
            value_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            value_cols = [col for col in value_cols if col not in [patient_id_col]]
        
        # Create anomaly flags
        anomaly_data = data.copy()
        
        for col in value_cols:
            if col not in data.columns:
                continue
            
            if method == "zscore":
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                anomaly_data[f"{col}_anomaly"] = z_scores > threshold
            elif method == "iqr":
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                anomaly_data[f"{col}_anomaly"] = (data[col] < lower_bound) | (data[col] > upper_bound)
            else:
                self.logger.warning(f"Unknown anomaly detection method: {method}")
                continue
        
        # Count anomalies per patient
        anomaly_cols = [col for col in anomaly_data.columns if col.endswith("_anomaly")]
        anomaly_data["total_anomalies"] = anomaly_data[anomaly_cols].sum(axis=1)
        
        self.logger.info(f"Anomaly detection completed. Found {anomaly_data['total_anomalies'].sum()} total anomalies")
        return anomaly_data
    
    def get_time_series_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a time series processing report.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Time series report dictionary
        """
        report = {
            "total_patients": data["patient_id"].nunique() if "patient_id" in data.columns else 0,
            "total_measurements": len(data),
            "avg_measurements_per_patient": len(data) / data["patient_id"].nunique() if "patient_id" in data.columns else 0,
            "time_series_features": [],
            "feature_statistics": {}
        }
        
        # Identify time series features
        ts_feature_patterns = [
            "_slope", "_trend", "_acceleration", "_volatility", "_stability",
            "_peaks", "_valleys", "_rolling_mean", "_rolling_std", "_rolling_slope"
        ]
        
        for col in data.columns:
            if any(pattern in col for pattern in ts_feature_patterns):
                report["time_series_features"].append(col)
        
        # Calculate statistics for time series features
        for col in report["time_series_features"]:
            if col in data.columns:
                report["feature_statistics"][col] = {
                    "mean": data[col].mean(),
                    "std": data[col].std(),
                    "min": data[col].min(),
                    "max": data[col].max(),
                    "missing_count": data[col].isnull().sum()
                }
        
        return report
