"""Tabular data loader for lab panels."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import yaml
from pydantic import ValidationError

from ..schemas.tabular_schema import LabPanelSchema
from ..utils.io import load_data, save_data
from ..utils.log import get_logger


class TabularLoader:
    """Loader for tabular lab data with validation and preprocessing."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the tabular loader.
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = get_logger(__name__)
        self.config = self._load_config(config_path)
        self.schemas = self._load_schemas()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file."""
        if config_path is None:
            config_path = "configs/data/tabular.yaml"
        
        try:
            return load_data(config_path, file_type="yaml")
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            return {}
    
    def _load_schemas(self) -> Dict[str, Any]:
        """Load data schemas."""
        return self.config.get("schemas", {})
    
    def load_panel_data(
        self,
        file_path: Union[str, Path],
        panel_type: str,
        file_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load data for a specific lab panel.
        
        Args:
            file_path: Path to the data file
            panel_type: Type of lab panel (cbc, cmp, lipid, thyroid, coag, vitamins_iron)
            file_type: Type of file (csv, parquet, json)
            
        Returns:
            Loaded DataFrame
        """
        file_path = Path(file_path)
        
        if file_type is None:
            file_type = file_path.suffix.lower().lstrip('.')
        
        self.logger.info(f"Loading {panel_type} data from {file_path}")
        
        try:
            # Load raw data
            data = load_data(file_path, file_type=file_type)
            
            if isinstance(data, dict):
                # Convert dict to DataFrame
                data = pd.DataFrame(data)
            elif not isinstance(data, pd.DataFrame):
                raise ValueError(f"Unsupported data type: {type(data)}")
            
            # Validate data
            validated_data = self._validate_panel_data(data, panel_type)
            
            self.logger.info(f"Successfully loaded {len(validated_data)} records for {panel_type}")
            return validated_data
            
        except Exception as e:
            self.logger.error(f"Failed to load {panel_type} data: {e}")
            raise
    
    def _validate_panel_data(self, data: pd.DataFrame, panel_type: str) -> pd.DataFrame:
        """
        Validate panel data against schema.
        
        Args:
            data: Raw data DataFrame
            panel_type: Type of lab panel
            
        Returns:
            Validated DataFrame
        """
        schema_config = self.schemas.get(panel_type, {})
        required_fields = schema_config.get("required_fields", [])
        optional_fields = schema_config.get("optional_fields", [])
        data_types = schema_config.get("data_types", {})
        
        # Check required fields
        missing_fields = [field for field in required_fields if field not in data.columns]
        if missing_fields:
            self.logger.warning(f"Missing required fields for {panel_type}: {missing_fields}")
        
        # Convert data types
        for field, dtype in data_types.items():
            if field in data.columns:
                try:
                    data[field] = data[field].astype(dtype)
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Failed to convert {field} to {dtype}: {e}")
        
        # Validate using Pydantic schema
        validated_records = []
        validation_errors = []
        
        for idx, row in data.iterrows():
            try:
                # Create a dict with only the fields that exist in the row
                row_dict = {k: v for k, v in row.to_dict().items() if pd.notna(v)}
                
                # Validate using the appropriate schema
                if panel_type == "cbc":
                    from ..schemas.tabular_schema import CBCSchema
                    validated_record = CBCSchema(**row_dict)
                elif panel_type == "cmp":
                    from ..schemas.tabular_schema import CMPSchema
                    validated_record = CMPSchema(**row_dict)
                elif panel_type == "lipid":
                    from ..schemas.tabular_schema import LipidSchema
                    validated_record = LipidSchema(**row_dict)
                elif panel_type == "thyroid":
                    from ..schemas.tabular_schema import ThyroidSchema
                    validated_record = ThyroidSchema(**row_dict)
                elif panel_type == "coag":
                    from ..schemas.tabular_schema import CoagSchema
                    validated_record = CoagSchema(**row_dict)
                elif panel_type == "vitamins_iron":
                    from ..schemas.tabular_schema import VitaminsIronSchema
                    validated_record = VitaminsIronSchema(**row_dict)
                else:
                    self.logger.warning(f"Unknown panel type: {panel_type}")
                    validated_record = row_dict
                
                validated_records.append(validated_record.dict() if hasattr(validated_record, 'dict') else validated_record)
                
            except ValidationError as e:
                validation_errors.append((idx, str(e)))
                self.logger.warning(f"Validation error for row {idx}: {e}")
            except Exception as e:
                validation_errors.append((idx, str(e)))
                self.logger.warning(f"Unexpected error for row {idx}: {e}")
        
        if validation_errors:
            self.logger.warning(f"Found {len(validation_errors)} validation errors")
        
        # Convert validated records back to DataFrame
        if validated_records:
            validated_data = pd.DataFrame(validated_records)
        else:
            validated_data = data.copy()
        
        return validated_data
    
    def load_combined_data(
        self,
        data_dir: Union[str, Path],
        panel_types: List[str],
        patient_id_col: str = "patient_id"
    ) -> pd.DataFrame:
        """
        Load and combine data from multiple lab panels.
        
        Args:
            data_dir: Directory containing lab panel files
            panel_types: List of panel types to load
            patient_id_col: Column name for patient ID
            
        Returns:
            Combined DataFrame
        """
        data_dir = Path(data_dir)
        combined_data = None
        
        for panel_type in panel_types:
            panel_files = list(data_dir.glob(f"*{panel_type}*"))
            
            if not panel_files:
                self.logger.warning(f"No files found for panel type: {panel_type}")
                continue
            
            for file_path in panel_files:
                try:
                    panel_data = self.load_panel_data(file_path, panel_type)
                    
                    if combined_data is None:
                        combined_data = panel_data
                    else:
                        # Merge on patient ID if available
                        if patient_id_col in panel_data.columns and patient_id_col in combined_data.columns:
                            combined_data = combined_data.merge(
                                panel_data, 
                                on=patient_id_col, 
                                how="outer",
                                suffixes=("", f"_{panel_type}")
                            )
                        else:
                            # Concatenate if no patient ID
                            combined_data = pd.concat([combined_data, panel_data], axis=1)
                    
                except Exception as e:
                    self.logger.error(f"Failed to load {panel_type} from {file_path}: {e}")
                    continue
        
        if combined_data is None:
            raise ValueError("No data could be loaded from any panel type")
        
        self.logger.info(f"Successfully combined data from {panel_types}")
        return combined_data
    
    def save_processed_data(
        self,
        data: pd.DataFrame,
        output_path: Union[str, Path],
        file_type: str = "parquet"
    ) -> None:
        """
        Save processed data to file.
        
        Args:
            data: Processed DataFrame
            output_path: Path to save the data
            file_type: Type of file to save
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            save_data(data, output_path, file_type=file_type)
            self.logger.info(f"Saved processed data to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save data to {output_path}: {e}")
            raise
    
    def load_metadata(self, metadata_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load metadata for the dataset.
        
        Args:
            metadata_path: Path to metadata file
            
        Returns:
            Metadata dictionary
        """
        try:
            return load_data(metadata_path, file_type="json")
        except Exception as e:
            self.logger.error(f"Failed to load metadata from {metadata_path}: {e}")
            return {}
    
    def save_metadata(
        self,
        metadata: Dict[str, Any],
        output_path: Union[str, Path]
    ) -> None:
        """
        Save metadata to file.
        
        Args:
            metadata: Metadata dictionary
            output_path: Path to save metadata
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            save_data(metadata, output_path, file_type="json")
            self.logger.info(f"Saved metadata to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save metadata to {output_path}: {e}")
            raise
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for the data.
        
        Args:
            data: DataFrame to summarize
            
        Returns:
            Summary dictionary
        """
        summary = {
            "total_records": len(data),
            "total_columns": len(data.columns),
            "missing_values": data.isnull().sum().to_dict(),
            "data_types": data.dtypes.to_dict(),
            "memory_usage": data.memory_usage(deep=True).sum(),
        }
        
        # Add basic statistics for numeric columns
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary["numeric_summary"] = data[numeric_cols].describe().to_dict()
        
        # Add value counts for categorical columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            summary["categorical_summary"] = {}
            for col in categorical_cols:
                summary["categorical_summary"][col] = data[col].value_counts().to_dict()
        
        return summary
