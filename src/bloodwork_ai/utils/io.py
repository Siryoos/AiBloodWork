"""I/O utilities for data and model loading/saving."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import pandas as pd
import yaml
from pydantic import BaseModel


def load_data(
    file_path: Union[str, Path],
    file_type: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Load data from various file formats.
    
    Args:
        file_path: Path to the data file
        file_type: Type of file (csv, parquet, json, pickle, yaml)
        **kwargs: Additional arguments for the loader
        
    Returns:
        Loaded data
    """
    file_path = Path(file_path)
    
    if file_type is None:
        file_type = file_path.suffix.lower().lstrip('.')
    
    if file_type == 'csv':
        return pd.read_csv(file_path, **kwargs)
    elif file_type == 'parquet':
        return pd.read_parquet(file_path, **kwargs)
    elif file_type == 'json':
        with open(file_path, 'r') as f:
            return json.load(f)
    elif file_type == 'pickle':
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    elif file_type in ['yaml', 'yml']:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def save_data(
    data: Any,
    file_path: Union[str, Path],
    file_type: Optional[str] = None,
    **kwargs
) -> None:
    """
    Save data to various file formats.
    
    Args:
        data: Data to save
        file_path: Path to save the data
        file_type: Type of file (csv, parquet, json, pickle, yaml)
        **kwargs: Additional arguments for the saver
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if file_type is None:
        file_type = file_path.suffix.lower().lstrip('.')
    
    if file_type == 'csv':
        data.to_csv(file_path, index=False, **kwargs)
    elif file_type == 'parquet':
        data.to_parquet(file_path, index=False, **kwargs)
    elif file_type == 'json':
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, **kwargs)
    elif file_type == 'pickle':
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, **kwargs)
    elif file_type in ['yaml', 'yml']:
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def load_model(
    model_path: Union[str, Path],
    model_type: Optional[str] = None
) -> Any:
    """
    Load a trained model.
    
    Args:
        model_path: Path to the model file
        model_type: Type of model (sklearn, xgboost, lightgbm, pytorch, onnx)
        
    Returns:
        Loaded model
    """
    model_path = Path(model_path)
    
    if model_type is None:
        # Try to infer from file extension
        if model_path.suffix == '.pkl':
            return joblib.load(model_path)
        elif model_path.suffix == '.pth':
            import torch
            return torch.load(model_path)
        elif model_path.suffix == '.onnx':
            import onnx
            return onnx.load(model_path)
        else:
            # Default to joblib
            return joblib.load(model_path)
    
    if model_type == 'sklearn':
        return joblib.load(model_path)
    elif model_type == 'xgboost':
        import xgboost as xgb
        return xgb.Booster()
    elif model_type == 'lightgbm':
        import lightgbm as lgb
        return lgb.Booster(model_file=str(model_path))
    elif model_type == 'pytorch':
        import torch
        return torch.load(model_path)
    elif model_type == 'onnx':
        import onnx
        return onnx.load(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def save_model(
    model: Any,
    model_path: Union[str, Path],
    model_type: Optional[str] = None,
    **kwargs
) -> None:
    """
    Save a trained model.
    
    Args:
        model: Model to save
        model_path: Path to save the model
        model_type: Type of model (sklearn, xgboost, lightgbm, pytorch, onnx)
        **kwargs: Additional arguments for the saver
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    if model_type is None:
        # Try to infer from file extension
        if model_path.suffix == '.pkl':
            joblib.dump(model, model_path, **kwargs)
        elif model_path.suffix == '.pth':
            import torch
            torch.save(model, model_path, **kwargs)
        elif model_path.suffix == '.onnx':
            import onnx
            onnx.save(model, model_path, **kwargs)
        else:
            # Default to joblib
            joblib.dump(model, model_path, **kwargs)
        return
    
    if model_type == 'sklearn':
        joblib.dump(model, model_path, **kwargs)
    elif model_type == 'xgboost':
        model.save_model(str(model_path))
    elif model_type == 'lightgbm':
        model.save_model(str(model_path))
    elif model_type == 'pytorch':
        import torch
        torch.save(model, model_path, **kwargs)
    elif model_type == 'onnx':
        import onnx
        onnx.save(model, model_path, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    return load_data(config_path, file_type='yaml')


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration
    """
    save_data(config, config_path, file_type='yaml')
