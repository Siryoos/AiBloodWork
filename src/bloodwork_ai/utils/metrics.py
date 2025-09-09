"""Metrics calculation utilities for model evaluation."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    task_type: str = "binary"
) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for binary/multiclass)
        task_type: Type of task (binary, multiclass, regression)
        
    Returns:
        Dictionary of calculated metrics
    """
    metrics = {}
    
    if task_type == "binary":
        metrics.update(_calculate_binary_metrics(y_true, y_pred, y_prob))
    elif task_type == "multiclass":
        metrics.update(_calculate_multiclass_metrics(y_true, y_pred, y_prob))
    elif task_type == "regression":
        metrics.update(_calculate_regression_metrics(y_true, y_pred))
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    return metrics


def _calculate_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Calculate binary classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary"),
        "recall": recall_score(y_true, y_pred, average="binary"),
        "f1": f1_score(y_true, y_pred, average="binary"),
    }
    
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        metrics["pr_auc_curve"] = auc(recall, precision)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        metrics["roc_auc_curve"] = auc(fpr, tpr)
    
    return metrics


def _calculate_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Calculate multiclass classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }
    
    if y_prob is not None:
        metrics["roc_auc_ovr"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
        metrics["roc_auc_ovo"] = roc_auc_score(y_true, y_prob, multi_class="ovo")
    
    return metrics


def _calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Calculate regression metrics."""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
    }


def calculate_clinical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    thresholds: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Calculate clinically relevant metrics at different thresholds.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        thresholds: List of probability thresholds to evaluate
        
    Returns:
        Dictionary of clinical metrics
    """
    metrics = {}
    
    if y_prob is not None:
        # Calculate metrics at different thresholds
        threshold_metrics = {}
        for threshold in thresholds:
            y_pred_thresh = (y_prob >= threshold).astype(int)
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresh).ravel()
            
            # Calculate clinical metrics
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            threshold_metrics[f"threshold_{threshold}"] = {
                "sensitivity": sensitivity,
                "specificity": specificity,
                "ppv": ppv,
                "npv": npv,
                "accuracy": accuracy_score(y_true, y_pred_thresh),
            }
        
        metrics["threshold_metrics"] = threshold_metrics
    
    # Calculate overall clinical metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics["ppv"] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    return metrics


def calculate_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Calculate calibration metrics for probability predictions.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration curve
        
    Returns:
        Dictionary of calibration metrics
    """
    # Calculate calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins
    )
    
    # Calculate Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    # Calculate Maximum Calibration Error (MCE)
    mce = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
    
    return {
        "ece": ece,
        "mce": mce,
        "fraction_of_positives": fraction_of_positives.tolist(),
        "mean_predicted_value": mean_predicted_value.tolist(),
    }


def calculate_vision_metrics(
    y_true: List[Dict],
    y_pred: List[Dict],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate vision-specific metrics for object detection.
    
    Args:
        y_true: List of ground truth annotations
        y_pred: List of predicted annotations
        iou_threshold: IoU threshold for matching
        
    Returns:
        Dictionary of vision metrics
    """
    # This is a simplified implementation
    # In practice, you would use COCO evaluation tools
    
    total_gt = sum(len(gt["boxes"]) for gt in y_true)
    total_pred = sum(len(pred["boxes"]) for pred in y_pred)
    
    # Calculate basic metrics
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    
    if total_pred > 0:
        precision = total_pred / total_pred  # Simplified
    if total_gt > 0:
        recall = total_pred / total_gt  # Simplified
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_gt": total_gt,
        "total_pred": total_pred,
    }
