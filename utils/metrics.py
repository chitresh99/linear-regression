import numpy as np
from typing import Tuple


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # Handle edge case where all y_true are the same
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1 - (ss_res / ss_tot)


def adjusted_r2_score(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    n_features: int
) -> float:
    y_true = np.asarray(y_true)
    n_samples = len(y_true)
    r2 = r2_score(y_true, y_pred)
    
    if n_samples <= n_features + 1:
        return 0.0
    
    return 1 - ((1 - r2) * (n_samples - 1)) / (n_samples - n_features - 1)


def compute_all_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    n_features: int
) -> dict:
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'adjusted_r2': adjusted_r2_score(y_true, y_pred, n_features)
    }


def print_metrics(metrics: dict, prefix: str = "") -> None:
    label = f"{prefix}: " if prefix else ""
    print(f"\\n{label}Regression Metrics:")
    print(f"  MSE:            {metrics['mse']:.6f}")
    print(f"  RMSE:           {metrics['rmse']:.6f}")
    print(f"  MAE:            {metrics['mae']:.6f}")
    print(f"  R² Score:       {metrics['r2']:.6f}")
    print(f"  Adjusted R²:    {metrics['adjusted_r2']:.6f}")
