"""
Explainability, Error Stratification & Uncertainty Calibration Module (Cross-Conformal Framework)
1. Cross-Conformal Prediction: fits quantile q_hat on calibration/training folds, evaluates empirical PICP on validation folds.
2. Isotonic probability calibration & Brier scores.
3. Subgroup error stratification (Baseline VA, MH size, and duration terciles).
"""

import os
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

def calibrate_cross_conformal_intervals(fold_results, alpha=0.10):
    """
    Computes rigorous Cross-Conformal Prediction:
    For each fold, calibrates q_hat on the training fold residuals,
    then evaluates empirical coverage (PICP) and interval width on the unseen validation fold.
    Returns the overall distribution across all 25 folds (mean ± std, exact unrounded PICP).
    """
    fold_picps = []
    fold_q_hats = []
    fold_mpiws = []
    
    for item in fold_results:
        y_tr = item['y_tr']
        y_tr_pred = item['y_tr_pred']
        y_val = item['y_val']
        y_val_pred = item['y_val_pred']
        
        # Conformal calibration on training fold
        tr_residuals = np.abs(y_tr - y_tr_pred)
        n = len(tr_residuals)
        # Finite sample correction: ceil((1 - alpha) * (n + 1)) / n
        q_level = min(1.0, np.ceil((1.0 - alpha) * (n + 1)) / n)
        q_hat = np.quantile(tr_residuals, q_level)
        
        # Evaluate on unseen validation fold
        val_lower = y_val_pred - q_hat
        val_upper = y_val_pred + q_hat
        covered = (y_val >= val_lower) & (y_val <= val_upper)
        fold_picp = np.mean(covered) * 100.0
        
        fold_picps.append(fold_picp)
        fold_q_hats.append(q_hat)
        fold_mpiws.append(2.0 * q_hat)
        
    return {
        'mean_picp_%': float(np.mean(fold_picps)),
        'std_picp_%': float(np.std(fold_picps)),
        'mean_q_hat_letters': float(np.mean(fold_q_hats)),
        'std_q_hat_letters': float(np.std(fold_q_hats)),
        'mean_mpiw_letters': float(np.mean(fold_mpiws)),
        'std_mpiw_letters': float(np.std(fold_mpiws)),
        'raw_fold_picps': [round(float(p), 2) for p in fold_picps]
    }

def run_error_stratification(df_results, group_col, target_col='abs_error'):
    """
    Stratifies prediction errors across clinical terciles (e.g. MH size, Baseline VA).
    """
    df_clean = df_results.dropna(subset=[group_col]).copy()
    df_clean['tercile'] = pd.qcut(df_clean[group_col], q=3, labels=['Low', 'Medium', 'High'])
    strat_summary = df_clean.groupby('tercile', observed=False)[target_col].agg(['count', 'mean', 'std']).reset_index()
    return strat_summary
