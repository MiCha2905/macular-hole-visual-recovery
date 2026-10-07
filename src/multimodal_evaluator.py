"""
Unified Multi-Modal Model Definition and Cross-Validation Engine
Serves as the single source of truth for all benchmarking, significance testing, 
diagnostic checks, and seed sensitivity analyses.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, brier_score_loss,
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from src.data_preprocessing import load_and_clean_data, generate_repeated_cv_splits
from src.extract_morphometry import extract_all_morphometry

CLINICAL_COLS = ['age', 'sex_encoded', 'pseudophakic', 'mh_duration', 'elevated_edge', 'mh_size', 'VA_baseline']
MORPH_COLS = ['mld_mean', 'bd_mean', 'height_mean', 'mhi_mean', 'thi_mean', 'dhi_mean']

EXPERIMENT_NAMES = [
    "E1: Baseline VA only",
    "E2: Clinical Only (7 features)",
    "E3: Vision Embeddings Only",
    "E4: Morphometry Only",
    "E5: Clinical + Morphometry",
    "E6: Clinical + Vision Embeddings",
    "E7: Full Multimodal Fusion"
]

def fit_predict_ensemble(X_tr, y_tr_clf, y_tr_reg, X_val, run_idx, scaler_tab, imputer):
    """
    Standardized ensemble used across the entire study:
    - 50% HistGradientBoosting + 50% L2-Regularized Linear Model (Logistic / Ridge).
    """
    # Impute and scale for linear models
    X_tr_imp = scaler_tab.fit_transform(imputer.fit_transform(X_tr))
    X_val_imp = scaler_tab.transform(imputer.transform(X_val))
    
    # Classification Models
    clf_gbdt = HistGradientBoostingClassifier(
        max_depth=2, max_iter=35, learning_rate=0.05, min_samples_leaf=4, l2_regularization=1.5, random_state=42 + run_idx
    )
    clf_gbdt.fit(X_tr, y_tr_clf)
    prob_gbdt = clf_gbdt.predict_proba(X_val)[:, 1]
    
    clf_lin = LogisticRegression(C=0.5, random_state=42 + run_idx)
    clf_lin.fit(X_tr_imp, y_tr_clf)
    prob_lin = clf_lin.predict_proba(X_val_imp)[:, 1]
    
    prob_ens = 0.5 * prob_gbdt + 0.5 * prob_lin
    pred_bin = (prob_ens >= 0.5).astype(int)
    
    # Regression Models
    reg_gbdt = HistGradientBoostingRegressor(
        max_depth=2, max_iter=35, learning_rate=0.05, min_samples_leaf=4, l2_regularization=1.5, random_state=42 + run_idx
    )
    reg_gbdt.fit(X_tr, y_tr_reg)
    pred_reg_gbdt = reg_gbdt.predict(X_val)
    
    reg_lin = Ridge(alpha=2.0, random_state=42 + run_idx)
    reg_lin.fit(X_tr_imp, y_tr_reg)
    pred_reg_lin = reg_lin.predict(X_val_imp)
    
    pred_reg_ens = 0.5 * pred_reg_gbdt + 0.5 * pred_reg_lin
    
    return prob_ens, pred_bin, pred_reg_ens

def run_single_cv_evaluation(df_dev, dev_raw_embs, folds, pca_components=8):
    """
    Executes a 25-fold CV evaluation using the exact ensemble and returns fold-level records.
    """
    fold_records = {exp: {'auroc': [], 'f1': [], 'r2': [], 'mae': [], 'rmse': [], 'brier': []} for exp in EXPERIMENT_NAMES}
    
    imputer = SimpleImputer(strategy='median')
    scaler_tab = StandardScaler()
    
    for fold_info in folds:
        run_idx = fold_info['run_idx']
        tr_idx = fold_info['train_indices']
        val_idx = fold_info['val_indices']
        
        y_tr_clf = df_dev.iloc[tr_idx]['VA_gain_6mo_binary'].values
        y_val_clf = df_dev.iloc[val_idx]['VA_gain_6mo_binary'].values
        y_tr_reg = df_dev.iloc[tr_idx]['delta_VA_6months'].values
        y_val_reg = df_dev.iloc[val_idx]['delta_VA_6months'].values
        
        # Leak-Proof Fold-Level PCA
        pca = PCA(n_components=pca_components, random_state=42)
        scaler_emb = StandardScaler()
        tr_emb_scaled = scaler_emb.fit_transform(dev_raw_embs[tr_idx])
        val_emb_scaled = scaler_emb.transform(dev_raw_embs[val_idx])
        
        tr_pca = pca.fit_transform(tr_emb_scaled)
        val_pca = pca.transform(val_emb_scaled)
        
        tr_clin = df_dev.iloc[tr_idx][CLINICAL_COLS].values
        val_clin = df_dev.iloc[val_idx][CLINICAL_COLS].values
        
        tr_morph = df_dev.iloc[tr_idx][MORPH_COLS].values
        val_morph = df_dev.iloc[val_idx][MORPH_COLS].values
        
        feat_sets = {
            "E1: Baseline VA only": (df_dev.iloc[tr_idx][['VA_baseline']].values, df_dev.iloc[val_idx][['VA_baseline']].values),
            "E2: Clinical Only (7 features)": (tr_clin, val_clin),
            "E3: Vision Embeddings Only": (tr_pca, val_pca),
            "E4: Morphometry Only": (tr_morph, val_morph),
            "E5: Clinical + Morphometry": (np.hstack([tr_clin, tr_morph]), np.hstack([val_clin, val_morph])),
            "E6: Clinical + Vision Embeddings": (np.hstack([tr_clin, tr_pca]), np.hstack([val_clin, val_pca])),
            "E7: Full Multimodal Fusion": (np.hstack([tr_clin, tr_morph, tr_pca]), np.hstack([val_clin, val_morph, val_pca]))
        }
        
        for exp_name, (X_tr, X_val) in feat_sets.items():
            prob, pred_bin, pred_reg = fit_predict_ensemble(X_tr, y_tr_clf, y_tr_reg, X_val, run_idx, scaler_tab, imputer)
            
            fold_records[exp_name]['auroc'].append(roc_auc_score(y_val_clf, prob))
            fold_records[exp_name]['f1'].append(f1_score(y_val_clf, pred_bin, zero_division=0))
            fold_records[exp_name]['brier'].append(brier_score_loss(y_val_clf, prob))
            fold_records[exp_name]['r2'].append(r2_score(y_val_reg, pred_reg))
            fold_records[exp_name]['mae'].append(mean_absolute_error(y_val_reg, pred_reg))
            fold_records[exp_name]['rmse'].append(np.sqrt(mean_squared_error(y_val_reg, pred_reg)))
            
    return fold_records
