"""
Track 0: Clinical-Only Baseline Models
Evaluates HistGradientBoosting (GBDT with native NaN handling), Random Forest, and Linear/Logistic Baselines on 25 Repeated Stratified CV Folds.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, brier_score_loss,
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.data_preprocessing import load_and_clean_data, generate_repeated_cv_splits

# 7 strictly pre-operative features
FEATURE_COLS = [
    'age',
    'sex_encoded',
    'pseudophakic',
    'mh_duration',
    'elevated_edge',
    'mh_size',
    'VA_baseline'
]

def train_and_eval_clinical_baselines(data_path="."):
    df = load_and_clean_data(data_path)
    df_dev, df_test, folds = generate_repeated_cv_splits(df, n_splits=5, n_repeats=5, random_state=42)
    
    print(f"Development set size: {len(df_dev)} patients")
    print(f"Held-out test set size: {len(df_test)} patients")
    print(f"Number of evaluation folds: {len(folds)} (5 repeats x 5 folds)")
    
    models = ['Logistic / Ridge', 'HistGBDT (Tree Baseline)', 'Random Forest']
    
    clf_results = {m: [] for m in models}
    reg_results = {m: [] for m in models}
    
    for fold_info in folds:
        run_idx = fold_info['run_idx']
        train_idx = fold_info['train_indices']
        val_idx = fold_info['val_indices']
        
        X_train = df_dev.iloc[train_idx][FEATURE_COLS].copy()
        y_train_clf = df_dev.iloc[train_idx]['VA_gain_6mo_binary'].values
        y_train_reg = df_dev.iloc[train_idx]['delta_VA_6months'].values
        
        X_val = df_dev.iloc[val_idx][FEATURE_COLS].copy()
        y_val_clf = df_dev.iloc[val_idx]['VA_gain_6mo_binary'].values
        y_val_reg = df_dev.iloc[val_idx]['delta_VA_6months'].values
        
        # --- 1. Linear / Logistic Baseline ---
        pipe_clf = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(C=0.5, penalty='l2', random_state=42))
        ])
        pipe_clf.fit(X_train, y_train_clf)
        preds_clf_prob = pipe_clf.predict_proba(X_val)[:, 1]
        preds_clf_bin = (preds_clf_prob >= 0.5).astype(int)
        
        pipe_reg = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=2.0, random_state=42))
        ])
        pipe_reg.fit(X_train, y_train_reg)
        preds_reg = pipe_reg.predict(X_val)
        
        clf_results['Logistic / Ridge'].append({
            'auroc': roc_auc_score(y_val_clf, preds_clf_prob),
            'f1': f1_score(y_val_clf, preds_clf_bin, zero_division=0),
            'accuracy': accuracy_score(y_val_clf, preds_clf_bin),
            'brier': brier_score_loss(y_val_clf, preds_clf_prob)
        })
        reg_results['Logistic / Ridge'].append({
            'r2': r2_score(y_val_reg, preds_reg),
            'mae': mean_absolute_error(y_val_reg, preds_reg),
            'rmse': np.sqrt(mean_squared_error(y_val_reg, preds_reg))
        })
        
        # --- 2. HistGradientBoosting (GBDT with native NaN handling) ---
        hgb_clf = HistGradientBoostingClassifier(
            max_depth=2,
            max_iter=40,
            learning_rate=0.05,
            min_samples_leaf=4,
            l2_regularization=1.5,
            random_state=42 + run_idx
        )
        hgb_clf.fit(X_train, y_train_clf)
        hgb_clf_probs = hgb_clf.predict_proba(X_val)[:, 1]
        hgb_clf_bin = (hgb_clf_probs >= 0.5).astype(int)
        
        hgb_reg = HistGradientBoostingRegressor(
            max_depth=2,
            max_iter=40,
            learning_rate=0.05,
            min_samples_leaf=4,
            l2_regularization=1.5,
            random_state=42 + run_idx
        )
        hgb_reg.fit(X_train, y_train_reg)
        hgb_reg_preds = hgb_reg.predict(X_val)
        
        clf_results['HistGBDT (Tree Baseline)'].append({
            'auroc': roc_auc_score(y_val_clf, hgb_clf_probs),
            'f1': f1_score(y_val_clf, hgb_clf_bin, zero_division=0),
            'accuracy': accuracy_score(y_val_clf, hgb_clf_bin),
            'brier': brier_score_loss(y_val_clf, hgb_clf_probs)
        })
        reg_results['HistGBDT (Tree Baseline)'].append({
            'r2': r2_score(y_val_reg, hgb_reg_preds),
            'mae': mean_absolute_error(y_val_reg, hgb_reg_preds),
            'rmse': np.sqrt(mean_squared_error(y_val_reg, hgb_reg_preds))
        })
        
        # --- 3. Random Forest (with Median Imputation) ---
        rf_clf_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', RandomForestClassifier(max_depth=3, n_estimators=50, min_samples_leaf=3, random_state=42 + run_idx))
        ])
        rf_clf_pipe.fit(X_train, y_train_clf)
        rf_probs = rf_clf_pipe.predict_proba(X_val)[:, 1]
        rf_bin = (rf_probs >= 0.5).astype(int)
        
        rf_reg_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', RandomForestRegressor(max_depth=3, n_estimators=50, min_samples_leaf=3, random_state=42 + run_idx))
        ])
        rf_reg_pipe.fit(X_train, y_train_reg)
        rf_preds = rf_reg_pipe.predict(X_val)
        
        clf_results['Random Forest'].append({
            'auroc': roc_auc_score(y_val_clf, rf_probs),
            'f1': f1_score(y_val_clf, rf_bin, zero_division=0),
            'accuracy': accuracy_score(y_val_clf, rf_bin),
            'brier': brier_score_loss(y_val_clf, rf_probs)
        })
        reg_results['Random Forest'].append({
            'r2': r2_score(y_val_reg, rf_preds),
            'mae': mean_absolute_error(y_val_reg, rf_preds),
            'rmse': np.sqrt(mean_squared_error(y_val_reg, rf_preds))
        })

    # Summary table across 25 folds
    summary = []
    for m in models:
        df_c = pd.DataFrame(clf_results[m])
        df_r = pd.DataFrame(reg_results[m])
        
        auroc_mean = df_c['auroc'].mean() * 100
        auroc_std = df_c['auroc'].std() * 100
        f1_mean = df_c['f1'].mean() * 100
        f1_std = df_c['f1'].std() * 100
        
        r2_mean = df_r['r2'].mean()
        r2_std = df_r['r2'].std()
        mae_mean = df_r['mae'].mean()
        mae_std = df_r['mae'].std()
        
        summary.append({
            'Model': m,
            'AUROC (%)': f"{auroc_mean:.1f} ± {auroc_std:.1f}",
            'F1 Score (%)': f"{f1_mean:.1f} ± {f1_std:.1f}",
            'R²': f"{r2_mean:.3f} ± {r2_std:.3f}",
            'MAE (letters)': f"{mae_mean:.2f} ± {mae_std:.2f}"
        })
        
    summary_df = pd.DataFrame(summary)
    print("\n" + "="*70)
    print("TRACK 0: CLINICAL-ONLY BASELINE PERFORMANCE (25 Repeated CV Folds)")
    print("="*70)
    print(summary_df.to_string(index=False))
    
    os.makedirs("results", exist_ok=True)
    summary_df.to_csv("results/clinical_baseline_summary.csv", index=False)
    
    return summary_df, clf_results, reg_results

if __name__ == "__main__":
    train_and_eval_clinical_baselines()
