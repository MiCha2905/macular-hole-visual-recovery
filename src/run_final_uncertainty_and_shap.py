"""
Final Uncertainty Quantification, Error Stratification & SHAP Explainability Engine
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, roc_auc_score, mean_absolute_error, r2_score
from src.data_preprocessing import load_and_clean_data, generate_repeated_cv_splits
from src.explainability_and_calibration import calibrate_cross_conformal_intervals, run_error_stratification

CLINICAL_COLS = ['age', 'sex_encoded', 'pseudophakic', 'mh_duration', 'elevated_edge', 'mh_size', 'VA_baseline']
MORPH_COLS = ['mld_mean', 'bd_mean', 'height_mean', 'mhi_mean', 'thi_mean', 'dhi_mean']

def run_uncertainty_and_explainability(data_path=".", pca_components=8):
    print("="*75)
    print("RUNNING CROSS-CONFORMAL UNCERTAINTY QUANTIFICATION & CLINICAL STRATIFICATION")
    print("="*75)
    
    df = load_and_clean_data(data_path)
    df_dev, df_test, folds = generate_repeated_cv_splits(df, n_splits=5, n_repeats=5, random_state=42)
    
    morph_df = pd.read_csv("data/morphometry_features.csv")
    df_dev = df_dev.merge(morph_df, on='id', how='left')
    
    raw_embs = np.load("data/vision_embeddings_raw.npy")
    pid_arr = np.load("data/embedding_patient_ids.npy")
    pid_to_emb_idx = {pid: i for i, pid in enumerate(pid_arr)}
    dev_raw_embs = raw_embs[[pid_to_emb_idx[pid] for pid in df_dev['id']]]
    
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    all_oof_y_true_clf = []
    all_oof_y_pred_prob = []
    all_oof_y_true_reg = []
    all_oof_y_pred_reg = []
    all_oof_pids = []
    
    fold_conformal_data = []
    
    for fold_info in folds:
        run_idx = fold_info['run_idx']
        tr_idx = fold_info['train_indices']
        val_idx = fold_info['val_indices']
        
        y_tr_clf = df_dev.iloc[tr_idx]['VA_gain_6mo_binary'].values
        y_val_clf = df_dev.iloc[val_idx]['VA_gain_6mo_binary'].values
        y_tr_reg = df_dev.iloc[tr_idx]['delta_VA_6months'].values
        y_val_reg = df_dev.iloc[val_idx]['delta_VA_6months'].values
        
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
        
        X_tr = np.hstack([tr_clin, tr_morph, tr_pca])
        X_val = np.hstack([val_clin, val_morph, val_pca])
        
        X_tr_imp = scaler.fit_transform(imputer.fit_transform(X_tr))
        X_val_imp = scaler.transform(imputer.transform(X_val))
        
        # Classifier Ensemble
        clf_gbdt = HistGradientBoostingClassifier(max_depth=2, max_iter=35, learning_rate=0.05, random_state=42+run_idx)
        clf_gbdt.fit(X_tr, y_tr_clf)
        clf_lin = LogisticRegression(C=0.5, penalty='l2', random_state=42+run_idx)
        clf_lin.fit(X_tr_imp, y_tr_clf)
        
        prob = 0.5 * clf_gbdt.predict_proba(X_val)[:, 1] + 0.5 * clf_lin.predict_proba(X_val_imp)[:, 1]
        
        # Regressor Ensemble
        reg_gbdt = HistGradientBoostingRegressor(max_depth=2, max_iter=35, learning_rate=0.05, random_state=42+run_idx)
        reg_gbdt.fit(X_tr, y_tr_reg)
        reg_lin = Ridge(alpha=2.0, random_state=42+run_idx)
        reg_lin.fit(X_tr_imp, y_tr_reg)
        
        pred_reg_tr = 0.5 * reg_gbdt.predict(X_tr) + 0.5 * reg_lin.predict(X_tr_imp)
        pred_reg_val = 0.5 * reg_gbdt.predict(X_val) + 0.5 * reg_lin.predict(X_val_imp)
        
        fold_conformal_data.append({
            'y_tr': y_tr_reg,
            'y_tr_pred': pred_reg_tr,
            'y_val': y_val_reg,
            'y_val_pred': pred_reg_val
        })
        
        all_oof_y_true_clf.extend(y_val_clf)
        all_oof_y_pred_prob.extend(prob)
        all_oof_y_true_reg.extend(y_val_reg)
        all_oof_y_pred_reg.extend(pred_reg_val)
        all_oof_pids.extend(df_dev.iloc[val_idx]['id'].values)
        
    y_true_clf = np.array(all_oof_y_true_clf)
    y_pred_prob = np.array(all_oof_y_pred_prob)
    y_true_reg = np.array(all_oof_y_true_reg)
    y_pred_reg = np.array(all_oof_y_pred_reg)
    
    # 1. Rigorous Cross-Conformal Prediction Evaluation
    conformal_summary = calibrate_cross_conformal_intervals(fold_conformal_data, alpha=0.10)
    print("\n--- 1. CROSS-CONFORMAL PREDICTION (OUT-OF-FOLD UNCERTAINTY) ---")
    print(f"Target Coverage: 90.0%")
    print(f"Empirical Validation Coverage (PICP): {conformal_summary['mean_picp_%']:.2f} ± {conformal_summary['std_picp_%']:.2f}%")
    print(f"Calibrated Prediction Interval Radius (q_hat): ±{conformal_summary['mean_q_hat_letters']:.2f} ± {conformal_summary['std_q_hat_letters']:.2f} letters")
    print(f"Mean Prediction Interval Width (MPIW): {conformal_summary['mean_mpiw_letters']:.2f} ± {conformal_summary['std_mpiw_letters']:.2f} letters")
    print(f"Sample fold-by-fold validation coverages (PICP %): {conformal_summary['raw_fold_picps'][:5]}...")
    
    # 2. Probability Calibration for Classification
    raw_brier = brier_score_loss(y_true_clf, y_pred_prob)
    print("\n--- 2. PROBABILITY CALIBRATION (CLASSIFICATION) ---")
    print(f"Pooled AUROC: {roc_auc_score(y_true_clf, y_pred_prob)*100:.2f}%")
    print(f"Overall Brier Calibration Score: {raw_brier:.4f}")
    
    # 3. Subgroup Error Stratification
    df_eval = pd.DataFrame({
        'id': all_oof_pids,
        'y_true_reg': y_true_reg,
        'y_pred_reg': y_pred_reg,
        'abs_error': np.abs(y_true_reg - y_pred_reg)
    })
    df_eval = df_eval.merge(df_dev[['id', 'mh_size', 'VA_baseline', 'mh_duration']].drop_duplicates(), on='id', how='left')
    
    print("\n--- 3. SUBGROUP ERROR STRATIFICATION ---")
    print("A. By Baseline Visual Acuity Tercile:")
    print(run_error_stratification(df_eval.copy(), 'VA_baseline', 'abs_error'))
    
    print("\nB. By Macular Hole Size Tercile:")
    print(run_error_stratification(df_eval.copy(), 'mh_size', 'abs_error'))
    
    # Save results
    uncertainty_report = {
        'conformal_target_coverage_%': 90.0,
        'conformal_out_of_fold_empirical_coverage_PICP_%': f"{conformal_summary['mean_picp_%']:.2f} ± {conformal_summary['std_picp_%']:.2f}",
        'conformal_mean_interval_radius_q_hat_letters': f"{conformal_summary['mean_q_hat_letters']:.2f} ± {conformal_summary['std_q_hat_letters']:.2f}",
        'conformal_mean_total_width_MPIW_letters': f"{conformal_summary['mean_mpiw_letters']:.2f} ± {conformal_summary['std_mpiw_letters']:.2f}",
        'raw_fold_picps_%': conformal_summary['raw_fold_picps'],
        'pooled_brier_score': round(raw_brier, 4),
        'pooled_auroc_%': round(roc_auc_score(y_true_clf, y_pred_prob)*100, 2)
    }
    
    with open("results/uncertainty_metrics.json", "w") as f:
        json.dump(uncertainty_report, f, indent=4)
        
    print("\nSaved uncertainty metrics to results/uncertainty_metrics.json")
    return uncertainty_report

if __name__ == "__main__":
    run_uncertainty_and_explainability()
