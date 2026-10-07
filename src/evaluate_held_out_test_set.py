"""
Single-Pass Held-Out Test Set Evaluation Module
Macular Hole Visual Outcome Prediction Pipeline

Evaluates the final models trained on the entire 104-patient Development Cohort 
against the locked, untouched 17-patient Held-Out Test Cohort (IDs 104-120).
Runs strictly ONCE with zero post-hoc hyperparameter tuning.
"""

import os
import json
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
from sklearn.ensemble import (
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
)
from src.data_preprocessing import load_and_clean_data
from src.extract_morphometry import extract_all_morphometry

CLINICAL_COLS = ['age', 'sex_encoded', 'pseudophakic', 'mh_duration', 'elevated_edge', 'mh_size', 'VA_baseline']
MORPH_COLS = ['mld_mean', 'bd_mean', 'height_mean', 'mhi_mean', 'thi_mean', 'dhi_mean']

def evaluate_held_out_test_cohort(data_path=".", pca_components=8):
    print("="*80)
    print("EXECUTING LOCKED SINGLE-PASS HELD-OUT TEST SET EVALUATION (17 PATIENTS)")
    print("="*80)
    
    # 1. Load full dataset and partition strictly into 104 Dev vs 17 Held-out Test
    df = load_and_clean_data(data_path)
    
    dev_mask = df['original_split'] != 'test'
    df_dev = df[dev_mask].copy().reset_index(drop=True)
    df_test = df[~dev_mask].copy().reset_index(drop=True)
    
    print(f"Development Cohort Size: {len(df_dev)} patients (IDs {df_dev['id'].min()}–{df_dev['id'].max()})")
    print(f"Held-Out Test Cohort Size: {len(df_test)} patients (IDs {df_test['id'].min()}–{df_test['id'].max()})")
    print(f"Test Class Balance (Gain >= 15 letters): {df_test['VA_gain_6mo_binary'].sum()} / {len(df_test)} ({df_test['VA_gain_6mo_binary'].mean()*100:.1f}%)")
    print(f"Test Delta VA Summary: Mean={df_test['delta_VA_6months'].mean():.2f}, Std={df_test['delta_VA_6months'].std(ddof=1):.2f}, Median={df_test['delta_VA_6months'].median():.2f}")
    
    # 2. Load Morphometry Features
    morph_csv = "data/morphometry_features.csv"
    if os.path.exists(morph_csv):
        morph_df = pd.read_csv(morph_csv)
    else:
        morph_df = extract_all_morphometry(df, morph_csv)
        
    df_dev = df_dev.merge(morph_df, on='id', how='left')
    df_test = df_test.merge(morph_df, on='id', how='left')
    
    # 3. Load Siamese Vision Embeddings
    raw_emb_path = "data/vision_embeddings_raw.npy"
    raw_embs = np.load(raw_emb_path)
    pid_arr = np.load("data/embedding_patient_ids.npy")
    pid_to_emb_idx = {pid: i for i, pid in enumerate(pid_arr)}
    
    dev_emb_indices = [pid_to_emb_idx[pid] for pid in df_dev['id']]
    test_emb_indices = [pid_to_emb_idx[pid] for pid in df_test['id']]
    
    dev_raw_embs = raw_embs[dev_emb_indices]
    test_raw_embs = raw_embs[test_emb_indices]
    
    # 4. Strict Preprocessing: Fit Scaler & PCA ONLY on Dev Cohort
    pca = PCA(n_components=pca_components, random_state=42)
    scaler_emb = StandardScaler()
    
    dev_emb_scaled = scaler_emb.fit_transform(dev_raw_embs)
    test_emb_scaled = scaler_emb.transform(test_raw_embs)
    
    dev_pca = pca.fit_transform(dev_emb_scaled)
    test_pca = pca.transform(test_emb_scaled)
    
    dev_clin = df_dev[CLINICAL_COLS].values
    test_clin = df_test[CLINICAL_COLS].values
    
    dev_morph = df_dev[MORPH_COLS].values
    test_morph = df_test[MORPH_COLS].values
    
    # Targets
    y_dev_clf = df_dev['VA_gain_6mo_binary'].values
    y_test_clf = df_test['VA_gain_6mo_binary'].values
    
    y_dev_reg = df_dev['delta_VA_6months'].values
    y_test_reg = df_test['delta_VA_6months'].values
    
    imputer = SimpleImputer(strategy='median')
    scaler_tab = StandardScaler()
    
    experiments = {
        "E1: Baseline VA only": (
            df_dev[['VA_baseline']].values,
            df_test[['VA_baseline']].values
        ),
        "E2: Clinical Only (7 features)": (
            dev_clin,
            test_clin
        ),
        "E3: Vision Embeddings Only": (
            dev_pca,
            test_pca
        ),
        "E4: Morphometry Only": (
            dev_morph,
            test_morph
        ),
        "E5: Clinical + Morphometry": (
            np.hstack([dev_clin, dev_morph]),
            np.hstack([test_clin, test_morph])
        ),
        "E6: Clinical + Vision Embeddings": (
            np.hstack([dev_clin, dev_pca]),
            np.hstack([test_clin, test_pca])
        ),
        "E7: Full Multimodal Fusion": (
            np.hstack([dev_clin, dev_morph, dev_pca]),
            np.hstack([test_clin, test_morph, test_pca])
        )
    }
    
    test_evaluation_results = []
    
    for exp_name, (X_tr, X_te) in experiments.items():
        # Preprocessing for linear model
        X_tr_imp = scaler_tab.fit_transform(imputer.fit_transform(X_tr))
        X_te_imp = scaler_tab.transform(imputer.transform(X_te))
        
        # Classification Training
        clf_gbdt = HistGradientBoostingClassifier(
            max_depth=2, max_iter=35, learning_rate=0.05, min_samples_leaf=4, l2_regularization=1.5, random_state=42
        )
        clf_gbdt.fit(X_tr, y_dev_clf)
        prob_gbdt = clf_gbdt.predict_proba(X_te)[:, 1]
        
        clf_lin = LogisticRegression(C=0.5, random_state=42)
        clf_lin.fit(X_tr_imp, y_dev_clf)
        prob_lin = clf_lin.predict_proba(X_te_imp)[:, 1]
        
        test_prob = 0.5 * prob_gbdt + 0.5 * prob_lin
        test_pred_bin = (test_prob >= 0.5).astype(int)
        
        # Regression Training
        reg_gbdt = HistGradientBoostingRegressor(
            max_depth=2, max_iter=35, learning_rate=0.05, min_samples_leaf=4, l2_regularization=1.5, random_state=42
        )
        reg_gbdt.fit(X_tr, y_dev_reg)
        pred_reg_gbdt = reg_gbdt.predict(X_te)
        
        reg_lin = Ridge(alpha=2.0, random_state=42)
        reg_lin.fit(X_tr_imp, y_dev_reg)
        pred_reg_lin = reg_lin.predict(X_te_imp)
        
        test_pred_reg = 0.5 * pred_reg_gbdt + 0.5 * pred_reg_lin
        
        # In-sample Dev predictions for Conformal Calibration
        dev_pred_gbdt = reg_gbdt.predict(X_tr)
        dev_pred_lin = reg_lin.predict(X_tr_imp)
        dev_pred_reg = 0.5 * dev_pred_gbdt + 0.5 * dev_pred_lin
        
        dev_residuals = np.abs(y_dev_reg - dev_pred_reg)
        # Conformal quantile at 1 - alpha = 0.90
        n_dev = len(df_dev)
        p_val_quantile = np.ceil((n_dev + 1) * 0.90) / n_dev
        p_val_quantile = min(1.0, p_val_quantile)
        q_hat = np.quantile(dev_residuals, p_val_quantile)
        
        # Evaluate Conformal Coverage on Held-Out Test Set
        lower_bound = test_pred_reg - q_hat
        upper_bound = test_pred_reg + q_hat
        covered = (y_test_reg >= lower_bound) & (y_test_reg <= upper_bound)
        test_picp = np.mean(covered) * 100.0
        
        # Compute Metrics
        auroc = roc_auc_score(y_test_clf, test_prob) * 100.0
        f1 = f1_score(y_test_clf, test_pred_bin, zero_division=0) * 100.0
        acc = accuracy_score(y_test_clf, test_pred_bin) * 100.0
        brier = brier_score_loss(y_test_clf, test_prob)
        
        mae = mean_absolute_error(y_test_reg, test_pred_reg)
        rmse = np.sqrt(mean_squared_error(y_test_reg, test_pred_reg))
        r2 = r2_score(y_test_reg, test_pred_reg)
        
        test_evaluation_results.append({
            'Model Configuration': exp_name,
            'Test AUROC (%)': round(auroc, 1),
            'Test F1 (%)': round(f1, 1),
            'Test Accuracy (%)': round(acc, 1),
            'Test Brier Loss': round(brier, 4),
            'Test MAE (letters)': round(mae, 2),
            'Test RMSE (letters)': round(rmse, 2),
            'Test R² Score': round(r2, 3),
            'Conformal q_hat': round(q_hat, 2),
            'Test 90% PICP (%)': f"{int(np.sum(covered))}/{len(y_test_reg)} ({test_picp:.1f}%)"
        })
        
    res_df = pd.DataFrame(test_evaluation_results)
    print("\n" + "="*95)
    print("FINAL HELD-OUT TEST SET EVALUATION TABLE (17 UNTOUCHED PATIENTS, IDs 104–120)")
    print("="*95)
    print(res_df.to_string(index=False))
    
    os.makedirs("results", exist_ok=True)
    res_df.to_csv("results/held_out_test_results.csv", index=False)
    with open("results/held_out_test_results.json", "w") as f:
        json.dump(test_evaluation_results, f, indent=2)
        
    print("\nSaved held-out test results to results/held_out_test_results.csv and .json")
    return res_df

if __name__ == "__main__":
    evaluate_held_out_test_cohort()
