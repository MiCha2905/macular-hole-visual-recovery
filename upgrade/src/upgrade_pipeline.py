"""
================================================================================
UPGRADE PIPELINE: ADVANCED MULTIMODAL MACULAR HOLE OUTCOME PREDICTION
================================================================================
Folder: upgrade/
Author: Sonali Gupta (Thapar Institute of Engineering and Technology)
Dataset: CHU de Québec HD-OCT Macular Hole Cohort (104 Dev Patients, 25-Fold CV)

Core Innovations Tested:
1. Feature Engineering 2.0:
   - Domain-specific clinical-morphometric interaction terms:
     - ceiling_headroom = (100 - VA_baseline)
     - VA_baseline_x_MLD, VA_baseline_x_Base, Duration_x_Height
     - Geometric indices: MHI, THI, Aspect Ratio, Traction Index
2. Vision 2.0: Supervised Partial Least Squares (PLS-4 / PLS-6) vs Unsupervised PCA-8.
3. Model Zoo 2.0:
   - ElasticNet (L1 + L2 regularized linear model)
   - Shallow Constrained Gradient Boosting (depth=2, l2_reg=1.5)
   - Blended Multimodal Meta-Regressor (ElasticNet + Shallow GBDT)
4. Conformal Prediction 2.0 (Adaptive Residual-Scaled Conformal CQR):
   - Heteroscedastic uncertainty estimation providing patient-specific interval widths.
================================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from scipy.stats import wilcoxon

# Add root directory to path to import data utilities safely
sys.path.insert(0, os.path.abspath("."))
from src.data_preprocessing import load_and_clean_data, generate_repeated_cv_splits
from src.extract_morphometry import extract_all_morphometry


CLINICAL_RAW_COLS = ['age', 'sex_encoded', 'pseudophakic', 'mh_duration', 'elevated_edge', 'mh_size', 'VA_baseline']
MACRO_MORPH_COLS = ['mld_mean', 'bd_mean', 'height_mean', 'mhi_mean', 'thi_mean', 'dhi_mean']


def engineer_advanced_features(df):
    """
    Creates clinical-geometric interaction features and physiological indices.
    """
    df_feat = df.copy()
    
    # 1. ETDRS Ceiling Headroom
    df_feat['ceiling_headroom'] = 100.0 - df_feat['VA_baseline']
    
    # 2. Geometric ratios (with epsilon protection)
    eps = 1e-5
    df_feat['mhi_calc'] = df_feat['height_mean'] / (df_feat['bd_mean'] + eps)
    df_feat['thi_calc'] = df_feat['height_mean'] / (df_feat['mld_mean'] + eps)
    df_feat['aspect_ratio'] = df_feat['mld_mean'] / (df_feat['bd_mean'] + eps)
    df_feat['traction_index'] = (df_feat['height_mean'] * df_feat['mld_mean']) / (df_feat['bd_mean'] + eps)
    
    # 3. Cross-domain interactions (scaled by 100 for numerical stability)
    df_feat['va_x_mld'] = df_feat['VA_baseline'] * (df_feat['mld_mean'] / 100.0)
    df_feat['va_x_bd'] = df_feat['VA_baseline'] * (df_feat['bd_mean'] / 100.0)
    df_feat['duration_x_height'] = (df_feat['mh_duration'].fillna(df_feat['mh_duration'].median())) * (df_feat['height_mean'] / 100.0)
    df_feat['duration_x_mld'] = (df_feat['mh_duration'].fillna(df_feat['mh_duration'].median())) * (df_feat['mld_mean'] / 100.0)
    df_feat['age_x_duration'] = (df_feat['age'] / 10.0) * (df_feat['mh_duration'].fillna(df_feat['mh_duration'].median()))
    
    return df_feat


def run_upgrade_benchmarks():
    print("=" * 100)
    print("EXECUTING UPGRADE EXPERIMENTS: FEATURE 2.0 + PLS VISION + ADAPTIVE CONFORMAL CQR")
    print("=" * 100)
    
    os.makedirs("upgrade/results", exist_ok=True)
    os.makedirs("upgrade/results/figures", exist_ok=True)

    # 1. Load base data
    df = load_and_clean_data(".")
    morph_csv = "data/morphometry_features.csv"
    if os.path.exists(morph_csv):
        morph_df = pd.read_csv(morph_csv)
    else:
        morph_df = extract_all_morphometry(df, morph_csv)
    df = df.merge(morph_df, on='id', how='left')

    # 2. Engineer Advanced Features
    df = engineer_advanced_features(df)
    
    ADVANCED_CLIN_MORPH_COLS = CLINICAL_RAW_COLS + [
        'ceiling_headroom', 'mhi_calc', 'thi_calc', 'aspect_ratio', 
        'traction_index', 'va_x_mld', 'va_x_bd', 'duration_x_height'
    ]

    # 3. Load Vision Embeddings with exact patient ID mapping
    raw_emb_path = "data/vision_embeddings_raw.npy"
    raw_embs = np.load(raw_emb_path)
    pid_arr = np.load("data/embedding_patient_ids.npy")
    pid_to_emb_idx = {pid: i for i, pid in enumerate(pid_arr)}

    # 4. Generate splits (Development Set N=104, Held-Out Test Set N=17)
    dev_df, test_df, folds = generate_repeated_cv_splits(df, n_splits=5, n_repeats=5, random_state=42)
    
    dev_emb_indices = [pid_to_emb_idx[pid] for pid in dev_df['id']]
    dev_raw_embs = raw_embs[dev_emb_indices]
    
    y_reg_dev = dev_df['delta_VA_6months'].values
    y_cls_dev = dev_df['VA_gain_6mo_binary'].values

    print(f"Dev Set: {len(dev_df)} patients across 25 outer evaluation folds.")
    print(f"Held-Out Test Set: {len(test_df)} patients (strictly isolated for post-tuning evaluation).")

    # Define Upgrade Tracks to Benchmark
    tracks = {
        'Baseline_E1': {'name': 'Baseline: VA Only (Ridge)', 'type': 'baseline_va'},
        'Baseline_E2': {'name': 'Baseline: Clinical 7 (Ridge)', 'type': 'baseline_clin'},
        'Baseline_E6': {'name': 'Baseline: Clinical + Vision PCA-8 (Ridge)', 'type': 'baseline_multimodal'},
        'Upgrade_T1_Interactions': {'name': 'Upgrade T1: Clinical + Morph Interactions (ElasticNet)', 'type': 'advanced_interactions'},
        'Upgrade_T2_PLS_Vision': {'name': 'Upgrade T2: Supervised PLS-4 Vision Only', 'type': 'pls_vision'},
        'Upgrade_T3_Multimodal_PLS': {'name': 'Upgrade T3: Advanced Clin + PLS-4 Vision (ElasticNet)', 'type': 'multimodal_pls'},
        'Upgrade_T4_GBDT_Ensemble': {'name': 'Upgrade T4: Shallow HistGBDT + ElasticNet Blend', 'type': 'gbdt_blend'},
    }

    results = {k: {
        'fold_r2': [], 'fold_mae': [], 'fold_rmse': [],
        'fold_auroc': [], 'fold_f1': [], 'fold_acc': [],
        'fold_picp_constant': [], 'fold_width_constant': [],
        'fold_picp_adaptive': [], 'fold_width_adaptive': [],
    } for k in tracks}

    print("\nRunning 25 Outer Cross-Validation Folds...")
    
    for fold_idx, fold_dict in enumerate(folds):
        train_idx = np.array(fold_dict['train_indices'])
        val_idx = np.array(fold_dict['val_indices'])

        # 1. Base Imputation and Scaling for Clinical Features
        imputer = SimpleImputer(strategy='median')
        scaler_clin = StandardScaler()
        
        # Train & Val clinical raw
        X_clin_raw_train = dev_df.iloc[train_idx][CLINICAL_RAW_COLS].values
        X_clin_raw_val = dev_df.iloc[val_idx][CLINICAL_RAW_COLS].values
        X_clin_imp_train = imputer.fit_transform(X_clin_raw_train)
        X_clin_imp_val = imputer.transform(X_clin_raw_val)
        X_clin_scaled_train = scaler_clin.fit_transform(X_clin_imp_train)
        X_clin_scaled_val = scaler_clin.transform(X_clin_imp_val)

        # Baseline VA only (1 feature)
        X_va_train = dev_df.iloc[train_idx][['VA_baseline']].values
        X_va_val = dev_df.iloc[val_idx][['VA_baseline']].values
        scaler_va = StandardScaler()
        X_va_scaled_train = scaler_va.fit_transform(X_va_train)
        X_va_scaled_val = scaler_va.transform(X_va_val)

        # Advanced Engineered Features
        imputer_adv = SimpleImputer(strategy='median')
        scaler_adv = StandardScaler()
        X_adv_train = dev_df.iloc[train_idx][ADVANCED_CLIN_MORPH_COLS].values
        X_adv_val = dev_df.iloc[val_idx][ADVANCED_CLIN_MORPH_COLS].values
        X_adv_imp_train = imputer_adv.fit_transform(X_adv_train)
        X_adv_imp_val = imputer_adv.transform(X_adv_val)
        X_adv_scaled_train = scaler_adv.fit_transform(X_adv_imp_train)
        X_adv_scaled_val = scaler_adv.transform(X_adv_imp_val)

        # Vision Representations
        scaler_emb = StandardScaler()
        V_train_sc = scaler_emb.fit_transform(dev_raw_embs[train_idx])
        V_val_sc = scaler_emb.transform(dev_raw_embs[val_idx])

        # Unsupervised PCA-8
        pca = PCA(n_components=8, random_state=42)
        V_pca_train = pca.fit_transform(V_train_sc)
        V_pca_val = pca.transform(V_val_sc)

        # Supervised Partial Least Squares (PLS-4) for Regression
        pls_reg = PLSRegression(n_components=4)
        pls_reg.fit(V_train_sc, y_reg_dev[train_idx])
        V_pls_train = pls_reg.x_scores_
        V_pls_val = pls_reg.transform(V_val_sc)

        # Targets
        y_r_tr = y_reg_dev[train_idx]
        y_r_va = y_reg_dev[val_idx]
        y_c_tr = y_cls_dev[train_idx]
        y_c_va = y_cls_dev[val_idx]

        # Execute Each Track
        for tr_key, tr_info in tracks.items():
            t_type = tr_info['type']
            
            if t_type == 'baseline_va':
                m_reg = Ridge(alpha=1.0)
                m_reg.fit(X_va_scaled_train, y_r_tr)
                p_reg_train = m_reg.predict(X_va_scaled_train)
                p_reg = m_reg.predict(X_va_scaled_val)

                m_cls = LogisticRegression(C=1.0, random_state=42, max_iter=35)
                m_cls.fit(X_va_scaled_train, y_c_tr)
                p_cls = m_cls.predict_proba(X_va_scaled_val)[:, 1]

            elif t_type == 'baseline_clin':
                m_reg = Ridge(alpha=1.0)
                m_reg.fit(X_clin_scaled_train, y_r_tr)
                p_reg_train = m_reg.predict(X_clin_scaled_train)
                p_reg = m_reg.predict(X_clin_scaled_val)

                m_cls = LogisticRegression(C=1.0, random_state=42, max_iter=35)
                m_cls.fit(X_clin_scaled_train, y_c_tr)
                p_cls = m_cls.predict_proba(X_clin_scaled_val)[:, 1]

            elif t_type == 'baseline_multimodal':
                X_multi_train = np.hstack([X_clin_scaled_train, V_pca_train])
                X_multi_val = np.hstack([X_clin_scaled_val, V_pca_val])
                m_reg = Ridge(alpha=1.0)
                m_reg.fit(X_multi_train, y_r_tr)
                p_reg_train = m_reg.predict(X_multi_train)
                p_reg = m_reg.predict(X_multi_val)

                m_cls = LogisticRegression(C=1.0, random_state=42, max_iter=35)
                m_cls.fit(X_multi_train, y_c_tr)
                p_cls = m_cls.predict_proba(X_multi_val)[:, 1]

            elif t_type == 'advanced_interactions':
                m_reg = ElasticNet(alpha=0.1, l1_ratio=0.3, random_state=42, max_iter=1000)
                m_reg.fit(X_adv_scaled_train, y_r_tr)
                p_reg_train = m_reg.predict(X_adv_scaled_train)
                p_reg = m_reg.predict(X_adv_scaled_val)

                m_cls = LogisticRegression(penalty='l1', solver='saga', C=0.5, random_state=42, max_iter=200)
                m_cls.fit(X_adv_scaled_train, y_c_tr)
                p_cls = m_cls.predict_proba(X_adv_scaled_val)[:, 1]

            elif t_type == 'pls_vision':
                m_reg = Ridge(alpha=10.0)
                m_reg.fit(V_pls_train, y_r_tr)
                p_reg_train = m_reg.predict(V_pls_train)
                p_reg = m_reg.predict(V_pls_val)

                m_cls = LogisticRegression(C=0.1, random_state=42)
                m_cls.fit(V_pls_train, y_c_tr)
                p_cls = m_cls.predict_proba(V_pls_val)[:, 1]

            elif t_type == 'multimodal_pls':
                X_adv_pls_tr = np.hstack([X_adv_scaled_train, V_pls_train])
                X_adv_pls_va = np.hstack([X_adv_scaled_val, V_pls_val])
                m_reg = ElasticNet(alpha=0.1, l1_ratio=0.2, random_state=42, max_iter=1000)
                m_reg.fit(X_adv_pls_tr, y_r_tr)
                p_reg_train = m_reg.predict(X_adv_pls_tr)
                p_reg = m_reg.predict(X_adv_pls_va)

                m_cls = LogisticRegression(C=0.5, random_state=42, max_iter=200)
                m_cls.fit(X_adv_pls_tr, y_c_tr)
                p_cls = m_cls.predict_proba(X_adv_pls_va)[:, 1]

            elif t_type == 'gbdt_blend':
                X_adv_pls_tr = np.hstack([X_adv_scaled_train, V_pls_train])
                X_adv_pls_va = np.hstack([X_adv_scaled_val, V_pls_val])
                
                # Regressor Blend
                m1_reg = ElasticNet(alpha=0.1, l1_ratio=0.2, random_state=42, max_iter=1000)
                m1_reg.fit(X_adv_pls_tr, y_r_tr)
                p1_tr = m1_reg.predict(X_adv_pls_tr)
                p1_reg = m1_reg.predict(X_adv_pls_va)
                
                m2_reg = HistGradientBoostingRegressor(max_depth=2, min_samples_leaf=6, l2_regularization=1.5, random_state=42)
                m2_reg.fit(X_adv_pls_tr, y_r_tr)
                p2_tr = m2_reg.predict(X_adv_pls_tr)
                p2_reg = m2_reg.predict(X_adv_pls_va)
                
                p_reg_train = 0.70 * p1_tr + 0.30 * p2_tr
                p_reg = 0.70 * p1_reg + 0.30 * p2_reg
                
                # Classifier Blend
                m1_cls = LogisticRegression(C=0.5, random_state=42, max_iter=200)
                m1_cls.fit(X_adv_pls_tr, y_c_tr)
                p1_cls = m1_cls.predict_proba(X_adv_pls_va)[:, 1]
                
                m2_cls = HistGradientBoostingClassifier(max_depth=2, min_samples_leaf=6, l2_regularization=1.5, random_state=42)
                m2_cls.fit(X_adv_pls_tr, y_c_tr)
                p2_cls = m2_cls.predict_proba(X_adv_pls_va)[:, 1]
                
                p_cls = 0.70 * p1_cls + 0.30 * p2_cls

            # Fold Regression Metrics
            r2 = r2_score(y_r_va, p_reg)
            mae = mean_absolute_error(y_r_va, p_reg)
            rmse = np.sqrt(mean_squared_error(y_r_va, p_reg))
            results[tr_key]['fold_r2'].append(r2)
            results[tr_key]['fold_mae'].append(mae)
            results[tr_key]['fold_rmse'].append(rmse)

            # Fold Classification Metrics
            try:
                auc = roc_auc_score(y_c_va, p_cls)
            except ValueError:
                auc = 0.5
            pred_binary = (p_cls >= 0.5).astype(int)
            f1 = f1_score(y_c_va, pred_binary, zero_division=0)
            acc = accuracy_score(y_c_va, pred_binary)
            results[tr_key]['fold_auroc'].append(auc)
            results[tr_key]['fold_f1'].append(f1)
            results[tr_key]['fold_acc'].append(acc)

            # -------------------------------------------------------------
            # Conformal Prediction: Constant vs. Adaptive Residual-Scaled
            # -------------------------------------------------------------
            train_resids = np.abs(y_r_tr - p_reg_train)
            alpha = 0.10
            n_tr = len(train_resids)
            
            # Constant Conformal Radius
            q_const = np.quantile(train_resids, min(1.0, np.ceil((n_tr + 1) * (1 - alpha)) / n_tr))
            covered_const = (np.abs(y_r_va - p_reg) <= q_const)
            results[tr_key]['fold_picp_constant'].append(np.mean(covered_const) * 100.0)
            results[tr_key]['fold_width_constant'].append(2.0 * q_const)

            # Adaptive Conformal (Heteroscedastic Residual Dispersion Model)
            disp_model = Ridge(alpha=10.0)
            disp_features_tr = X_clin_scaled_train[:, [0, 5, 6]] # age, mh_size, VA_baseline
            disp_features_va = X_clin_scaled_val[:, [0, 5, 6]]
            disp_model.fit(disp_features_tr, train_resids)
            pred_scale_tr = np.clip(disp_model.predict(disp_features_tr), 2.0, 25.0)
            pred_scale_va = np.clip(disp_model.predict(disp_features_va), 2.0, 25.0)

            # Normalized non-conformity scores
            norm_scores = train_resids / pred_scale_tr
            q_adapt = np.quantile(norm_scores, min(1.0, np.ceil((n_tr + 1) * (1 - alpha)) / n_tr))
            
            # Patient-specific bounds
            covered_adapt = (np.abs(y_r_va - p_reg) <= (q_adapt * pred_scale_va))
            results[tr_key]['fold_picp_adaptive'].append(np.mean(covered_adapt) * 100.0)
            results[tr_key]['fold_width_adaptive'].append(np.mean(2.0 * q_adapt * pred_scale_va))

    # =========================================================================
    # COMPILATION & COMPARATIVE STATISTICAL ANALYSIS
    # =========================================================================
    print("\n" + "=" * 110)
    print("UPGRADE BENCHMARK RESULTS (25 REPEATED STRATIFIED CV EVALUATION FOLDS)")
    print("=" * 110)

    summary_rows = []
    for tr_key, tr_info in tracks.items():
        res = results[tr_key]
        r2_m, r2_s = np.mean(res['fold_r2']), np.std(res['fold_r2'], ddof=0)
        mae_m, mae_s = np.mean(res['fold_mae']), np.std(res['fold_mae'], ddof=0)
        auc_m, auc_s = np.mean(res['fold_auroc']) * 100.0, np.std(res['fold_auroc'], ddof=0) * 100.0
        f1_m, f1_s = np.mean(res['fold_f1']) * 100.0, np.std(res['fold_f1'], ddof=0) * 100.0
        
        picp_c_m = np.mean(res['fold_picp_constant'])
        w_c_m = np.mean(res['fold_width_constant'])
        picp_a_m = np.mean(res['fold_picp_adaptive'])
        w_a_m = np.mean(res['fold_width_adaptive'])

        summary_rows.append({
            'Track': tr_info['name'],
            'CV R²': f"{r2_m:.3f} ± {r2_s:.3f}",
            'CV MAE (let)': f"{mae_m:.2f} ± {mae_s:.2f}",
            'CV AUROC (%)': f"{auc_m:.2f} ± {auc_s:.2f}",
            'CV F1 (%)': f"{f1_m:.2f} ± {f1_s:.2f}",
            'Constant PICP (%)': f"{picp_c_m:.1f}% (±{w_c_m/2:.2f} let)",
            'Adaptive CQR PICP (%)': f"{picp_a_m:.1f}% (avg ±{w_a_m/2:.2f} let)",
            'raw_r2_m': r2_m, 'raw_mae_m': mae_m, 'raw_auc_m': auc_m,
            'raw_w_const': w_c_m, 'raw_w_adapt': w_a_m
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df[['Track', 'CV R²', 'CV MAE (let)', 'CV AUROC (%)', 'Constant PICP (%)', 'Adaptive CQR PICP (%)']].to_string(index=False))

    # Save to CSV
    summary_df.to_csv("upgrade/results/upgrade_cv_comparison.csv", index=False)

    # Statistical Pairwise Testing: Baseline E6 vs Upgrade T3 (Multimodal PLS)
    diff_r2 = np.array(results['Upgrade_T3_Multimodal_PLS']['fold_r2']) - np.array(results['Baseline_E6']['fold_r2'])
    diff_mae = np.array(results['Upgrade_T3_Multimodal_PLS']['fold_mae']) - np.array(results['Baseline_E6']['fold_mae'])
    
    w_p_r2 = wilcoxon(diff_r2).pvalue if not np.all(diff_r2 == 0) else 1.0
    w_p_mae = wilcoxon(diff_mae).pvalue if not np.all(diff_mae == 0) else 1.0

    print("\n" + "=" * 80)
    print("PAIRWISE STATISTICAL VALIDATION (UPGRADE T3 vs BASELINE E6 across 25 folds):")
    print(f"Mean Δ R² Improvement: {np.mean(diff_r2):+.4f} (Wilcoxon p = {w_p_r2:.4f})")
    print(f"Mean Δ MAE Improvement: {np.mean(diff_mae):+.4f} letters (Wilcoxon p = {w_p_mae:.4f})")
    print("=" * 80)

    # Save detailed experiment log
    with open("upgrade/results/upgrade_experiment_summary.json", "w") as f:
        json.dump({
            "summary": summary_rows,
            "wilcoxon_r2_p": float(w_p_r2),
            "wilcoxon_mae_p": float(w_p_mae)
        }, f, indent=2)

    return summary_df


if __name__ == "__main__":
    run_upgrade_benchmarks()
