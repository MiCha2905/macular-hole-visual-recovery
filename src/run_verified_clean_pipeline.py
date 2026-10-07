"""
================================================================================
VERIFIED CLEAN PIPELINE: MACULAR HOLE VISUAL OUTCOME PREDICTION
================================================================================
Author: Sonali Gupta (Thapar Institute of Engineering and Technology)
Dataset: CHU de Québec HD-OCT Macular Hole Cohort (121 Benchmark Patients)

Key Scientific Safeguards:
1. Zero Data Leakage: Imputers, scalers, and PCA fit strictly on training folds.
2. 25-Fold Repeated Stratified CV (5 repeats x 5 folds) on Dev Cohort (N=104).
3. Conformal Calibration: Strictly Out-of-Fold (OOF) validation coverage across
   520 pooled predictions, computing fold-wise PICP and interval width.
4. Single-Pass Held-Out Test Set (N=17, IDs 104-120): Evaluated strictly ONCE.
5. Strictly Excludes unverified layer segmentations (no EZ/ELM claims).
================================================================================
"""

import os
import sys
import json
sys.path.insert(0, os.path.abspath("."))
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
MACRO_MORPH_COLS = ['mld_mean', 'bd_mean', 'height_mean', 'mhi_mean', 'thi_mean', 'dhi_mean']


def run_pipeline():
    print("=" * 100)
    print("EXECUTING VERIFIED CLEAN MACULAR HOLE PIPELINE (NO UNVERIFIED LAYERS, ZERO LEAKAGE)")
    print("=" * 100)

    # 1. Load cleaned dataset
    df = load_and_clean_data(".")
    print(f"Total Cohort Size: {len(df)} patients")

    # 2. Load Macro Morphometry (MLD, Base, Height, MHI, THI)
    morph_csv = "data/morphometry_features.csv"
    if os.path.exists(morph_csv):
        morph_df = pd.read_csv(morph_csv)
    else:
        morph_df = extract_all_morphometry(df, morph_csv)
    df = df.merge(morph_df, on='id', how='left')

    # 3. Load Siamese Vision Embeddings (2048-D)
    raw_emb_path = "data/vision_embeddings_raw.npy"
    if not os.path.exists(raw_emb_path):
        raise FileNotFoundError(f"Vision embeddings not found at {raw_emb_path}")
    raw_embs = np.load(raw_emb_path)
    pid_arr = np.load("data/embedding_patient_ids.npy")
    pid_to_emb_idx = {pid: i for i, pid in enumerate(pid_arr)}

    # 4. Partition Dev (104) vs Locked Test (17)
    df_dev, df_test, folds = generate_repeated_cv_splits(df, n_splits=5, n_repeats=5, random_state=42)
    print(f"Development Set: {len(df_dev)} patients (IDs {df_dev['id'].min()}–{df_dev['id'].max()})")
    print(f"Held-Out Test Set: {len(df_test)} patients (IDs {df_test['id'].min()}–{df_test['id'].max()})")
    print(f"Test Class Balance (Gain >= 15): {df_test['VA_gain_6mo_binary'].sum()} / {len(df_test)} ({df_test['VA_gain_6mo_binary'].mean()*100:.1f}%)")

    dev_emb_indices = [pid_to_emb_idx[pid] for pid in df_dev['id']]
    test_emb_indices = [pid_to_emb_idx[pid] for pid in df_test['id']]
    dev_raw_embs = raw_embs[dev_emb_indices]
    test_raw_embs = raw_embs[test_emb_indices]

    y_dev_clf = df_dev['VA_gain_6mo_binary'].values
    y_dev_reg = df_dev['delta_VA_6months'].values

    y_test_clf = df_test['VA_gain_6mo_binary'].values
    y_test_reg = df_test['delta_VA_6months'].values

    track_keys = [
        "E1: Baseline VA Only",
        "E2: Full Clinical Baseline (7 features)",
        "E3: Retinal Vision Embeddings (PCA-8)",
        "E4: Macro Morphometry (MLD, Base, Height, MHI, THI)",
        "E5: Clinical + Morphometry Fusion",
        "E6: Clinical + Vision Fusion (Multimodal)",
        "E7: Full Multimodal Fusion (Clin + Morph + Vision)"
    ]

    cv_results = {
        k: {
            'auroc': [], 'f1': [], 'acc': [], 'brier': [],
            'r2': [], 'mae': [], 'rmse': [],
            'conformal_q': [], 'conformal_picp': []
        } for k in track_keys
    }

    print("\n" + "=" * 100)
    print("1. EVALUATING 25-FOLD REPEATED STRATIFIED CV ON DEV SET (N=104)")
    print("=" * 100)

    imputer = SimpleImputer(strategy='median')
    scaler_tab = StandardScaler()

    for fold_idx, fold_info in enumerate(folds):
        run_idx = fold_info['run_idx']
        tr_idx = fold_info['train_indices']
        val_idx = fold_info['val_indices']

        y_tr_clf = y_dev_clf[tr_idx]
        y_val_clf = y_dev_clf[val_idx]
        y_tr_reg = y_dev_reg[tr_idx]
        y_val_reg = y_dev_reg[val_idx]

        # In-fold PCA for Vision (fit strictly on tr_idx)
        pca = PCA(n_components=8, random_state=42)
        scaler_emb = StandardScaler()
        tr_vis_pca = pca.fit_transform(scaler_emb.fit_transform(dev_raw_embs[tr_idx]))
        val_vis_pca = pca.transform(scaler_emb.transform(dev_raw_embs[val_idx]))

        tr_clin = df_dev.iloc[tr_idx][CLINICAL_COLS].values
        val_clin = df_dev.iloc[val_idx][CLINICAL_COLS].values

        tr_morph = df_dev.iloc[tr_idx][MACRO_MORPH_COLS].values
        val_morph = df_dev.iloc[val_idx][MACRO_MORPH_COLS].values

        tr_va = df_dev.iloc[tr_idx][['VA_baseline']].values
        val_va = df_dev.iloc[val_idx][['VA_baseline']].values

        track_data = {
            "E1: Baseline VA Only": (tr_va, val_va),
            "E2: Full Clinical Baseline (7 features)": (tr_clin, val_clin),
            "E3: Retinal Vision Embeddings (PCA-8)": (tr_vis_pca, val_vis_pca),
            "E4: Macro Morphometry (MLD, Base, Height, MHI, THI)": (tr_morph, val_morph),
            "E5: Clinical + Morphometry Fusion": (np.hstack([tr_clin, tr_morph]), np.hstack([val_clin, val_morph])),
            "E6: Clinical + Vision Fusion (Multimodal)": (np.hstack([tr_clin, tr_vis_pca]), np.hstack([val_clin, val_vis_pca])),
            "E7: Full Multimodal Fusion (Clin + Morph + Vision)": (
                np.hstack([tr_clin, tr_morph, tr_vis_pca]), np.hstack([val_clin, val_morph, val_vis_pca])
            )
        }

        for t_name, (X_tr_raw, X_val_raw) in track_data.items():
            X_tr = scaler_tab.fit_transform(imputer.fit_transform(X_tr_raw))
            X_val = scaler_tab.transform(imputer.transform(X_val_raw))

            # Classification
            clf_lr = LogisticRegression(C=0.5, max_iter=500, random_state=42 + run_idx)
            clf_lr.fit(X_tr, y_tr_clf)
            prob_lr = clf_lr.predict_proba(X_val)[:, 1]

            clf_gbdt = HistGradientBoostingClassifier(
                max_depth=2, max_iter=35, learning_rate=0.05, min_samples_leaf=4, l2_regularization=1.5, random_state=42 + run_idx
            )
            clf_gbdt.fit(X_tr_raw, y_tr_clf)
            prob_gbdt = clf_gbdt.predict_proba(X_val_raw)[:, 1]

            prob = 0.5 * prob_lr + 0.5 * prob_gbdt
            pred_bin = (prob >= 0.5).astype(int)

            # Regression
            reg_ridge = Ridge(alpha=2.0, random_state=42 + run_idx)
            reg_ridge.fit(X_tr, y_tr_reg)
            pred_ridge_val = reg_ridge.predict(X_val)
            pred_ridge_tr = reg_ridge.predict(X_tr)

            reg_gbdt = HistGradientBoostingRegressor(
                max_depth=2, max_iter=35, learning_rate=0.05, min_samples_leaf=4, l2_regularization=1.5, random_state=42 + run_idx
            )
            reg_gbdt.fit(X_tr_raw, y_tr_reg)
            pred_gbdt_val = reg_gbdt.predict(X_val_raw)
            pred_gbdt_tr = reg_gbdt.predict(X_tr_raw)

            pred_reg_val = 0.5 * pred_ridge_val + 0.5 * pred_gbdt_val
            pred_reg_tr = 0.5 * pred_ridge_tr + 0.5 * pred_gbdt_tr

            # Out-of-Fold Conformal Prediction Calibration:
            # 1. Compute training non-conformity scores
            tr_residuals = np.abs(y_tr_reg - pred_reg_tr)
            n_tr = len(tr_idx)
            p_quantile = min(1.0, np.ceil((n_tr + 1) * 0.90) / n_tr)
            q_hat_fold = np.quantile(tr_residuals, p_quantile)

            # 2. Evaluate coverage strictly Out-of-Fold on validation fold
            val_lower = pred_reg_val - q_hat_fold
            val_upper = pred_reg_val + q_hat_fold
            val_covered = (y_val_reg >= val_lower) & (y_val_reg <= val_upper)
            fold_picp = np.mean(val_covered) * 100.0

            # Store metrics
            cv_results[t_name]['auroc'].append(roc_auc_score(y_val_clf, prob))
            cv_results[t_name]['f1'].append(f1_score(y_val_clf, pred_bin, zero_division=0))
            cv_results[t_name]['acc'].append(accuracy_score(y_val_clf, pred_bin))
            cv_results[t_name]['brier'].append(brier_score_loss(y_val_clf, prob))
            cv_results[t_name]['r2'].append(r2_score(y_val_reg, pred_reg_val))
            cv_results[t_name]['mae'].append(mean_absolute_error(y_val_reg, pred_reg_val))
            cv_results[t_name]['rmse'].append(np.sqrt(mean_squared_error(y_val_reg, pred_reg_val)))
            cv_results[t_name]['conformal_q'].append(q_hat_fold)
            cv_results[t_name]['conformal_picp'].append(fold_picp)

    # Print Full 25-Fold CV Table
    print(f"{'Model Track':<48} | {'AUROC (%)':<16} | {'F1 (%)':<14} | {'Reg. R²':<14} | {'MAE (let)':<12} | {'OOF 90% PICP (%)':<18} | {'Calib. q_hat'}")
    print("-" * 138)
    cv_table = []
    for t_name in track_keys:
        auc_m = np.mean(cv_results[t_name]['auroc']) * 100
        auc_sd = np.std(cv_results[t_name]['auroc']) * 100
        f1_m = np.mean(cv_results[t_name]['f1']) * 100
        f1_sd = np.std(cv_results[t_name]['f1']) * 100
        r2_m = np.mean(cv_results[t_name]['r2'])
        r2_sd = np.std(cv_results[t_name]['r2'])
        mae_m = np.mean(cv_results[t_name]['mae'])
        mae_sd = np.std(cv_results[t_name]['mae'])
        picp_m = np.mean(cv_results[t_name]['conformal_picp'])
        picp_sd = np.std(cv_results[t_name]['conformal_picp'])
        q_m = np.mean(cv_results[t_name]['conformal_q'])
        q_sd = np.std(cv_results[t_name]['conformal_q'])

        print(f"{t_name:<48} | {auc_m:5.2f} ± {auc_sd:4.2f}% | {f1_m:5.2f} ± {f1_sd:4.2f}% | {r2_m:6.3f} ± {r2_sd:5.3f} | {mae_m:5.2f} ± {mae_sd:4.2f} | {picp_m:5.2f} ± {picp_sd:4.2f}%    | ±{q_m:5.2f} ± {q_sd:4.2f}")
        cv_table.append({
            'Model Track': t_name,
            'CV AUROC (%)': f"{auc_m:.2f} ± {auc_sd:.2f}",
            'CV F1 (%)': f"{f1_m:.2f} ± {f1_sd:.2f}",
            'CV R²': f"{r2_m:.3f} ± {r2_sd:.3f}",
            'CV MAE (letters)': f"{mae_m:.2f} ± {mae_sd:.2f}",
            'OOF Conformal PICP (%)': f"{picp_m:.2f} ± {picp_sd:.2f}",
            'Conformal Radius (q_hat)': f"±{q_m:.2f} ± {q_sd:.2f}"
        })

    # =========================================================================
    # SINGLE-PASS HELD-OUT TEST EVALUATION (N=17)
    # =========================================================================
    print("\n" + "=" * 100)
    print("2. SINGLE-PASS HELD-OUT TEST SET EVALUATION (N=17, IDs 104–120)")
    print("=" * 100)

    pca_test = PCA(n_components=8, random_state=42)
    scaler_emb_test = StandardScaler()
    dev_vis_pca = pca_test.fit_transform(scaler_emb_test.fit_transform(dev_raw_embs))
    test_vis_pca = pca_test.transform(scaler_emb_test.transform(test_raw_embs))

    dev_clin = df_dev[CLINICAL_COLS].values
    test_clin = df_test[CLINICAL_COLS].values

    dev_morph = df_dev[MACRO_MORPH_COLS].values
    test_morph = df_test[MACRO_MORPH_COLS].values

    dev_va = df_dev[['VA_baseline']].values
    test_va = df_test[['VA_baseline']].values

    test_track_data = {
        "E1: Baseline VA Only": (dev_va, test_va),
        "E2: Full Clinical Baseline (7 features)": (dev_clin, test_clin),
        "E3: Retinal Vision Embeddings (PCA-8)": (dev_vis_pca, test_vis_pca),
        "E4: Macro Morphometry (MLD, Base, Height, MHI, THI)": (dev_morph, test_morph),
        "E5: Clinical + Morphometry Fusion": (np.hstack([dev_clin, dev_morph]), np.hstack([test_clin, test_morph])),
        "E6: Clinical + Vision Fusion (Multimodal)": (np.hstack([dev_clin, dev_vis_pca]), np.hstack([test_clin, test_vis_pca])),
        "E7: Full Multimodal Fusion (Clin + Morph + Vision)": (
            np.hstack([dev_clin, dev_morph, dev_vis_pca]), np.hstack([test_clin, test_morph, test_vis_pca])
        )
    }

    test_table = []
    e6_test_patient_breakdown = []

    for t_name, (X_tr_raw, X_te_raw) in test_track_data.items():
        X_tr = scaler_tab.fit_transform(imputer.fit_transform(X_tr_raw))
        X_te = scaler_tab.transform(imputer.transform(X_te_raw))

        # Classification
        clf_lr = LogisticRegression(C=0.5, max_iter=500, random_state=42)
        clf_lr.fit(X_tr, y_dev_clf)
        prob_lr = clf_lr.predict_proba(X_te)[:, 1]

        clf_gbdt = HistGradientBoostingClassifier(
            max_depth=2, max_iter=35, learning_rate=0.05, min_samples_leaf=4, l2_regularization=1.5, random_state=42
        )
        clf_gbdt.fit(X_tr_raw, y_dev_clf)
        prob_gbdt = clf_gbdt.predict_proba(X_te_raw)[:, 1]

        test_prob = 0.5 * prob_lr + 0.5 * prob_gbdt
        test_pred_bin = (test_prob >= 0.5).astype(int)

        # Regression
        reg_ridge = Ridge(alpha=2.0, random_state=42)
        reg_ridge.fit(X_tr, y_dev_reg)
        pred_ridge_te = reg_ridge.predict(X_te)
        pred_ridge_dev = reg_ridge.predict(X_tr)

        reg_gbdt = HistGradientBoostingRegressor(
            max_depth=2, max_iter=35, learning_rate=0.05, min_samples_leaf=4, l2_regularization=1.5, random_state=42
        )
        reg_gbdt.fit(X_tr_raw, y_dev_reg)
        pred_gbdt_te = reg_gbdt.predict(X_te_raw)
        pred_gbdt_dev = reg_gbdt.predict(X_tr_raw)

        test_pred_reg = 0.5 * pred_ridge_te + 0.5 * pred_gbdt_te
        dev_pred_reg = 0.5 * pred_ridge_dev + 0.5 * pred_gbdt_dev

        # Conformal calibration on full Dev Cohort (N=104)
        dev_residuals = np.abs(y_dev_reg - dev_pred_reg)
        n_dev = len(df_dev)
        p_quantile = min(1.0, np.ceil((n_dev + 1) * 0.90) / n_dev)
        q_hat_dev = np.quantile(dev_residuals, p_quantile)

        # Apply to test set
        test_lower = test_pred_reg - q_hat_dev
        test_upper = test_pred_reg + q_hat_dev
        test_covered = (y_test_reg >= test_lower) & (y_test_reg <= test_upper)
        test_picp = np.mean(test_covered) * 100.0

        t_auc = roc_auc_score(y_test_clf, test_prob) * 100.0
        t_f1 = f1_score(y_test_clf, test_pred_bin, zero_division=0) * 100.0
        t_acc = accuracy_score(y_test_clf, test_pred_bin) * 100.0
        t_r2 = r2_score(y_test_reg, test_pred_reg)
        t_mae = mean_absolute_error(y_test_reg, test_pred_reg)
        t_rmse = np.sqrt(mean_squared_error(y_test_reg, test_pred_reg))

        test_table.append({
            'Model Track': t_name,
            'Test AUROC (%)': round(t_auc, 1),
            'Test Acc (%)': round(t_acc, 1),
            'Test R²': round(t_r2, 3),
            'Test MAE (let)': round(t_mae, 2),
            'Test RMSE (let)': round(t_rmse, 2),
            'Dev q_hat': round(q_hat_dev, 2),
            'Test Coverage (N=17)': f"{int(np.sum(test_covered))}/17 ({test_picp:.1f}%)"
        })

        if t_name == "E6: Clinical + Vision Fusion (Multimodal)":
            for p_i, pid in enumerate(df_test['id'].values):
                e6_test_patient_breakdown.append({
                    'id': int(pid),
                    'actual_gain': float(y_test_reg[p_i]),
                    'pred_gain': float(test_pred_reg[p_i]),
                    'lower_bound_90': float(test_lower[p_i]),
                    'upper_bound_90': float(test_upper[p_i]),
                    'is_covered': bool(test_covered[p_i]),
                    'residual': float(abs(y_test_reg[p_i] - test_pred_reg[p_i])),
                    'actual_binary': int(y_test_clf[p_i]),
                    'pred_prob': float(test_prob[p_i]),
                    'pred_binary': int(test_pred_bin[p_i])
                })

    test_df = pd.DataFrame(test_table)
    print(test_df.to_string(index=False))

    print("\n" + "=" * 100)
    print("3. E6 MULTIMODAL HELD-OUT PATIENT-BY-PATIENT BREAKDOWN (N=17)")
    print("=" * 100)
    p_df = pd.DataFrame(e6_test_patient_breakdown)
    print(p_df.to_string(index=False))

    # Save to disk
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(cv_table).to_csv("results/verified_clean_cv_results.csv", index=False)
    test_df.to_csv("results/verified_clean_test_results.csv", index=False)
    p_df.to_csv("results/verified_clean_patient_breakdown.csv", index=False)

    payload = {
        'cv_results': cv_table,
        'test_results': test_table,
        'e6_patient_breakdown': e6_test_patient_breakdown
    }
    with open("results/verified_clean_results.json", "w") as f:
        json.dump(payload, f, indent=2)

    print("\nOutputs saved to results/verified_clean_results.json")
    print("=" * 100)
    return payload


if __name__ == "__main__":
    run_pipeline()
