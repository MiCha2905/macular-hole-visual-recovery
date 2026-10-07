"""
Diagnostic Validation & Seed Sensitivity Module
Uses the unified multimodal evaluation engine (GBDT + Logistic/Ridge Ensemble)
to test:
1. Ceiling-Effect Hypothesis (VA_baseline <= 70 vs All) with the exact unified ensemble.
2. Seed Sensitivity Analysis across 5 random seeds (seed = 42, 123, 456, 789, 2026).
3. Embedding-Space Proximity (Cosine Similarity & Distance) for misclassified test cases.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, mean_absolute_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from src.data_preprocessing import load_and_clean_data, generate_repeated_cv_splits
from src.extract_morphometry import extract_all_morphometry
from src.multimodal_evaluator import fit_predict_ensemble, CLINICAL_COLS, MORPH_COLS

def run_diagnostic_validation(data_path="."):
    print("="*85)
    print("RUNNING DIAGNOSTIC VALIDATION & SEED SENSITIVITY ENGINE")
    print("="*85)
    
    # 1. Load Data
    df = load_and_clean_data(data_path)
    df_dev, df_test, folds_42 = generate_repeated_cv_splits(df, n_splits=5, n_repeats=5, random_state=42)
    
    # -------------------------------------------------------------
    # DIAGNOSTIC 1: Exact Unified Ensemble on Ceiling Subset
    # -------------------------------------------------------------
    print("\n--- Diagnostic 1: Exact Unified Ensemble on Ceiling Subset (Random State 42) ---")
    
    n_ceiling_total = (df['VA_baseline'] > 70).sum()
    n_ceiling_dev = (df_dev['VA_baseline'] > 70).sum()
    n_ceiling_test = (df_test['VA_baseline'] > 70).sum()
    
    print(f"Patients with VA_baseline > 70: Total={n_ceiling_total} / {len(df)} ({n_ceiling_total/len(df)*100:.1f}%), Dev={n_ceiling_dev} / {len(df_dev)}, Test={n_ceiling_test} / {len(df_test)}")
    
    e1_aurocs_all, e2_aurocs_all = [], []
    e1_aurocs_nc, e2_aurocs_nc = [], []
    
    imputer = SimpleImputer(strategy='median')
    scaler_tab = StandardScaler()
    
    for fold_info in folds_42:
        run_idx = fold_info['run_idx']
        tr_idx = fold_info['train_indices']
        val_idx = fold_info['val_indices']
        
        y_tr_clf = df_dev.iloc[tr_idx]['VA_gain_6mo_binary'].values
        y_val_clf = df_dev.iloc[val_idx]['VA_gain_6mo_binary'].values
        y_tr_reg = df_dev.iloc[tr_idx]['delta_VA_6months'].values
        y_val_reg = df_dev.iloc[val_idx]['delta_VA_6months'].values
        
        X_tr_e1 = df_dev.iloc[tr_idx][['VA_baseline']].values
        X_val_e1 = df_dev.iloc[val_idx][['VA_baseline']].values
        
        X_tr_e2 = df_dev.iloc[tr_idx][CLINICAL_COLS].values
        X_val_e2 = df_dev.iloc[val_idx][CLINICAL_COLS].values
        
        # Train E1 (Unified Ensemble)
        prob_e1, _, _ = fit_predict_ensemble(X_tr_e1, y_tr_clf, y_tr_reg, X_val_e1, run_idx, scaler_tab, imputer)
        e1_aurocs_all.append(roc_auc_score(y_val_clf, prob_e1))
        
        # Train E2 (Unified Ensemble)
        prob_e2, _, _ = fit_predict_ensemble(X_tr_e2, y_tr_clf, y_tr_reg, X_val_e2, run_idx, scaler_tab, imputer)
        e2_aurocs_all.append(roc_auc_score(y_val_clf, prob_e2))
        
        # Non-ceiling subset in this validation fold
        val_nc_indices = [i for i, idx in enumerate(val_idx) if df_dev.iloc[idx]['VA_baseline'] <= 70]
        if len(np.unique(y_val_clf[val_nc_indices])) > 1:
            e1_aurocs_nc.append(roc_auc_score(y_val_clf[val_nc_indices], prob_e1[val_nc_indices]))
            e2_aurocs_nc.append(roc_auc_score(y_val_clf[val_nc_indices], prob_e2[val_nc_indices]))
            
    print(f"\nAll Dev Patients (N=104, Seed 42 Unified Ensemble):")
    print(f"  E1 (VA Baseline): AUROC = {np.mean(e1_aurocs_all)*100:.2f} ± {np.std(e1_aurocs_all, ddof=1)*100:.2f}%")
    print(f"  E2 (Full Clinical): AUROC = {np.mean(e2_aurocs_all)*100:.2f} ± {np.std(e2_aurocs_all, ddof=1)*100:.2f}%")
    print(f"  Gap (E1 - E2): {np.mean(np.array(e1_aurocs_all) - np.array(e2_aurocs_all))*100:+.2f}%")
    
    print(f"\nNon-Ceiling Subset (VA_baseline <= 70, N=101 in Dev, Seed 42 Unified Ensemble):")
    print(f"  E1 (VA Baseline): AUROC = {np.mean(e1_aurocs_nc)*100:.2f} ± {np.std(e1_aurocs_nc, ddof=1)*100:.2f}%")
    print(f"  E2 (Full Clinical): AUROC = {np.mean(e2_aurocs_nc)*100:.2f} ± {np.std(e2_aurocs_nc, ddof=1)*100:.2f}%")
    print(f"  Gap (E1 - E2): {np.mean(np.array(e1_aurocs_nc) - np.array(e2_aurocs_nc))*100:+.2f}%")
    
    # -------------------------------------------------------------
    # DIAGNOSTIC 2: Multi-Seed Sensitivity Analysis (5 Random Seeds)
    # -------------------------------------------------------------
    print("\n--- Diagnostic 2: Seed Sensitivity Analysis Across 5 Random Seeds ---")
    seeds = [42, 123, 456, 789, 2026]
    seed_records = []
    
    for s in seeds:
        _, _, seed_folds = generate_repeated_cv_splits(df, n_splits=5, n_repeats=5, random_state=s)
        s_e1_aurocs = []
        s_e2_aurocs = []
        s_e1_maes = []
        s_e2_maes = []
        
        for fold_info in seed_folds:
            run_idx = fold_info['run_idx']
            tr_idx = fold_info['train_indices']
            val_idx = fold_info['val_indices']
            
            y_tr_clf = df_dev.iloc[tr_idx]['VA_gain_6mo_binary'].values
            y_val_clf = df_dev.iloc[val_idx]['VA_gain_6mo_binary'].values
            y_tr_reg = df_dev.iloc[tr_idx]['delta_VA_6months'].values
            y_val_reg = df_dev.iloc[val_idx]['delta_VA_6months'].values
            
            X_tr_e1 = df_dev.iloc[tr_idx][['VA_baseline']].values
            X_val_e1 = df_dev.iloc[val_idx][['VA_baseline']].values
            X_tr_e2 = df_dev.iloc[tr_idx][CLINICAL_COLS].values
            X_val_e2 = df_dev.iloc[val_idx][CLINICAL_COLS].values
            
            prob_e1, _, pred_reg_e1 = fit_predict_ensemble(X_tr_e1, y_tr_clf, y_tr_reg, X_val_e1, run_idx, scaler_tab, imputer)
            prob_e2, _, pred_reg_e2 = fit_predict_ensemble(X_tr_e2, y_tr_clf, y_tr_reg, X_val_e2, run_idx, scaler_tab, imputer)
            
            s_e1_aurocs.append(roc_auc_score(y_val_clf, prob_e1))
            s_e2_aurocs.append(roc_auc_score(y_val_clf, prob_e2))
            s_e1_maes.append(mean_absolute_error(y_val_reg, pred_reg_e1))
            s_e2_maes.append(mean_absolute_error(y_val_reg, pred_reg_e2))
            
        mean_e1_auc = np.mean(s_e1_aurocs)*100
        mean_e2_auc = np.mean(s_e2_aurocs)*100
        auc_gap = mean_e1_auc - mean_e2_auc
        
        mean_e1_mae = np.mean(s_e1_maes)
        mean_e2_mae = np.mean(s_e2_maes)
        
        seed_records.append({
            'Random_Seed': s,
            'E1_AUROC (%)': round(mean_e1_auc, 2),
            'E2_AUROC (%)': round(mean_e2_auc, 2),
            'AUROC_Gap (E1 - E2)': round(auc_gap, 2),
            'E1_MAE (letters)': round(mean_e1_mae, 2),
            'E2_MAE (letters)': round(mean_e2_mae, 2)
        })
        
    seed_df = pd.DataFrame(seed_records)
    print(seed_df.to_string(index=False))
    print(f"\nMean E1 AUROC Across 5 Seeds: {seed_df['E1_AUROC (%)'].mean():.2f} ± {seed_df['E1_AUROC (%)'].std(ddof=1):.2f}%")
    print(f"Mean E2 AUROC Across 5 Seeds: {seed_df['E2_AUROC (%)'].mean():.2f} ± {seed_df['E2_AUROC (%)'].std(ddof=1):.2f}%")
    print(f"Mean AUROC Gap Across 5 Seeds: {seed_df['AUROC_Gap (E1 - E2)'].mean():+.2f} ± {seed_df['AUROC_Gap (E1 - E2)'].std(ddof=1):.2f}%")
    
    # -------------------------------------------------------------
    # DIAGNOSTIC 3: Embedding-Space Proximity for Test Patients
    # -------------------------------------------------------------
    print("\n--- Diagnostic 3: Embedding-Space Proximity for Test Patients ---")
    raw_emb_path = "data/vision_embeddings_raw.npy"
    raw_embs = np.load(raw_emb_path)
    pid_arr = np.load("data/embedding_patient_ids.npy")
    pid_to_idx = {pid: i for i, pid in enumerate(pid_arr)}
    
    dev_embs = raw_embs[[pid_to_idx[p] for p in df_dev['id']]]
    test_embs = raw_embs[[pid_to_idx[p] for p in df_test['id']]]
    
    scaler_emb = StandardScaler()
    pca = PCA(n_components=8, random_state=42)
    dev_pca = pca.fit_transform(scaler_emb.fit_transform(dev_embs))
    test_pca = pca.transform(scaler_emb.transform(test_embs))
    
    y_dev = df_dev['VA_gain_6mo_binary'].values
    y_test = df_test['VA_gain_6mo_binary'].values
    
    dev_pos_pca = dev_pca[y_dev == 1]
    dev_neg_pca = dev_pca[y_dev == 0]
    
    cos_sim_to_pos = cosine_similarity(test_pca, dev_pos_pca)
    cos_sim_to_neg = cosine_similarity(test_pca, dev_neg_pca)
    
    mean_sim_pos = np.mean(cos_sim_to_pos, axis=1)
    mean_sim_neg = np.mean(cos_sim_to_neg, axis=1)
    
    test_proximity_df = pd.DataFrame({
        'test_id': df_test['id'],
        'y_true': y_test,
        'mean_sim_to_dev_pos': mean_sim_pos.round(3),
        'mean_sim_to_dev_neg': mean_sim_neg.round(3),
        'relative_pos_affinity': (mean_sim_pos - mean_sim_neg).round(3)
    })
    
    miscl_neg = test_proximity_df[test_proximity_df['y_true'] == 0].sort_values('relative_pos_affinity', ascending=False)
    print("\nNegative Test Patients Ranked by Embedding-Space Affinity to Positive Training Set:")
    print(miscl_neg.head(5).to_string(index=False))
    
    results = {
        'n_ceiling_total': int(n_ceiling_total),
        'e1_all_seed42': float(np.mean(e1_aurocs_all)*100),
        'e2_all_seed42': float(np.mean(e2_aurocs_all)*100),
        'gap_all_seed42': float(np.mean(np.array(e1_aurocs_all) - np.array(e2_aurocs_all))*100),
        'e1_nc_seed42': float(np.mean(e1_aurocs_nc)*100),
        'e2_nc_seed42': float(np.mean(e2_aurocs_nc)*100),
        'gap_nc_seed42': float(np.mean(np.array(e1_aurocs_nc) - np.array(e2_aurocs_nc))*100),
        'seed_sensitivity': seed_records,
        'test_proximity': test_proximity_df.to_dict(orient='records')
    }
    
    with open("results/diagnostic_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDiagnostic validation results saved to results/diagnostic_validation_results.json")
    return results

if __name__ == "__main__":
    run_diagnostic_validation()
