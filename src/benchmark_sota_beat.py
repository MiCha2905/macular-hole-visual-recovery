"""
Master SOTA-Beating Benchmark:
Evaluates whether:
1. Retinal Contrastive SSL Vision Encoder fixes the Vision-Only collapse (E3 > 70% vs ImageNet 59.5%).
2. EZ & ELM Layer Integrity Biomarkers add genuine biological signal over macro-geometry.
3. Full Regularized Multimodal Fusion decisively outperforms Lachance et al. (TVST 2022, 81.9% AUROC).
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, kendalltau
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, brier_score_loss,
    r2_score, mean_absolute_error, mean_squared_error, cohen_kappa_score
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from src.data_preprocessing import load_and_clean_data, generate_repeated_cv_splits
from src.extract_morphometry import extract_all_morphometry
from src.extract_layer_morphometry import extract_all_layer_morphometry
from src.experiment_sparsity_and_ordinal import FrankHallOrdinalClassifier, construct_ordinal_targets

CLINICAL_COLS = ['age', 'sex_encoded', 'pseudophakic', 'mh_duration', 'elevated_edge', 'mh_size', 'VA_baseline']
MACRO_MORPH_COLS = ['mld_mean', 'bd_mean', 'height_mean', 'mhi_mean', 'thi_mean', 'dhi_mean']
LAYER_BIOMARKER_COLS = ['ez_defect_mean', 'elm_defect_mean', 'ez_integrity_mean', 'elm_integrity_mean', 'photoreceptor_cuff_mean']

def run_sota_comparison_benchmark(df, dev_raw_ssl_embs, test_raw_ssl_embs, folds):
    print("\n" + "="*80)
    print("EXECUTING MASTER SOTA-BEATING MULTIMODAL BENCHMARK (25 FOLDS)")
    print("="*80)
    
    dev_mask = df['original_split'] != 'test'
    df_dev = df[dev_mask].copy().reset_index(drop=True)
    df_test = df[~dev_mask].copy().reset_index(drop=True)
    
    df_dev['ordinal_target'] = construct_ordinal_targets(df_dev)
    df_test['ordinal_target'] = construct_ordinal_targets(df_test)
    
    imputer = SimpleImputer(strategy='median')
    scaler_tab = StandardScaler()
    
    track_names = [
        "1. Baseline VA Only",
        "2. Clinical 7 Features (L1 C=0.2)",
        "3. Previous SOTA Benchmark (TVST 2022 Ref: 81.9%)",
        "4. Retinal SSL Vision Only (PCA-8)",
        "5. EZ/ELM Layer Integrity Only",
        "6. Clinical + Retinal SSL Vision",
        "7. Clinical + EZ/ELM Layer Integrity",
        "8. Full Next-Gen Fusion (Clin + EZ/ELM + Retinal SSL, L1 C=0.5)"
    ]
    
    cv_records = {t: {'auroc': [], 'f1': [], 'brier': [], 'r2': [], 'mae': [], 'qwk': []} for t in track_names if "Ref" not in t}
    
    for fold_info in folds:
        run_idx = fold_info['run_idx']
        tr_idx = fold_info['train_indices']
        val_idx = fold_info['val_indices']
        
        y_tr_clf = df_dev.iloc[tr_idx]['VA_gain_6mo_binary'].values
        y_val_clf = df_dev.iloc[val_idx]['VA_gain_6mo_binary'].values
        y_tr_reg = df_dev.iloc[tr_idx]['delta_VA_6months'].values
        y_val_reg = df_dev.iloc[val_idx]['delta_VA_6months'].values
        y_tr_ord = df_dev.iloc[tr_idx]['ordinal_target'].values
        y_val_ord = df_dev.iloc[val_idx]['ordinal_target'].values
        
        # PCA on Retinal SSL Embeddings (strictly inside fold)
        pca = PCA(n_components=8, random_state=42)
        scaler_emb = StandardScaler()
        tr_ssl_pca = pca.fit_transform(scaler_emb.fit_transform(dev_raw_ssl_embs[tr_idx]))
        val_ssl_pca = pca.transform(scaler_emb.transform(dev_raw_ssl_embs[val_idx]))
        
        tr_clin = df_dev.iloc[tr_idx][CLINICAL_COLS].values
        val_clin = df_dev.iloc[val_idx][CLINICAL_COLS].values
        
        tr_layer = df_dev.iloc[tr_idx][LAYER_BIOMARKER_COLS].values
        val_layer = df_dev.iloc[val_idx][LAYER_BIOMARKER_COLS].values
        
        feat_configs = {
            "1. Baseline VA Only": (df_dev.iloc[tr_idx][['VA_baseline']].values, df_dev.iloc[val_idx][['VA_baseline']].values, 1.0),
            "2. Clinical 7 Features (L1 C=0.2)": (tr_clin, val_clin, 0.2),
            "4. Retinal SSL Vision Only (PCA-8)": (tr_ssl_pca, val_ssl_pca, 0.5),
            "5. EZ/ELM Layer Integrity Only": (tr_layer, val_layer, 0.5),
            "6. Clinical + Retinal SSL Vision": (np.hstack([tr_clin, tr_ssl_pca]), np.hstack([val_clin, val_ssl_pca]), 0.3),
            "7. Clinical + EZ/ELM Layer Integrity": (np.hstack([tr_clin, tr_layer]), np.hstack([val_clin, val_layer]), 0.3),
            "8. Full Next-Gen Fusion (Clin + EZ/ELM + Retinal SSL, L1 C=0.5)": (
                np.hstack([tr_clin, tr_layer, tr_ssl_pca]), np.hstack([val_clin, val_layer, val_ssl_pca]), 0.5
            )
        }
        
        for t_name, (X_tr_raw, X_val_raw, c_val) in feat_configs.items():
            X_tr = scaler_tab.fit_transform(imputer.fit_transform(X_tr_raw))
            X_val = scaler_tab.transform(imputer.transform(X_val_raw))
            
            # Binary Classifier (L1 Regularized)
            clf = LogisticRegression(penalty='l1', C=c_val, solver='liblinear', random_state=42 + run_idx)
            clf.fit(X_tr, y_tr_clf)
            prob = clf.predict_proba(X_val)[:, 1]
            pred_bin = (prob >= 0.5).astype(int)
            
            # Regression Model (Ridge)
            reg = Ridge(alpha=2.0, random_state=42 + run_idx)
            reg.fit(X_tr, y_tr_reg)
            pred_reg = reg.predict(X_val)
            
            # Ordinal Model (Frank-Hall L1)
            ord_clf = FrankHallOrdinalClassifier(penalty='l1', C=c_val, random_state=42 + run_idx)
            ord_clf.fit(X_tr, y_tr_ord)
            pred_ord = ord_clf.predict(X_val)
            qwk = cohen_kappa_score(y_val_ord, pred_ord, weights='quadratic')
            
            cv_records[t_name]['auroc'].append(roc_auc_score(y_val_clf, prob))
            cv_records[t_name]['f1'].append(f1_score(y_val_clf, pred_bin, zero_division=0))
            cv_records[t_name]['brier'].append(brier_score_loss(y_val_clf, prob))
            cv_records[t_name]['r2'].append(r2_score(y_val_reg, pred_reg))
            cv_records[t_name]['mae'].append(mean_absolute_error(y_val_reg, pred_reg))
            cv_records[t_name]['qwk'].append(qwk)
            
    # Print Table
    print("\n" + "="*80)
    print("MASTER SOTA COMPARISON RESULTS (25-Fold Repeated Stratified CV)")
    print("="*80)
    print(f"{'Model Architecture / Track':<55} | {'AUROC (CV)':<18} | {'QWK (Ordinal)':<16} | {'MAE (Letters)':<14}")
    print("-" * 110)
    
    summary_list = []
    for t_name in track_names:
        if "Ref: 81.9%" in t_name:
            print(f"{t_name:<55} | 81.9 ± 5.2%        | N/A              | N/A")
            summary_list.append({'track': t_name, 'auroc_mean': 0.819, 'auroc_sd': 0.052, 'qwk_mean': np.nan, 'mae_mean': np.nan})
            continue
            
        auc_m = np.mean(cv_records[t_name]['auroc']) * 100
        auc_sd = np.std(cv_records[t_name]['auroc']) * 100
        qwk_m = np.mean(cv_records[t_name]['qwk'])
        qwk_sd = np.std(cv_records[t_name]['qwk'])
        mae_m = np.mean(cv_records[t_name]['mae'])
        mae_sd = np.std(cv_records[t_name]['mae'])
        
        print(f"{t_name:<55} | {auc_m:4.1f} ± {auc_sd:4.1f}%      | {qwk_m:5.3f} ± {qwk_sd:5.3f}   | {mae_m:5.2f} ± {mae_sd:4.2f}")
        summary_list.append({
            'track': t_name,
            'auroc_mean': float(auc_m / 100.0),
            'auroc_sd': float(auc_sd / 100.0),
            'qwk_mean': float(qwk_m),
            'qwk_sd': float(qwk_sd),
            'mae_mean': float(mae_m),
            'mae_sd': float(mae_sd)
        })
        
    return cv_records, summary_list

def run_held_out_test_comparison(df, dev_raw_ssl_embs, test_raw_ssl_embs):
    print("\n" + "="*80)
    print("LOCKED HELD-OUT TEST SET EVALUATION (N=17)")
    print("="*80)
    
    dev_mask = df['original_split'] != 'test'
    df_dev = df[dev_mask].copy().reset_index(drop=True)
    df_test = df[~dev_mask].copy().reset_index(drop=True)
    
    df_dev['ordinal_target'] = construct_ordinal_targets(df_dev)
    df_test['ordinal_target'] = construct_ordinal_targets(df_test)
    
    y_dev_clf = df_dev['VA_gain_6mo_binary'].values
    y_test_clf = df_test['VA_gain_6mo_binary'].values
    y_dev_ord = df_dev['ordinal_target'].values
    y_test_ord = df_test['ordinal_target'].values
    
    pca = PCA(n_components=8, random_state=42)
    scaler_emb = StandardScaler()
    dev_ssl_pca = pca.fit_transform(scaler_emb.fit_transform(dev_raw_ssl_embs))
    test_ssl_pca = pca.transform(scaler_emb.transform(test_raw_ssl_embs))
    
    dev_clin = df_dev[CLINICAL_COLS].values
    test_clin = df_test[CLINICAL_COLS].values
    
    dev_layer = df_dev[LAYER_BIOMARKER_COLS].values
    test_layer = df_test[LAYER_BIOMARKER_COLS].values
    
    imputer = SimpleImputer(strategy='median')
    scaler_tab = StandardScaler()
    
    test_configs = {
        "1. Baseline VA Only": (df_dev[['VA_baseline']].values, df_test[['VA_baseline']].values, 1.0),
        "2. Clinical 7 Features (L1 C=0.2)": (dev_clin, test_clin, 0.2),
        "3. Published SOTA Benchmark (TVST 2022 Ref)": (None, None, 0.0),
        "4. Retinal SSL Vision Only": (dev_ssl_pca, test_ssl_pca, 0.5),
        "5. EZ/ELM Layer Integrity Only": (dev_layer, test_layer, 0.5),
        "6. Clinical + Retinal SSL Vision": (np.hstack([dev_clin, dev_ssl_pca]), np.hstack([test_clin, test_ssl_pca]), 0.3),
        "7. Full Next-Gen Fusion (Clin + EZ/ELM + Retinal SSL)": (
            np.hstack([dev_clin, dev_layer, dev_ssl_pca]), np.hstack([test_clin, test_layer, test_ssl_pca]), 0.5
        )
    }
    
    test_results = []
    print(f"{'Model Track':<55} | {'Test AUROC':<12} | {'Test QWK':<12} | {'Exact Acc':<12} | {'Adj Acc':<12}")
    print("-" * 110)
    
    for t_name, (X_dev_raw, X_test_raw, c_val) in test_configs.items():
        if "Published SOTA" in t_name:
            print(f"{t_name:<55} | 81.9%        | N/A          | N/A          | N/A")
            test_results.append({'track': t_name, 'test_auroc': 0.819, 'test_qwk': np.nan, 'exact_acc': np.nan, 'adj_acc': np.nan})
            continue
            
        X_dev = scaler_tab.fit_transform(imputer.fit_transform(X_dev_raw))
        X_test = scaler_tab.transform(imputer.transform(X_test_raw))
        
        clf = LogisticRegression(penalty='l1', C=c_val, solver='liblinear', random_state=42)
        clf.fit(X_dev, y_dev_clf)
        auc = roc_auc_score(y_test_clf, clf.predict_proba(X_test)[:, 1])
        
        ord_clf = FrankHallOrdinalClassifier(penalty='l1', C=c_val, random_state=42)
        ord_clf.fit(X_dev, y_dev_ord)
        pred_ord = ord_clf.predict(X_test)
        qwk = cohen_kappa_score(y_test_ord, pred_ord, weights='quadratic')
        acc = accuracy_score(y_test_ord, pred_ord)
        adj_acc = np.mean(np.abs(y_test_ord - pred_ord) <= 1)
        
        print(f"{t_name:<55} | {auc*100:4.1f}%       | {qwk:5.3f}        | {acc*100:4.1f}%        | {adj_acc*100:4.1f}%")
        test_results.append({
            'track': t_name,
            'test_auroc': float(auc),
            'test_qwk': float(qwk),
            'exact_acc': float(acc),
            'adj_acc': float(adj_acc)
        })
        
    return test_results

def main():
    df = load_and_clean_data(".")
    df_dev, df_test, folds = generate_repeated_cv_splits(df, n_splits=5, n_repeats=5, random_state=42)
    
    # Merge Morphometry + Layer Integrity
    macro_df = extract_all_morphometry(df)
    layer_df = extract_all_layer_morphometry(df)
    
    df = df.merge(macro_df, on='id', how='left')
    df = df.merge(layer_df, on='id', how='left')
    
    # Load Retinal SSL Embeddings
    ssl_embs = np.load("data/retinal_ssl_embeddings.npy")
    dev_mask = (df['original_split'] != 'test').values
    dev_raw_ssl_embs = ssl_embs[dev_mask]
    test_raw_ssl_embs = ssl_embs[~dev_mask]
    
    # Run CV Benchmark
    cv_records, cv_summary = run_sota_comparison_benchmark(df, dev_raw_ssl_embs, test_raw_ssl_embs, folds)
    
    # Run Held-Out Test Evaluation
    test_summary = run_held_out_test_comparison(df, dev_raw_ssl_embs, test_raw_ssl_embs)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/sota_beat_comparison.json", "w") as f:
        json.dump({'cv_summary': cv_summary, 'test_summary': test_summary}, f, indent=2)
        
    print("\nSaved full SOTA comparison to results/sota_beat_comparison.json")

if __name__ == "__main__":
    main()
