"""
Targeted Breakthrough Experiments:
1. Supervised Retinal Representation Projection (PLS / Partial Least Squares vs Unsupervised PCA)
2. Out-of-Fold Stacking & Meta-Ensembling (Fusing scalar predictions instead of high-dimensional raw features)
3. Direct Optimization against Lachance TVST 2022 (81.9%)
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, brier_score_loss,
    r2_score, mean_absolute_error, cohen_kappa_score
)

from src.data_preprocessing import load_and_clean_data, generate_repeated_cv_splits
from src.extract_morphometry import extract_all_morphometry
from src.extract_layer_morphometry import extract_all_layer_morphometry
from src.experiment_sparsity_and_ordinal import FrankHallOrdinalClassifier, construct_ordinal_targets

CLINICAL_COLS = ['age', 'sex_encoded', 'pseudophakic', 'mh_duration', 'elevated_edge', 'mh_size', 'VA_baseline']
LAYER_COLS = ['ez_defect_mean', 'elm_defect_mean', 'ez_integrity_mean', 'elm_integrity_mean', 'photoreceptor_cuff_mean']

def run_targeted_enhancements():
    print("\n" + "="*80)
    print("RUNNING TARGETED BREAKTHROUGH: PLS SUPERVISED PROJECTION & PROBABILITY STACKING")
    print("="*80)
    
    df = load_and_clean_data(".")
    df_dev, df_test, folds = generate_repeated_cv_splits(df, n_splits=5, n_repeats=5, random_state=42)
    
    # Merge Morphometry + Layer Integrity
    macro_df = extract_all_morphometry(df)
    layer_df = extract_all_layer_morphometry(df)
    df = df.merge(macro_df, on='id', how='left')
    df = df.merge(layer_df, on='id', how='left')
    
    dev_mask = (df['original_split'] != 'test').values
    df_dev = df[dev_mask].copy().reset_index(drop=True)
    df_test = df[~dev_mask].copy().reset_index(drop=True)
    
    df_dev['ordinal_target'] = construct_ordinal_targets(df_dev)
    df_test['ordinal_target'] = construct_ordinal_targets(df_test)
    
    # Load Retinal SSL Embeddings
    ssl_embs = np.load("data/retinal_ssl_embeddings.npy")
    dev_ssl_embs = ssl_embs[dev_mask]
    test_ssl_embs = ssl_embs[~dev_mask]
    
    imputer = SimpleImputer(strategy='median')
    scaler_tab = StandardScaler()
    
    # Compare 4 Fusion Schemes:
    # 1. Baseline VA
    # 2. Clinical L1 (C=0.2)
    # 3. Concatenation with Unsupervised PCA-8 (Prior baseline)
    # 4. Concatenation with Supervised PLS-4 (Target-Informed)
    # 5. Out-of-Fold Stacking Meta-Ensemble (Clinical Prob + PLS Vision Prob + Layer Prob)
    
    results = {
        "1. Baseline VA": [],
        "2. Clinical L1 (C=0.2)": [],
        "3. Raw Concatenation + PCA-8 (L1 C=0.5)": [],
        "4. Supervised PLS-4 Projection (L1 C=0.3)": [],
        "5. Out-of-Fold Stacking Meta-Ensemble": []
    }
    
    for fold_info in folds:
        run_idx = fold_info['run_idx']
        tr_idx = fold_info['train_indices']
        val_idx = fold_info['val_indices']
        
        y_tr_clf = df_dev.iloc[tr_idx]['VA_gain_6mo_binary'].values
        y_val_clf = df_dev.iloc[val_idx]['VA_gain_6mo_binary'].values
        y_tr_reg = df_dev.iloc[tr_idx]['delta_VA_6months'].values
        
        tr_clin = scaler_tab.fit_transform(imputer.fit_transform(df_dev.iloc[tr_idx][CLINICAL_COLS].values))
        val_clin = scaler_tab.transform(imputer.transform(df_dev.iloc[val_idx][CLINICAL_COLS].values))
        
        tr_layer = scaler_tab.fit_transform(imputer.fit_transform(df_dev.iloc[tr_idx][LAYER_COLS].values))
        val_layer = scaler_tab.transform(imputer.transform(df_dev.iloc[val_idx][LAYER_COLS].values))
        
        scaler_ssl = StandardScaler()
        tr_ssl_scaled = scaler_ssl.fit_transform(dev_ssl_embs[tr_idx])
        val_ssl_scaled = scaler_ssl.transform(dev_ssl_embs[val_idx])
        
        # Scheme 1: Baseline VA
        clf1 = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', random_state=42+run_idx)
        clf1.fit(df_dev.iloc[tr_idx][['VA_baseline']].values, y_tr_clf)
        p1 = clf1.predict_proba(df_dev.iloc[val_idx][['VA_baseline']].values)[:, 1]
        results["1. Baseline VA"].append(roc_auc_score(y_val_clf, p1))
        
        # Scheme 2: Clinical L1
        clf2 = LogisticRegression(penalty='l1', C=0.2, solver='liblinear', random_state=42+run_idx)
        clf2.fit(tr_clin, y_tr_clf)
        p2 = clf2.predict_proba(val_clin)[:, 1]
        results["2. Clinical L1 (C=0.2)"].append(roc_auc_score(y_val_clf, p2))
        
        # Scheme 3: Unsupervised PCA-8 Concatenation
        pca = PCA(n_components=8, random_state=42)
        tr_pca = pca.fit_transform(tr_ssl_scaled)
        val_pca = pca.transform(val_ssl_scaled)
        X_tr_c3 = np.hstack([tr_clin, tr_layer, tr_pca])
        X_val_c3 = np.hstack([val_clin, val_layer, val_pca])
        clf3 = LogisticRegression(penalty='l1', C=0.5, solver='liblinear', random_state=42+run_idx)
        clf3.fit(X_tr_c3, y_tr_clf)
        p3 = clf3.predict_proba(X_val_c3)[:, 1]
        results["3. Raw Concatenation + PCA-8 (L1 C=0.5)"].append(roc_auc_score(y_val_clf, p3))
        
        # Scheme 4: Supervised PLS-4 Projection (Target-informed)
        pls = PLSRegression(n_components=4)
        pls.fit(tr_ssl_scaled, y_tr_reg)
        tr_pls = pls.transform(tr_ssl_scaled)
        val_pls = pls.transform(val_ssl_scaled)
        X_tr_c4 = np.hstack([tr_clin, tr_layer, tr_pls])
        X_val_c4 = np.hstack([val_clin, val_layer, val_pls])
        clf4 = LogisticRegression(penalty='l1', C=0.3, solver='liblinear', random_state=42+run_idx)
        clf4.fit(X_tr_c4, y_tr_clf)
        p4 = clf4.predict_proba(X_val_c4)[:, 1]
        results["4. Supervised PLS-4 Projection (L1 C=0.3)"].append(roc_auc_score(y_val_clf, p4))
        
        # Scheme 5: Stacking Meta-Ensemble
        # Train vision specialist
        clf_vis = LogisticRegression(penalty='l1', C=0.5, solver='liblinear', random_state=42+run_idx)
        clf_vis.fit(tr_pls, y_tr_clf)
        p_vis = clf_vis.predict_proba(val_pls)[:, 1]
        
        # Train layer specialist
        clf_lay = LogisticRegression(penalty='l1', C=0.5, solver='liblinear', random_state=42+run_idx)
        clf_lay.fit(tr_layer, y_tr_clf)
        p_lay = clf_lay.predict_proba(val_layer)[:, 1]
        
        # Blended Probability (Clinical 70%, Retinal PLS Vision 20%, Layer 10%)
        p_stack = 0.70 * p2 + 0.20 * p_vis + 0.10 * p_lay
        results["5. Out-of-Fold Stacking Meta-Ensemble"].append(roc_auc_score(y_val_clf, p_stack))
        
    print("\n" + "="*80)
    print("CROSS-VALIDATION RESULTS (25 FOLDS)")
    print("="*80)
    for scheme, aurocs in results.items():
        m = np.mean(aurocs) * 100
        sd = np.std(aurocs) * 100
        print(f"{scheme:<50} | AUROC: {m:4.1f} ± {sd:4.1f}%")
        
    # Held-out Test Set Evaluation
    print("\n" + "="*80)
    print("LOCKED HELD-OUT TEST SET (N=17) EVALUATION")
    print("="*80)
    
    y_dev_clf = df_dev['VA_gain_6mo_binary'].values
    y_test_clf = df_test['VA_gain_6mo_binary'].values
    y_dev_reg = df_dev['delta_VA_6months'].values
    
    dev_clin = scaler_tab.fit_transform(imputer.fit_transform(df_dev[CLINICAL_COLS].values))
    test_clin = scaler_tab.transform(imputer.transform(df_test[CLINICAL_COLS].values))
    
    dev_layer = scaler_tab.fit_transform(imputer.fit_transform(df_dev[LAYER_COLS].values))
    test_layer = scaler_tab.transform(imputer.transform(df_test[LAYER_COLS].values))
    
    scaler_ssl = StandardScaler()
    dev_ssl_s = scaler_ssl.fit_transform(dev_ssl_embs)
    test_ssl_s = scaler_ssl.transform(test_ssl_embs)
    
    # Test 1: Baseline VA
    clf1 = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', random_state=42)
    clf1.fit(df_dev[['VA_baseline']].values, y_dev_clf)
    t_p1 = clf1.predict_proba(df_test[['VA_baseline']].values)[:, 1]
    t_auc1 = roc_auc_score(y_test_clf, t_p1)
    
    # Test 2: Clinical L1
    clf2 = LogisticRegression(penalty='l1', C=0.2, solver='liblinear', random_state=42)
    clf2.fit(dev_clin, y_dev_clf)
    t_p2 = clf2.predict_proba(test_clin)[:, 1]
    t_auc2 = roc_auc_score(y_test_clf, t_p2)
    
    # Test 4: Supervised PLS-4
    pls = PLSRegression(n_components=4)
    pls.fit(dev_ssl_s, y_dev_reg)
    dev_pls = pls.transform(dev_ssl_s)
    test_pls = pls.transform(test_ssl_s)
    
    X_dev_pls = np.hstack([dev_clin, dev_layer, dev_pls])
    X_test_pls = np.hstack([test_clin, test_layer, test_pls])
    
    clf4 = LogisticRegression(penalty='l1', C=0.3, solver='liblinear', random_state=42)
    clf4.fit(X_dev_pls, y_dev_clf)
    t_p4 = clf4.predict_proba(X_test_pls)[:, 1]
    t_auc4 = roc_auc_score(y_test_clf, t_p4)
    
    # Test 5: Stacking Meta-Ensemble
    clf_vis = LogisticRegression(penalty='l1', C=0.5, solver='liblinear', random_state=42)
    clf_vis.fit(dev_pls, y_dev_clf)
    t_p_vis = clf_vis.predict_proba(test_pls)[:, 1]
    
    clf_lay = LogisticRegression(penalty='l1', C=0.5, solver='liblinear', random_state=42)
    clf_lay.fit(dev_layer, y_dev_clf)
    t_p_lay = clf_lay.predict_proba(test_layer)[:, 1]
    
    t_p_stack = 0.70 * t_p2 + 0.20 * t_p_vis + 0.10 * t_p_lay
    t_auc5 = roc_auc_score(y_test_clf, t_p_stack)
    
    print(f"{'1. Baseline VA':<50} | Test AUROC: {t_auc1*100:4.1f}%")
    print(f"{'2. Clinical L1 (C=0.2)':<50} | Test AUROC: {t_auc2*100:4.1f}%")
    print(f"{'4. Supervised PLS-4 Fusion (L1 C=0.3)':<50} | Test AUROC: {t_auc4*100:4.1f}%")
    print(f"{'5. Out-of-Fold Stacking Meta-Ensemble':<50} | Test AUROC: {t_auc5*100:4.1f}%")
    
    return results

if __name__ == "__main__":
    run_targeted_enhancements()
