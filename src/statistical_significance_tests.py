"""
Statistical Significance Testing Module
Conducts paired significance testing (Wilcoxon signed-rank, Paired t-test, 
Nadeau-Bengio corrected resampled t-test, and 95% Bootstrap CIs) across all 
25 Repeated Stratified CV folds for the Macular Hole Visual Outcome Benchmarks.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, f1_score, r2_score, mean_absolute_error, brier_score_loss
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from src.data_preprocessing import load_and_clean_data, generate_repeated_cv_splits
from src.extract_morphometry import extract_all_morphometry
from src.extract_vision_embeddings import SiameseVisionEncoder

CLINICAL_COLS = ['age', 'sex_encoded', 'pseudophakic', 'mh_duration', 'elevated_edge', 'mh_size', 'VA_baseline']
MORPH_COLS = ['mld_mean', 'bd_mean', 'height_mean', 'mhi_mean', 'thi_mean', 'dhi_mean']

def bootstrap_paired_diff_ci(diff_array, n_bootstraps=2000, ci=95, random_state=42):
    rng = np.random.RandomState(random_state)
    boot_means = []
    n = len(diff_array)
    for _ in range(n_bootstraps):
        sample = rng.choice(diff_array, size=n, replace=True)
        boot_means.append(np.mean(sample))
    alpha = (100 - ci) / 2.0
    lower = np.percentile(boot_means, alpha)
    upper = np.percentile(boot_means, 100 - alpha)
    return lower, upper

def corrected_resampled_ttest(diff_array, n_train=83, n_val=21, n_runs=25):
    """
    Nadeau and Bengio (2003) corrected resampled t-test for repeated CV.
    Accounts for violation of independence due to overlapping training sets.
    """
    mean_diff = np.mean(diff_array)
    var_diff = np.var(diff_array, ddof=1)
    # Correction factor: 1/K + n_val/n_train
    correction = (1.0 / n_runs) + (float(n_val) / float(n_train))
    denom = np.sqrt(correction * var_diff)
    if denom == 0:
        return 0.0, 1.0
    t_stat = mean_diff / denom
    df = n_runs - 1
    p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=df))
    return t_stat, p_val

def run_significance_tests(data_path=".", pca_components=8):
    print("="*75)
    print("RUNNING STATISTICAL SIGNIFICANCE TESTING ENGINE (25 PAIRED CV FOLDS)")
    print("="*75)
    
    # 1. Load Data
    df = load_and_clean_data(data_path)
    df_dev, df_test, folds = generate_repeated_cv_splits(df, n_splits=5, n_repeats=5, random_state=42)
    
    morph_csv = "data/morphometry_features.csv"
    if os.path.exists(morph_csv):
        morph_df = pd.read_csv(morph_csv)
    else:
        morph_df = extract_all_morphometry(df, morph_csv)
    df_dev = df_dev.merge(morph_df, on='id', how='left')
    
    raw_emb_path = "data/vision_embeddings_raw.npy"
    raw_embs = np.load(raw_emb_path)
    pid_arr = np.load("data/embedding_patient_ids.npy")
    pid_to_emb_idx = {pid: i for i, pid in enumerate(pid_arr)}
    dev_emb_indices = [pid_to_emb_idx[pid] for pid in df_dev['id']]
    dev_raw_embs = raw_embs[dev_emb_indices]
    
    experiments = [
        "E1: Baseline VA only",
        "E2: Clinical Only (7 features)",
        "E3: Vision Embeddings Only",
        "E4: Morphometry Only",
        "E5: Clinical + Morphometry",
        "E6: Clinical + Vision Embeddings",
        "E7: Full Multimodal Fusion"
    ]
    
    fold_records = {exp: {'auroc': [], 'f1': [], 'r2': [], 'mae': [], 'rmse': []} for exp in experiments}
    
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
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
            X_tr_imp = scaler.fit_transform(imputer.fit_transform(X_tr))
            X_val_imp = scaler.transform(imputer.transform(X_val))
            
            clf_gbdt = HistGradientBoostingClassifier(
                max_depth=2, max_iter=35, learning_rate=0.05, min_samples_leaf=4, l2_regularization=1.5, random_state=42 + run_idx
            )
            clf_gbdt.fit(X_tr, y_tr_clf)
            prob_gbdt = clf_gbdt.predict_proba(X_val)[:, 1]
            
            clf_lin = LogisticRegression(C=0.5, penalty='l2', random_state=42 + run_idx)
            clf_lin.fit(X_tr_imp, y_tr_clf)
            prob_lin = clf_lin.predict_proba(X_val_imp)[:, 1]
            prob = 0.5 * prob_gbdt + 0.5 * prob_lin
            pred_bin = (prob >= 0.5).astype(int)
            
            reg_gbdt = HistGradientBoostingRegressor(
                max_depth=2, max_iter=35, learning_rate=0.05, min_samples_leaf=4, l2_regularization=1.5, random_state=42 + run_idx
            )
            reg_gbdt.fit(X_tr, y_tr_reg)
            pred_reg_gbdt = reg_gbdt.predict(X_val)
            
            reg_lin = Ridge(alpha=2.0, random_state=42 + run_idx)
            reg_lin.fit(X_tr_imp, y_tr_reg)
            pred_reg_lin = reg_lin.predict(X_val_imp)
            pred_reg = 0.5 * pred_reg_gbdt + 0.5 * pred_reg_lin
            
            fold_records[exp_name]['auroc'].append(roc_auc_score(y_val_clf, prob))
            fold_records[exp_name]['f1'].append(f1_score(y_val_clf, pred_bin, zero_division=0))
            fold_records[exp_name]['r2'].append(r2_score(y_val_reg, pred_reg))
            fold_records[exp_name]['mae'].append(mean_absolute_error(y_val_reg, pred_reg))
            fold_records[exp_name]['rmse'].append(np.sqrt(np.mean((y_val_reg - pred_reg)**2)))
            
    # Key Comparisons to Test
    comparisons = [
        {
            "pair": ("E1: Baseline VA only", "E2: Clinical Only (7 features)"),
            "name": "E1 (VA Baseline) vs E2 (Full Clinical)",
            "metrics": ['auroc', 'f1', 'mae', 'r2']
        },
        {
            "pair": ("E2: Clinical Only (7 features)", "E6: Clinical + Vision Embeddings"),
            "name": "E2 (Clinical) vs E6 (Clinical + Siamese Vision)",
            "metrics": ['mae', 'r2', 'auroc']
        },
        {
            "pair": ("E2: Clinical Only (7 features)", "E5: Clinical + Morphometry"),
            "name": "E2 (Clinical) vs E5 (Clinical + Morphometry)",
            "metrics": ['auroc', 'mae', 'r2']
        },
        {
            "pair": ("E2: Clinical Only (7 features)", "E7: Full Multimodal Fusion"),
            "name": "E2 (Clinical) vs E7 (Full Multimodal)",
            "metrics": ['auroc', 'mae', 'r2']
        },
        {
            "pair": ("E3: Vision Embeddings Only", "E4: Morphometry Only"),
            "name": "E3 (Vision Embeddings) vs E4 (Morphometry)",
            "metrics": ['auroc', 'mae']
        }
    ]
    
    test_results = []
    
    for comp in comparisons:
        mA, mB = comp['pair']
        for metric in comp['metrics']:
            arrA = np.array(fold_records[mA][metric])
            arrB = np.array(fold_records[mB][metric])
            diff = arrB - arrA  # Difference: Model B - Model A
            
            meanA = np.mean(arrA)
            stdA = np.std(arrA, ddof=1)
            meanB = np.mean(arrB)
            stdB = np.std(arrB, ddof=1)
            mean_diff = np.mean(diff)
            std_diff = np.std(diff, ddof=1)
            
            # Wilcoxon signed-rank test
            try:
                w_stat, p_wilcoxon = stats.wilcoxon(diff, zero_method='wilcox', correction=True)
            except Exception:
                w_stat, p_wilcoxon = 0.0, 1.0
                
            # Standard paired t-test
            t_stat, p_paired_t = stats.ttest_rel(arrB, arrA)
            
            # Nadeau-Bengio corrected resampled t-test
            t_nb, p_nb = corrected_resampled_ttest(diff, n_train=83, n_val=21, n_runs=25)
            
            # 95% Bootstrap CI on difference
            ci_low, ci_high = bootstrap_paired_diff_ci(diff, n_bootstraps=2000, ci=95)
            
            # Formulate scientific interpretation
            is_sig_wilcoxon = p_wilcoxon < 0.05
            is_sig_nb = p_nb < 0.05
            
            interpretation = "Statistically Significant (p < 0.05)" if is_sig_wilcoxon else "Non-Significant / Marginal Trend (p >= 0.05)"
            
            test_results.append({
                "Comparison": comp['name'],
                "Metric": metric.upper(),
                "Model_A_Mean_Std": f"{meanA:.4f} ± {stdA:.4f}",
                "Model_B_Mean_Std": f"{meanB:.4f} ± {stdB:.4f}",
                "Mean_Diff (B - A)": f"{mean_diff:+.4f} ± {std_diff:.4f}",
                "95%_CI_Diff": f"[{ci_low:+.4f}, {ci_high:+.4f}]",
                "Wilcoxon_p": float(p_wilcoxon),
                "Paired_t_p": float(p_paired_t),
                "Nadeau_Bengio_p": float(p_nb),
                "Interpretation": interpretation
            })
            
    res_df = pd.DataFrame(test_results)
    print("\n" + "="*95)
    print("STATISTICAL SIGNIFICANCE TESTING SUMMARY TABLE (25 PAIRED OUTER FOLDS)")
    print("="*95)
    print(res_df.to_string(index=False))
    
    os.makedirs("results", exist_ok=True)
    res_df.to_csv("results/significance_testing_results.csv", index=False)
    with open("results/significance_testing_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
        
    print("\nSaved significance test results to results/significance_testing_results.csv and .json")
    return res_df

if __name__ == "__main__":
    run_significance_tests()
