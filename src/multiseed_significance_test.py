"""
Multi-Seed Hierarchical & Repeated-Measures Significance Testing Module
Evaluates E1 (Baseline VA alone) vs E2 (7 Clinical Features) across 125 total outer folds 
(5 independent random seeds × 5 repeats × 5 folds) to rigorously test the Baseline VA Dominance hypothesis.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from src.data_preprocessing import load_and_clean_data, generate_repeated_cv_splits
from src.multimodal_evaluator import fit_predict_ensemble, CLINICAL_COLS

def run_multiseed_significance_test(data_path="."):
    print("="*85)
    print("RUNNING MULTI-SEED HIERARCHICAL SIGNIFICANCE TEST (125 TOTAL FOLDS)")
    print("="*85)
    
    df = load_and_clean_data(data_path)
    dev_mask = df['original_split'] != 'test'
    df_dev = df[dev_mask].reset_index(drop=True)
    
    seeds = [42, 123, 456, 789, 2026]
    imputer = SimpleImputer(strategy='median')
    scaler_tab = StandardScaler()
    
    fold_level_records = []
    seed_level_summaries = []
    
    for s_idx, s in enumerate(seeds):
        _, _, seed_folds = generate_repeated_cv_splits(df, n_splits=5, n_repeats=5, random_state=s)
        s_e1_auc, s_e2_auc = [], []
        s_e1_mae, s_e2_mae = [], []
        
        for f in seed_folds:
            run_idx = f['run_idx']
            tr_idx = f['train_indices']
            val_idx = f['val_indices']
            
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
            
            auc_e1 = roc_auc_score(y_val_clf, prob_e1)
            auc_e2 = roc_auc_score(y_val_clf, prob_e2)
            mae_e1 = mean_absolute_error(y_val_reg, pred_reg_e1)
            mae_e2 = mean_absolute_error(y_val_reg, pred_reg_e2)
            
            s_e1_auc.append(auc_e1)
            s_e2_auc.append(auc_e2)
            s_e1_mae.append(mae_e1)
            s_e2_mae.append(mae_e2)
            
            fold_level_records.append({
                'seed': s,
                'run_idx': run_idx,
                'e1_auc': auc_e1,
                'e2_auc': auc_e2,
                'auc_diff (e1 - e2)': auc_e1 - auc_e2,
                'e1_mae': mae_e1,
                'e2_mae': mae_e2,
                'mae_diff (e1 - e2)': mae_e1 - mae_e2
            })
            
        seed_level_summaries.append({
            'seed': s,
            'mean_e1_auc': np.mean(s_e1_auc)*100,
            'mean_e2_auc': np.mean(s_e2_auc)*100,
            'auc_gap': (np.mean(s_e1_auc) - np.mean(s_e2_auc))*100,
            'mean_e1_mae': np.mean(s_e1_mae),
            'mean_e2_mae': np.mean(s_e2_mae),
            'mae_gap': np.mean(s_e1_mae) - np.mean(s_e2_mae)
        })
        
    df_folds = pd.DataFrame(fold_level_records)
    df_seeds = pd.DataFrame(seed_level_summaries)
    
    print("\n--- Seed-Level Aggregated Benchmarks (5 Independent Seeds) ---")
    print(df_seeds.to_string(index=False))
    
    # 1. Non-parametric Sign Test on Seed-Level Means
    # Under Null H0: P(E1 > E2) = 0.5. With 5/5 positive:
    # Binomial p-value = (0.5)^5 = 0.03125 (one-tailed) or 0.0625 (two-tailed)
    n_seeds = len(seeds)
    n_pos = sum(df_seeds['auc_gap'] > 0)
    binom_p = stats.binomtest(n_pos, n_seeds, p=0.5, alternative='two-sided').pvalue
    
    # 2. Seed-level Wilcoxon Signed-Rank Test (n=5)
    w_stat_seed, wilcox_p_seed = stats.wilcoxon(df_seeds['mean_e1_auc'], df_seeds['mean_e2_auc'])
    
    # 3. Seed-level Paired t-test (n=5)
    t_stat_seed, t_p_seed = stats.ttest_rel(df_seeds['mean_e1_auc'], df_seeds['mean_e2_auc'])
    
    # 4. Pooled 125-Fold Paired Tests
    all_auc_diff = df_folds['auc_diff (e1 - e2)'].values
    all_mae_diff = df_folds['mae_diff (e1 - e2)'].values
    
    w_stat_125, wilcox_p_125 = stats.wilcoxon(all_auc_diff)
    t_stat_125, t_p_125 = stats.ttest_1samp(all_auc_diff, 0)
    
    # 95% Bootstrap CI across 125 folds
    rng = np.random.RandomState(42)
    boot_diffs = [np.mean(rng.choice(all_auc_diff, size=len(all_auc_diff), replace=True)) for _ in range(5000)]
    ci_low_125 = np.percentile(boot_diffs, 2.5) * 100
    ci_high_125 = np.percentile(boot_diffs, 97.5) * 100
    
    print("\n" + "="*85)
    print("MULTI-SEED STATISTICAL TEST RESULTS SUMMARY")
    print("="*85)
    print(f"1. Seed-Level Directional Sign Test (5/5 positive): Two-sided p = {binom_p:.4f} (One-sided p = {binom_p/2:.4f})")
    print(f"2. Seed-Level Paired t-test (n=5 independent seeds): t = {t_stat_seed:.3f}, Two-sided p = {t_p_seed:.4f}")
    print(f"3. Seed-Level Wilcoxon Signed-Rank Test (n=5): W = {w_stat_seed:.1f}, Two-sided p = {wilcox_p_seed:.4f}")
    print(f"4. Pooled 125-Fold Wilcoxon Test: W = {w_stat_125:.1f}, Two-sided p = {wilcox_p_125:.4e}")
    print(f"5. Pooled 125-Fold 95% Bootstrap CI on AUROC Gap: [{ci_low_125:+.2f}%, {ci_high_125:+.2f}%] (Mean: {np.mean(all_auc_diff)*100:+.2f}%)")
    
    results = {
        'seed_summaries': seed_level_summaries,
        'seed_level_ttest_p': float(t_p_seed),
        'seed_level_wilcoxon_p': float(wilcox_p_seed),
        'seed_level_binom_p': float(binom_p),
        'pooled_125_wilcoxon_p': float(wilcox_p_125),
        'pooled_125_ci': [float(ci_low_125), float(ci_high_125)],
        'pooled_125_mean_gap': float(np.mean(all_auc_diff)*100)
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/multiseed_significance_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nSaved multi-seed significance results to results/multiseed_significance_results.json")
    return results

if __name__ == "__main__":
    run_multiseed_significance_test()
