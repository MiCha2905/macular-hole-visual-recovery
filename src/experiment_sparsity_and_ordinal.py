"""
Directions #4 & #5: L1 Sparsity / ElasticNet Regularization & 3-Tier Ordinal Reformulation Engine
Evaluates whether feature sparsity prevents multimodal degradation on classification and
tests 3-tier ordinal clinical staging against binary and continuous targets.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import kendalltau
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, brier_score_loss,
    cohen_kappa_score, r2_score, mean_absolute_error
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel

from src.data_preprocessing import load_and_clean_data, generate_repeated_cv_splits
from src.extract_morphometry import extract_all_morphometry

CLINICAL_COLS = ['age', 'sex_encoded', 'pseudophakic', 'mh_duration', 'elevated_edge', 'mh_size', 'VA_baseline']
MORPH_COLS = ['mld_mean', 'bd_mean', 'height_mean', 'mhi_mean', 'thi_mean', 'dhi_mean']

EXPERIMENTS = {
    "E1: Baseline VA only": lambda c, m, p, df, idx: df.iloc[idx][['VA_baseline']].values,
    "E2: Clinical Only (7 features)": lambda c, m, p, df, idx: c,
    "E5: Clinical + Morphometry": lambda c, m, p, df, idx: np.hstack([c, m]),
    "E6: Clinical + Vision (PCA-8)": lambda c, m, p, df, idx: np.hstack([c, p]),
    "E7: Full Multimodal Fusion": lambda c, m, p, df, idx: np.hstack([c, m, p])
}


# ==============================================================================
# Direction #5: Ordinal Model Implementations
# ==============================================================================

class ProportionalOddsOrdinalLogistic:
    """
    Cumulative link / proportional odds ordinal logistic regression model:
    P(Y <= k | X) = sigmoid(theta_k - w^T X)
    Enforces theta_0 < theta_1 < ... < theta_{K-2}
    """
    def __init__(self, l2_reg=1.0, max_iter=200):
        self.l2_reg = l2_reg
        self.max_iter = max_iter
        self.coef_ = None
        self.cutpoints_ = None
        self.n_classes_ = 3

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.n_classes_ = len(np.unique(y))
        n_cutpoints = self.n_classes_ - 1
        
        # Initial guess: w = 0, cutpoints evenly spaced
        init_params = np.zeros(n_features + n_cutpoints)
        init_params[n_features] = -0.5
        init_params[n_features + 1] = 0.5
        
        def loss_and_grad(params):
            w = params[:n_features]
            # Parameterize cutpoints with exp delta to enforce monotonicity: theta_1 = theta_0 + exp(delta)
            theta0 = params[n_features]
            delta = params[n_features + 1]
            theta1 = theta0 + np.exp(delta)
            thetas = [theta0, theta1]
            
            # Linear score
            eta = X @ w # (n_samples,)
            
            # Cumulative probabilities
            # P(Y <= 0) = sigmoid(theta0 - eta)
            # P(Y <= 1) = sigmoid(theta1 - eta)
            # P(Y <= 2) = 1.0
            
            # Compute loss
            eps = 1e-12
            loss = 0.0
            p0 = 1.0 / (1.0 + np.exp(-(theta0 - eta)))
            p1 = 1.0 / (1.0 + np.exp(-(theta1 - eta)))
            
            # Category probabilities
            prob_cat0 = np.clip(p0, eps, 1 - eps)
            prob_cat1 = np.clip(p1 - p0, eps, 1 - eps)
            prob_cat2 = np.clip(1.0 - p1, eps, 1 - eps)
            
            # NLL
            nll0 = -np.sum(np.log(prob_cat0[y == 0]))
            nll1 = -np.sum(np.log(prob_cat1[y == 1]))
            nll2 = -np.sum(np.log(prob_cat2[y == 2]))
            
            l2_penalty = 0.5 * self.l2_reg * np.sum(w ** 2)
            total_loss = (nll0 + nll1 + nll2) / n_samples + l2_penalty
            return total_loss

        res = minimize(loss_and_grad, init_params, method='L-BFGS-B', options={'maxiter': self.max_iter})
        w = res.x[:n_features]
        theta0 = res.x[n_features]
        delta = res.x[n_features + 1]
        theta1 = theta0 + np.exp(delta)
        
        self.coef_ = w
        self.cutpoints_ = np.array([theta0, theta1])
        return self

    def predict_proba(self, X):
        eta = X @ self.coef_
        theta0, theta1 = self.cutpoints_
        p0 = 1.0 / (1.0 + np.exp(-(theta0 - eta)))
        p1 = 1.0 / (1.0 + np.exp(-(theta1 - eta)))
        
        prob_cat0 = p0
        prob_cat1 = np.maximum(0.0, p1 - p0)
        prob_cat2 = np.maximum(0.0, 1.0 - p1)
        
        probs = np.column_stack([prob_cat0, prob_cat1, prob_cat2])
        # Normalize rows to sum to 1
        row_sums = probs.sum(axis=1, keepdims=True) + 1e-12
        return probs / row_sums

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


class FrankHallOrdinalClassifier:
    """
    Frank & Hall Ordinal Classification method:
    Trains K-1 binary classifiers for P(Y > 0) and P(Y > 1).
    """
    def __init__(self, penalty='l1', C=0.5, random_state=42):
        self.penalty = penalty
        self.C = C
        self.random_state = random_state
        self.clf1 = LogisticRegression(penalty=penalty, C=C, solver='liblinear' if penalty=='l1' else 'lbfgs', random_state=random_state)
        self.clf2 = LogisticRegression(penalty=penalty, C=C, solver='liblinear' if penalty=='l1' else 'lbfgs', random_state=random_state)

    def fit(self, X, y):
        # Binary task 1: Y > 0 (i.e. Class 1 or 2 vs Class 0)
        y1 = (y > 0).astype(int)
        self.clf1.fit(X, y1)
        
        # Binary task 2: Y > 1 (i.e. Class 2 vs Class 0 or 1)
        y2 = (y > 1).astype(int)
        self.clf2.fit(X, y2)
        return self

    def predict_proba(self, X):
        # P(Y > 0)
        p_gt_0 = self.clf1.predict_proba(X)[:, 1]
        # P(Y > 1)
        p_gt_1 = self.clf2.predict_proba(X)[:, 1]
        
        # Enforce monotonicity P(Y > 1) <= P(Y > 0)
        p_gt_1 = np.minimum(p_gt_1, p_gt_0)
        
        p_class0 = 1.0 - p_gt_0
        p_class1 = p_gt_0 - p_gt_1
        p_class2 = p_gt_1
        
        probs = np.column_stack([p_class0, p_class1, p_class2])
        row_sums = probs.sum(axis=1, keepdims=True) + 1e-12
        return np.clip(probs / row_sums, 0.0, 1.0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# ==============================================================================
# Direction #4: L1-Regularized / Sparse Binary Classification Evaluator
# ==============================================================================

def evaluate_l1_sparsity_sweep(df_dev, dev_raw_embs, folds, C_values=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]):
    """
    Evaluates L1-penalized Logistic Regression across different C strengths
    to determine if optimal sparsity improves E2-E7 AUROC relative to baseline VA.
    """
    print("\n" + "="*80)
    print("DIRECTION #4: L1-REGULARIZED SPARSITY SWEEP ACROSS FEATURE TIERS")
    print("="*80)
    
    results = {}
    
    for exp_name in EXPERIMENTS.keys():
        results[exp_name] = {}
        for C in C_values:
            results[exp_name][C] = {'aurocs': [], 'non_zero_coeffs': []}
            
    for C in C_values:
        imputer = SimpleImputer(strategy='median')
        scaler_tab = StandardScaler()
        
        for fold_info in folds:
            run_idx = fold_info['run_idx']
            tr_idx = fold_info['train_indices']
            val_idx = fold_info['val_indices']
            
            y_tr = df_dev.iloc[tr_idx]['VA_gain_6mo_binary'].values
            y_val = df_dev.iloc[val_idx]['VA_gain_6mo_binary'].values
            
            # PCA for embeddings
            pca = PCA(n_components=8, random_state=42)
            scaler_emb = StandardScaler()
            tr_emb_scaled = scaler_emb.fit_transform(dev_raw_embs[tr_idx])
            val_emb_scaled = scaler_emb.transform(dev_raw_embs[val_idx])
            tr_pca = pca.fit_transform(tr_emb_scaled)
            val_pca = pca.transform(val_emb_scaled)
            
            tr_clin = df_dev.iloc[tr_idx][CLINICAL_COLS].values
            val_clin = df_dev.iloc[val_idx][CLINICAL_COLS].values
            
            tr_morph = df_dev.iloc[tr_idx][MORPH_COLS].values
            val_morph = df_dev.iloc[val_idx][MORPH_COLS].values
            
            for exp_name, feat_fn in EXPERIMENTS.items():
                X_tr_raw = feat_fn(tr_clin, tr_morph, tr_pca, df_dev, tr_idx)
                X_val_raw = feat_fn(val_clin, val_morph, val_pca, df_dev, val_idx)
                
                # Scale & Impute
                X_tr = scaler_tab.fit_transform(imputer.fit_transform(X_tr_raw))
                X_val = scaler_tab.transform(imputer.transform(X_val_raw))
                
                # Fit L1 Logistic Regression
                clf = LogisticRegression(penalty='l1', C=C, solver='liblinear', random_state=42 + run_idx)
                clf.fit(X_tr, y_tr)
                prob = clf.predict_proba(X_val)[:, 1]
                
                auroc = roc_auc_score(y_val, prob)
                non_zero = np.sum(np.abs(clf.coef_) > 1e-4)
                
                results[exp_name][C]['aurocs'].append(auroc)
                results[exp_name][C]['non_zero_coeffs'].append(non_zero)
                
    # Summary Table
    print(f"\n{'Experiment Tier':<35} | " + " | ".join([f"C={c:4.2f}" for c in C_values]))
    print("-" * 95)
    
    summary_data = []
    for exp_name in EXPERIMENTS.keys():
        row_str = f"{exp_name:<35} | "
        c_means = []
        for C in C_values:
            mean_auc = np.mean(results[exp_name][C]['aurocs']) * 100
            sd_auc = np.std(results[exp_name][C]['aurocs']) * 100
            mean_nz = np.mean(results[exp_name][C]['non_zero_coeffs'])
            c_means.append(f"{mean_auc:4.1f}% ({mean_nz:.1f})")
        print(row_str + " | ".join(c_means))
        summary_data.append({
            'experiment': exp_name,
            **{f"C_{c}_mean_auroc": float(np.mean(results[exp_name][c]['aurocs'])) for c in C_values},
            **{f"C_{c}_mean_nonzero": float(np.mean(results[exp_name][c]['non_zero_coeffs'])) for c in C_values}
        })
        
    return results, summary_data


# ==============================================================================
# Direction #5: 3-Tier Ordinal Evaluation Engine
# ==============================================================================

def construct_ordinal_targets(df):
    """
    Constructs 3-Tier Ordinal Target:
    Class 0 (<5 letters gain)
    Class 1 (5 to 14 letters gain)
    Class 2 (>=15 letters gain)
    """
    delta = df['delta_VA_6months'].values
    ordinal_target = np.zeros(len(df), dtype=int)
    ordinal_target[(delta >= 5) & (delta < 15)] = 1
    ordinal_target[delta >= 15] = 2
    return ordinal_target

def evaluate_ordinal_models(df_dev, dev_raw_embs, folds):
    """
    Evaluates Ordinal Models across the 25-fold CV protocol.
    """
    print("\n" + "="*80)
    print("DIRECTION #5: 3-TIER ORDINAL OUTCOME REFORMULATION BENCHMARK")
    print("="*80)
    
    df_dev = df_dev.copy()
    df_dev['ordinal_target'] = construct_ordinal_targets(df_dev)
    
    print("Development Set Ordinal Class Distribution:")
    counts = df_dev['ordinal_target'].value_counts().sort_index()
    for cat, name in zip([0, 1, 2], ['Class 0 (<5 letters)', 'Class 1 (5-14 letters)', 'Class 2 (>=15 letters)']):
        print(f"  {name}: {counts.get(cat, 0)} patients ({counts.get(cat, 0)/len(df_dev)*100:.1f}%)")
        
    models_to_test = {
        "Frank-Hall L1 Sparse (C=0.2)": lambda run_idx: FrankHallOrdinalClassifier(penalty='l1', C=0.2, random_state=42 + run_idx),
        "Frank-Hall L2 (C=0.5)": lambda run_idx: FrankHallOrdinalClassifier(penalty='l2', C=0.5, random_state=42 + run_idx),
        "Proportional Odds Logistic (L2=1.0)": lambda run_idx: ProportionalOddsOrdinalLogistic(l2_reg=1.0),
        "Ordinal Discretized GBDT Regressor": "gbdt_reg"
    }
    
    ordinal_results = {}
    for model_name in models_to_test.keys():
        ordinal_results[model_name] = {}
        for exp_name in EXPERIMENTS.keys():
            ordinal_results[model_name][exp_name] = {
                'qwk': [], 'kendall_tau': [], 'accuracy': [], 'adjacent_acc': [], 'macro_auroc': []
            }
            
    imputer = SimpleImputer(strategy='median')
    scaler_tab = StandardScaler()
    
    for fold_info in folds:
        run_idx = fold_info['run_idx']
        tr_idx = fold_info['train_indices']
        val_idx = fold_info['val_indices']
        
        y_tr_ord = df_dev.iloc[tr_idx]['ordinal_target'].values
        y_val_ord = df_dev.iloc[val_idx]['ordinal_target'].values
        y_tr_reg = df_dev.iloc[tr_idx]['delta_VA_6months'].values
        
        # PCA for embeddings
        pca = PCA(n_components=8, random_state=42)
        scaler_emb = StandardScaler()
        tr_emb_scaled = scaler_emb.fit_transform(dev_raw_embs[tr_idx])
        val_emb_scaled = scaler_emb.transform(dev_raw_embs[val_idx])
        tr_pca = pca.fit_transform(tr_emb_scaled)
        val_pca = pca.transform(val_emb_scaled)
        
        tr_clin = df_dev.iloc[tr_idx][CLINICAL_COLS].values
        val_clin = df_dev.iloc[val_idx][CLINICAL_COLS].values
        
        tr_morph = df_dev.iloc[tr_idx][MORPH_COLS].values
        val_morph = df_dev.iloc[val_idx][MORPH_COLS].values
        
        for exp_name, feat_fn in EXPERIMENTS.items():
            X_tr_raw = feat_fn(tr_clin, tr_morph, tr_pca, df_dev, tr_idx)
            X_val_raw = feat_fn(val_clin, val_morph, val_pca, df_dev, val_idx)
            
            X_tr = scaler_tab.fit_transform(imputer.fit_transform(X_tr_raw))
            X_val = scaler_tab.transform(imputer.transform(X_val_raw))
            
            for m_name, model_fn in models_to_test.items():
                if m_name == "Ordinal Discretized GBDT Regressor":
                    # Fit GBDT regressor and map continuous predictions to ordinal tiers: <5 -> 0, 5-15 -> 1, >=15 -> 2
                    reg = HistGradientBoostingRegressor(max_depth=2, max_iter=35, learning_rate=0.05, min_samples_leaf=4, random_state=42 + run_idx)
                    reg.fit(X_tr_raw, y_tr_reg)
                    pred_cont = reg.predict(X_val_raw)
                    
                    pred_ord = np.zeros_like(pred_cont, dtype=int)
                    pred_ord[(pred_cont >= 5.0) & (pred_cont < 15.0)] = 1
                    pred_ord[pred_cont >= 15.0] = 2
                    
                    # For ROC: distance-based pseudo-probs
                    prob_ord = np.zeros((len(pred_cont), 3))
                    for i, val in enumerate(pred_cont):
                        prob_ord[i, 0] = max(0.01, 1.0 - (val / 10.0)) if val < 10 else 0.01
                        prob_ord[i, 1] = max(0.01, 1.0 - abs(val - 10.0)/10.0)
                        prob_ord[i, 2] = max(0.01, (val - 10.0)/10.0) if val > 10 else 0.01
                        prob_ord[i] /= prob_ord[i].sum()
                else:
                    clf = model_fn(run_idx)
                    clf.fit(X_tr, y_tr_ord)
                    prob_ord = clf.predict_proba(X_val)
                    pred_ord = clf.predict(X_val)
                    
                # Calculate Ordinal Metrics
                qwk = cohen_kappa_score(y_val_ord, pred_ord, weights='quadratic')
                tau, _ = kendalltau(y_val_ord, pred_ord)
                if np.isnan(tau): tau = 0.0
                acc = accuracy_score(y_val_ord, pred_ord)
                adj_acc = np.mean(np.abs(y_val_ord - pred_ord) <= 1)
                
                try:
                    macro_auc = roc_auc_score(pd.get_dummies(y_val_ord).reindex(columns=[0, 1, 2], fill_value=0), prob_ord, multi_class='ovr', average='macro')
                except Exception:
                    macro_auc = np.nan
                    
                ordinal_results[m_name][exp_name]['qwk'].append(qwk)
                ordinal_results[m_name][exp_name]['kendall_tau'].append(tau)
                ordinal_results[m_name][exp_name]['accuracy'].append(acc)
                ordinal_results[m_name][exp_name]['adjacent_acc'].append(adj_acc)
                ordinal_results[m_name][exp_name]['macro_auroc'].append(macro_auc)
                
    # Print Ordinal Results Table
    print("\n" + "="*80)
    print("ORDINAL BENCHMARK RESULTS (Quadratic Weighted Kappa / Kendall's Tau / Exact Acc / Adj Acc)")
    print("="*80)
    
    summary_ordinal = []
    for m_name in models_to_test.keys():
        print(f"\n--- Model: {m_name} ---")
        print(f"{'Feature Tier':<35} | {'QWK':<15} | {'Kendall Tau':<15} | {'Exact Acc':<12} | {'Adj Acc (±1)':<12}")
        print("-" * 95)
        for exp_name in EXPERIMENTS.keys():
            qwk_m = np.mean(ordinal_results[m_name][exp_name]['qwk'])
            qwk_sd = np.std(ordinal_results[m_name][exp_name]['qwk'])
            tau_m = np.mean(ordinal_results[m_name][exp_name]['kendall_tau'])
            tau_sd = np.std(ordinal_results[m_name][exp_name]['kendall_tau'])
            acc_m = np.mean(ordinal_results[m_name][exp_name]['accuracy']) * 100
            adj_m = np.mean(ordinal_results[m_name][exp_name]['adjacent_acc']) * 100
            
            print(f"{exp_name:<35} | {qwk_m:6.3f} ± {qwk_sd:5.3f} | {tau_m:6.3f} ± {tau_sd:5.3f} | {acc_m:5.1f}%      | {adj_m:5.1f}%")
            summary_ordinal.append({
                'model': m_name,
                'tier': exp_name,
                'qwk_mean': float(qwk_m),
                'qwk_sd': float(qwk_sd),
                'tau_mean': float(tau_m),
                'tau_sd': float(tau_sd),
                'acc_mean': float(acc_m),
                'adj_acc_mean': float(adj_m)
            })
            
    return ordinal_results, summary_ordinal


# ==============================================================================
# Held-Out Test Set Verification
# ==============================================================================

def evaluate_on_held_out_test(df_dev, df_test, dev_raw_embs, test_raw_embs):
    """
    Evaluates best L1 sparse model and Ordinal model on the locked 17-patient held-out test set.
    """
    print("\n" + "="*80)
    print("HELD-OUT TEST SET (N=17) EVALUATION: L1 SPARSITY & ORDINAL REFORMULATION")
    print("="*80)
    
    df_dev = df_dev.copy()
    df_test = df_test.copy()
    
    df_dev['ordinal_target'] = construct_ordinal_targets(df_dev)
    df_test['ordinal_target'] = construct_ordinal_targets(df_test)
    
    y_dev_clf = df_dev['VA_gain_6mo_binary'].values
    y_test_clf = df_test['VA_gain_6mo_binary'].values
    y_dev_ord = df_dev['ordinal_target'].values
    y_test_ord = df_test['ordinal_target'].values
    
    pca = PCA(n_components=8, random_state=42)
    scaler_emb = StandardScaler()
    dev_pca = pca.fit_transform(scaler_emb.fit_transform(dev_raw_embs))
    test_pca = pca.transform(scaler_emb.transform(test_raw_embs))
    
    dev_clin = df_dev[CLINICAL_COLS].values
    test_clin = df_test[CLINICAL_COLS].values
    
    dev_morph = df_dev[MORPH_COLS].values
    test_morph = df_test[MORPH_COLS].values
    
    imputer = SimpleImputer(strategy='median')
    scaler_tab = StandardScaler()
    
    test_results = []
    
    for exp_name, feat_fn in EXPERIMENTS.items():
        X_dev_raw = feat_fn(dev_clin, dev_morph, dev_pca, df_dev, range(len(df_dev)))
        X_test_raw = feat_fn(test_clin, test_morph, test_pca, df_test, range(len(df_test)))
        
        X_dev = scaler_tab.fit_transform(imputer.fit_transform(X_dev_raw))
        X_test = scaler_tab.transform(imputer.transform(X_test_raw))
        
        # 1. L2 Baseline Logistic (C=0.5)
        clf_l2 = LogisticRegression(penalty='l2', C=0.5, random_state=42)
        clf_l2.fit(X_dev, y_dev_clf)
        auc_l2 = roc_auc_score(y_test_clf, clf_l2.predict_proba(X_test)[:, 1])
        
        # 2. L1 Sparse Logistic (C=0.2)
        clf_l1 = LogisticRegression(penalty='l1', C=0.2, solver='liblinear', random_state=42)
        clf_l1.fit(X_dev, y_dev_clf)
        auc_l1 = roc_auc_score(y_test_clf, clf_l1.predict_proba(X_test)[:, 1])
        nz_l1 = int(np.sum(np.abs(clf_l1.coef_) > 1e-4))
        
        # 3. Ordinal Frank-Hall (L1, C=0.2)
        ord_fh = FrankHallOrdinalClassifier(penalty='l1', C=0.2, random_state=42)
        ord_fh.fit(X_dev, y_dev_ord)
        pred_ord = ord_fh.predict(X_test)
        qwk = cohen_kappa_score(y_test_ord, pred_ord, weights='quadratic')
        acc = accuracy_score(y_test_ord, pred_ord)
        adj_acc = np.mean(np.abs(y_test_ord - pred_ord) <= 1)
        
        test_results.append({
            'tier': exp_name,
            'l2_test_auroc': float(auc_l2),
            'l1_test_auroc': float(auc_l1),
            'l1_nonzero_features': nz_l1,
            'ordinal_qwk': float(qwk),
            'ordinal_acc': float(acc),
            'ordinal_adj_acc': float(adj_acc)
        })
        
        print(f"{exp_name:<35} | L2 AUC: {auc_l2*100:4.1f}% | L1 AUC: {auc_l1*100:4.1f}% (nz={nz_l1}) | Ordinal QWK: {qwk:5.3f} | AdjAcc: {adj_acc*100:4.1f}%")
        
    return test_results


def main():
    base_dir = "."
    df = load_and_clean_data(base_dir)
    df_dev, df_test, folds = generate_repeated_cv_splits(df, n_splits=5, n_repeats=5, random_state=42)
    
    # Load raw vision embeddings
    raw_embs = np.load("data/vision_embeddings_raw.npy")
    dev_mask = (df['original_split'] != 'test').values
    dev_raw_embs = raw_embs[dev_mask]
    test_raw_embs = raw_embs[~dev_mask]
    
    # Extract / ensure morphometry
    morph_df = extract_all_morphometry(df)
    # Merge morphometry if not already present
    if 'mld_mean' not in df.columns:
        df = df.merge(morph_df, on='id', how='left')
        df_dev = df_dev.merge(morph_df, on='id', how='left')
        df_test = df_test.merge(morph_df, on='id', how='left')
    
    # Run Direction #4 (L1 Sparsity Sweep)
    l1_results, l1_summary = evaluate_l1_sparsity_sweep(df_dev, dev_raw_embs, folds)
    
    # Run Direction #5 (3-Tier Ordinal Benchmark)
    ord_results, ord_summary = evaluate_ordinal_models(df_dev, dev_raw_embs, folds)
    
    # Run Held-Out Test Evaluation
    test_summary = evaluate_on_held_out_test(df_dev, df_test, dev_raw_embs, test_raw_embs)
    
    # Save full results
    os.makedirs("results", exist_ok=True)
    out_payload = {
        'l1_sparsity_summary': l1_summary,
        'ordinal_benchmark_summary': ord_summary,
        'held_out_test_summary': test_summary
    }
    
    with open("results/sparsity_and_ordinal_results.json", "w") as f:
        json.dump(out_payload, f, indent=2)
        
    print("\nSaved full results to results/sparsity_and_ordinal_results.json")

if __name__ == "__main__":
    main()
