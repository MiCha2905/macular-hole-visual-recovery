"""
Data Preprocessing & Target Engineering Module
Macular Hole Visual Outcome Prediction Pipeline
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

def load_and_clean_data(base_dir="."):
    """
    Loads clinical_data.csv from train, val, and test splits (121 benchmark patients),
    applies data cleaning rules, verifies pre-op OCT images, and constructs targets.
    """
    splits = ['train', 'val', 'test']
    dfs = []
    
    for split in splits:
        csv_path = os.path.join(base_dir, split, 'clinical_data.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing clinical data file: {csv_path}")
        
        df_split = pd.read_csv(csv_path)
        df_split['original_split'] = split
        
        # Verify OCT image presence for each patient at baseline
        oct_dir = os.path.join(base_dir, split, 'octs')
        h_paths = []
        v_paths = []
        
        for pid in df_split['id']:
            h_file = os.path.join(oct_dir, f"{pid}_baseline_H.tiff")
            v_file = os.path.join(oct_dir, f"{pid}_baseline_V.tiff")
            h_paths.append(h_file if os.path.exists(h_file) else None)
            v_paths.append(v_file if os.path.exists(v_file) else None)
            
        df_split['oct_baseline_H_path'] = h_paths
        df_split['oct_baseline_V_path'] = v_paths
        dfs.append(df_split)
        
    df = pd.concat(dfs, ignore_index=True)
    
    # 1. Clean clinical features: Sentinel -9 -> NaN
    sentinel_cols = ['mh_duration', 'elevated_edge', 'mh_size']
    for col in sentinel_cols:
        if col in df.columns:
            df[col] = df[col].replace(-9, np.nan)
            
    # 2. Re-encode sex: 1 (Male) -> 0, 2 (Female) -> 1
    df['sex_encoded'] = df['sex'].map({1: 0, 2: 1})
    
    # 3. Construct Targets
    # Primary Binary Classification: VA_gain_6mo_binary = 1 if (VA_6months - VA_baseline) >= 15 else 0
    df['delta_VA_6months'] = df['VA_6months'] - df['VA_baseline']
    df['VA_gain_6mo_binary'] = (df['delta_VA_6months'] >= 15).astype(int)
    
    # Additional ceiling indicator for error analysis
    df['is_ceiling_constrained'] = (df['VA_baseline'] > 70).astype(int)
    
    return df

def generate_repeated_cv_splits(df, n_splits=5, n_repeats=5, random_state=42, test_split='test'):
    """
    Creates:
    1. A development set (train + val: 104 patients) with 25 evaluation folds (5x5 Repeated Stratified K-Fold).
    2. A fixed held-out test set (test: 17 patients) to be evaluated once at the very end.
    """
    dev_mask = df['original_split'] != test_split
    df_dev = df[dev_mask].copy().reset_index(drop=True)
    df_test = df[~dev_mask].copy().reset_index(drop=True)
    
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    
    folds = []
    for run_idx, (train_idx, val_idx) in enumerate(rskf.split(df_dev, df_dev['VA_gain_6mo_binary'])):
        folds.append({
            'run_idx': run_idx,
            'repeat': run_idx // n_splits,
            'fold': run_idx % n_splits,
            'train_indices': train_idx.tolist(),
            'val_indices': val_idx.tolist()
        })
        
    return df_dev, df_test, folds

if __name__ == "__main__":
    df = load_and_clean_data(".")
    print(f"Loaded {len(df)} total benchmark patients.")
    print(f"Pre-op OCT H-scans present: {df['oct_baseline_H_path'].notnull().sum()} / {len(df)}")
    print(f"Pre-op OCT V-scans present: {df['oct_baseline_V_path'].notnull().sum()} / {len(df)}")
    print(f"\nTarget Class Balance (VA gain >= 15 letters at 6 months):")
    print(df['VA_gain_6mo_binary'].value_counts(normalize=True).round(3))
    print(f"\nDelta VA Statistics (6 months - baseline):")
    print(df['delta_VA_6months'].describe().round(2))
    
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/processed_clinical.csv", index=False)
    print("Saved processed data to data/processed_clinical.csv")
