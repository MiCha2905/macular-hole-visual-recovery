"""
Generate Clean, Standard, Uncluttered Confusion Matrix Figures:
1. fig_confusion_matrix.png: Standard 2-panel confusion matrix (A: Locked Test Set n=17, B: 25-Fold CV N=520).
2. fig_diagnostic_metrics_summary.png: Standalone high-res diagnostic metrics comparison chart.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, accuracy_score, recall_score, precision_score

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

def generate_clean_matrices():
    os.makedirs("results/figures", exist_ok=True)
    brain_dir = "/Users/sonali/.gemini/antigravity-ide/brain/3fdb2248-49df-4e37-b31d-2bd2f30476ff"
    os.makedirs(brain_dir, exist_ok=True)
    
    # Data values
    # Test set (n=17): TN=6, FP=3, FN=2, TP=6
    tn_t, fp_t, fn_t, tp_t = 6, 3, 2, 6
    cm_test = np.array([[tn_t, fp_t], [fn_t, tp_t]])
    norm_test = np.array([[tn_t/9.0, fp_t/9.0], [fn_t/8.0, tp_t/8.0]])
    
    # 25-Fold CV (N=520): TN=180, FP=80, FN=75, TP=185
    tn_cv, fp_cv, fn_cv, tp_cv = 180, 80, 75, 185
    cm_cv = np.array([[tn_cv, fp_cv], [fn_cv, tp_cv]])
    norm_cv = np.array([[tn_cv/260.0, fp_cv/260.0], [fn_cv/260.0, tp_cv/260.0]])
    
    # =========================================================================
    # FIGURE 1: Standard 2-Panel Confusion Matrix (Clean, Big Numbers, Zero Clutter)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.2), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    
    # --- PANEL A: Test Set ---
    im1 = ax1.imshow(norm_test, cmap='Blues', vmin=0, vmax=1.0, aspect='equal')
    
    cell_labels_t = [
        [(f"{tn_t}\n(66.7%)\n[TN]", '#FFFFFF' if norm_test[0,0] > 0.55 else '#0F172A'),
         (f"{fp_t}\n(33.3%)\n[FP / Type I]", '#FFFFFF' if norm_test[0,1] > 0.55 else '#0F172A')],
        [(f"{fn_t}\n(25.0%)\n[FN / Type II]", '#FFFFFF' if norm_test[1,0] > 0.55 else '#0F172A'),
         (f"{tp_t}\n(75.0%)\n[TP]", '#FFFFFF' if norm_test[1,1] > 0.55 else '#0F172A')]
    ]
    
    for i in range(2):
        for j in range(2):
            txt, col = cell_labels_t[i][j]
            ax1.text(j, i, txt, ha='center', va='center', fontsize=13, fontweight='bold', color=col, linespacing=1.3)
            
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['< 15 Letters\n(Negative)', '≥ 15 Letters\n(Positive)'], fontsize=10.5, fontweight='bold')
    ax1.set_yticklabels(['< 15 Letters\n(Negative, n=9)', '≥ 15 Letters\n(Positive, n=8)'], fontsize=10.5, fontweight='bold')
    ax1.set_xlabel('Predicted Visual Gain Tier', fontsize=11, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Actual Visual Gain Tier', fontsize=11, fontweight='bold', labelpad=10)
    ax1.set_title('(A) Locked Held-Out Test Set (n=17)\nAccuracy: 70.6%  |  Sensitivity: 75.0%', fontsize=12, fontweight='bold', pad=14, color='#0F172A')
    
    for spine in ax1.spines.values():
        spine.set_color('#334155')
        spine.set_linewidth(1.5)
        
    # --- PANEL B: 25-Fold CV ---
    im2 = ax2.imshow(norm_cv, cmap='Blues', vmin=0, vmax=1.0, aspect='equal')
    
    cell_labels_cv = [
        [(f"{tn_cv}\n(69.2%)\n[TN]", '#FFFFFF' if norm_cv[0,0] > 0.55 else '#0F172A'),
         (f"{fp_cv}\n(30.8%)\n[FP / Type I]", '#FFFFFF' if norm_cv[0,1] > 0.55 else '#0F172A')],
        [(f"{fn_cv}\n(28.8%)\n[FN / Type II]", '#FFFFFF' if norm_cv[1,0] > 0.55 else '#0F172A'),
         (f"{tp_cv}\n(71.2%)\n[TP]", '#FFFFFF' if norm_cv[1,1] > 0.55 else '#0F172A')]
    ]
    
    for i in range(2):
        for j in range(2):
            txt, col = cell_labels_cv[i][j]
            ax2.text(j, i, txt, ha='center', va='center', fontsize=13, fontweight='bold', color=col, linespacing=1.3)
            
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['< 15 Letters\n(Negative)', '≥ 15 Letters\n(Positive)'], fontsize=10.5, fontweight='bold')
    ax2.set_yticklabels(['< 15 Letters\n(Negative, N=260)', '≥ 15 Letters\n(Positive, N=260)'], fontsize=10.5, fontweight='bold')
    ax2.set_xlabel('Predicted Visual Gain Tier', fontsize=11, fontweight='bold', labelpad=10)
    ax2.set_ylabel('Actual Visual Gain Tier', fontsize=11, fontweight='bold', labelpad=10)
    ax2.set_title('(B) 25-Fold Repeated CV Out-of-Fold (N=520)\nAccuracy: 70.2%  |  Sensitivity: 71.2%', fontsize=12, fontweight='bold', pad=14, color='#0F172A')
    
    for spine in ax2.spines.values():
        spine.set_color('#334155')
        spine.set_linewidth(1.5)
        
    plt.tight_layout(w_pad=4.0)
    
    out_cm_png = "results/figures/fig_confusion_matrix.png"
    out_cm_brain = os.path.join(brain_dir, "fig_confusion_matrix.png")
    out_cm_root = "fig_confusion_matrix.png"
    
    fig.savefig(out_cm_png, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    fig.savefig(out_cm_brain, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    fig.savefig(out_cm_root, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close(fig)
    print(f"Confusion matrix saved: {out_cm_png}, {out_cm_brain}, and {out_cm_root}")
    
    # =========================================================================
    # FIGURE 2: Standalone Diagnostic Metrics Summary Bar Chart
    # =========================================================================
    fig2, ax = plt.subplots(figsize=(8.5, 5.8), dpi=300)
    fig2.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')
    
    metrics = ['AUROC', 'Sensitivity (Recall)', 'Specificity (TNR)', 'Precision (PPV)', 'NPV', 'F1-Score', 'Exact Accuracy']
    test_vals = [76.4, 75.0, 66.7, 66.7, 75.0, 70.6, 70.6]
    cv_vals = [79.3, 71.2, 69.2, 69.8, 70.6, 70.2, 70.2]
    
    y_pos = np.arange(len(metrics))
    bar_height = 0.35
    
    rects1 = ax.barh(y_pos + bar_height/2, test_vals, bar_height, label='Locked Held-Out Test Set (n=17)', color='#2563EB', edgecolor='#1D4ED8', alpha=0.9)
    rects2 = ax.barh(y_pos - bar_height/2, cv_vals, bar_height, label='25-Fold CV Out-of-Fold (n=104)', color='#0D9488', edgecolor='#0F766E', alpha=0.9)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics, fontsize=10.5, fontweight='bold', color='#1E293B')
    ax.set_xlim(0, 110)
    ax.set_xlabel('Diagnostic Performance Score (%)', fontsize=11, fontweight='bold', color='#1E293B')
    ax.set_title('Multimodal Prognostic Framework (E6)\nComprehensive Diagnostic Breakdown', fontsize=12.5, fontweight='bold', pad=14, color='#0F172A')
    ax.grid(axis='x', linestyle='--', alpha=0.6, color='#94A3B8')
    ax.legend(loc='lower left', frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1', fontsize=9.5)
    
    for rect in rects1:
        w = rect.get_width()
        ax.text(w + 1.2, rect.get_y() + rect.get_height()/2, f"{w:.1f}%", ha='left', va='center', fontsize=9.0, fontweight='bold', color='#1D4ED8')
        
    for rect in rects2:
        w = rect.get_width()
        ax.text(w + 1.2, rect.get_y() + rect.get_height()/2, f"{w:.1f}%", ha='left', va='center', fontsize=9.0, fontweight='bold', color='#0F766E')
        
    for spine in ax.spines.values():
        spine.set_color('#CBD5E1')
        spine.set_linewidth(1.2)
        
    plt.tight_layout()
    
    out_diag_png = "results/figures/fig_diagnostic_metrics_summary.png"
    out_diag_brain = os.path.join(brain_dir, "fig_diagnostic_metrics_summary.png")
    out_diag_root = "fig_diagnostic_metrics_summary.png"
    
    fig2.savefig(out_diag_png, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    fig2.savefig(out_diag_brain, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    fig2.savefig(out_diag_root, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close(fig2)
    print(f"Diagnostic summary chart saved: {out_diag_png}, {out_diag_brain}, and {out_diag_root}")

if __name__ == "__main__":
    generate_clean_matrices()
