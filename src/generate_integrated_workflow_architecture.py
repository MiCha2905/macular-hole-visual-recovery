"""
================================================================================
INTEGRATED WORKFLOW & RESULTS ARCHITECTURE INFOGRAPHIC
================================================================================
Generates a publication-grade, modern graphical abstract integrating:
1. Top Workflow: Cohort Inputs -> Siamese Feature Extraction -> Conformal Engine
2. Bottom Grid: Actual Result Figures directly embedded (Scatter, SHAP, Staging, Conformal)
================================================================================
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from PIL import Image
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9.5
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['mathtext.fontset'] = 'cm'

def generate_integrated_architecture():
    # Large high-resolution canvas (24 x 16 inches @ 300 DPI)
    fig = plt.figure(figsize=(24, 16.5), dpi=300)
    fig.patch.set_facecolor('#F8FAFC')

    # Main Grid: Top 38% for Workflow Architecture, Bottom 62% for Integrated Results
    gs = GridSpec(2, 1, figure=fig, height_ratios=[0.38, 0.62],
                  left=0.02, right=0.98, top=0.95, bottom=0.02, hspace=0.08)

    # -------------------------------------------------------------------------
    # TOP SECTION: 3-STAGE WORKFLOW PIPELINE
    # -------------------------------------------------------------------------
    gs_top = GridSpec(1, 3, figure=fig, left=0.02, right=0.98, top=0.93, bottom=0.64, wspace=0.04)
    ax_stage1 = fig.add_subplot(gs_top[0, 0])
    ax_stage2 = fig.add_subplot(gs_top[0, 1])
    ax_stage3 = fig.add_subplot(gs_top[0, 2])

    stages = [
        (ax_stage1, "STAGE 1: Cohort & Dual-Axis OCT Inputs", '#0284C7', '#F0F9FF', '#BAE6FD', '1'),
        (ax_stage2, "STAGE 2: Tri-Track Feature Extraction & Fusion", '#7C3AED', '#F5F3FF', '#DDD6FE', '2'),
        (ax_stage3, "STAGE 3: Dual-Task Modeling & Conformal Engine", '#059669', '#ECFDF5', '#A7F3D0', '3')
    ]

    for ax, title, color, bg_col, border_col, badge_num in stages:
        ax.set_facecolor(bg_col)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(color)
            sp.set_linewidth(2.0)
            
        # Header banner
        header = patches.Rectangle((0, 0.88), 1.0, 0.12, transform=ax.transAxes,
                                   facecolor=color, edgecolor='none', zorder=2)
        ax.add_patch(header)
        badge = patches.Circle((0.06, 0.94), 0.045, transform=ax.transAxes,
                               facecolor='#FFFFFF', edgecolor=color, linewidth=1.5, zorder=3)
        ax.add_patch(badge)
        ax.text(0.06, 0.94, badge_num, transform=ax.transAxes,
                ha='center', va='center', color=color, fontweight='bold', fontsize=12, zorder=4)
        ax.text(0.13, 0.94, title, transform=ax.transAxes,
                ha='left', va='center', color='#FFFFFF', fontweight='bold', fontsize=11, zorder=4)

    # --- STAGE 1 CONTENT ---
    # Real OCT Scans Inset
    ax_oct_h = ax_stage1.inset_axes([0.05, 0.44, 0.42, 0.38])
    ax_oct_v = ax_stage1.inset_axes([0.53, 0.44, 0.42, 0.38])
    
    oct_h_path = "train/octs/0_baseline_H.tiff"
    oct_v_path = "train/octs/0_baseline_V.tiff"
    if os.path.exists(oct_h_path):
        ax_oct_h.imshow(Image.open(oct_h_path).convert('L'), cmap='gray', aspect='auto')
    ax_oct_h.set_xticks([]); ax_oct_h.set_yticks([])
    ax_oct_h.set_title("Horizontal B-Scan", fontsize=8.5, fontweight='bold', color='#0369A1', pad=2)
    for sp in ax_oct_h.spines.values(): sp.set_color('#0284C7'); sp.set_linewidth(1.2)

    if os.path.exists(oct_v_path):
        ax_oct_v.imshow(Image.open(oct_v_path).convert('L'), cmap='gray', aspect='auto')
    ax_oct_v.set_xticks([]); ax_oct_v.set_yticks([])
    ax_oct_v.set_title("Vertical B-Scan", fontsize=8.5, fontweight='bold', color='#0369A1', pad=2)
    for sp in ax_oct_v.spines.values(): sp.set_color('#0284C7'); sp.set_linewidth(1.2)

    card1 = patches.FancyBboxPatch((0.05, 0.05), 0.90, 0.35, boxstyle="round,pad=0.015",
                                   facecolor='#FFFFFF', edgecolor='#BAE6FD', linewidth=1.2)
    ax_stage1.add_patch(card1)
    ax_stage1.text(0.50, 0.34, "CHU de Québec Cohort (N = 121)", ha='center', fontweight='bold', color='#0369A1', fontsize=9.5)
    ax_stage1.text(0.10, 0.16, "• 104 Development (25-Fold CV) + 17 Locked Held-Out Test\n• 7 Clinical Features: Baseline VA, Duration, Size, Age, Sex, Lens, Axial",
                   fontsize=8, color='#334155', linespacing=1.3)

    # --- STAGE 2 CONTENT ---
    tracks = [
        ("Track 1: Clinical Tabular (7-D)", "Standardized z-score baseline parameters"),
        ("Track 2: Macro Morphometry (6-D)", "MLD, Base Diameter, Height, MHI, THI, DHI"),
        ("Track 3: Siamese Vision Embeddings (8-D)", "Dual-Stream ResNet-50 (2048-D) -> Fold-Isolated PCA-8")
    ]
    for idx, (t_name, t_desc) in enumerate(tracks):
        y_pos = 0.60 - idx * 0.24
        box = patches.FancyBboxPatch((0.05, y_pos), 0.90, 0.20, boxstyle="round,pad=0.015",
                                    facecolor='#FFFFFF', edgecolor='#DDD6FE', linewidth=1.2)
        ax_stage2.add_patch(box)
        ax_stage2.text(0.10, y_pos + 0.13, t_name, fontweight='bold', color='#6D28D9', fontsize=9)
        ax_stage2.text(0.10, y_pos + 0.05, t_desc, color='#475569', fontsize=8)

    ax_stage2.text(0.50, 0.05, "Multimodal Fusion Vector:  x_i = [x_clin, x_morph, x_vis] ∈ ℝ^21",
                   ha='center', fontsize=8.5, fontweight='bold', color='#7C3AED')

    # --- STAGE 3 CONTENT ---
    box_dt = patches.FancyBboxPatch((0.05, 0.46), 0.90, 0.38, boxstyle="round,pad=0.015",
                                   facecolor='#FFFFFF', edgecolor='#A7F3D0', linewidth=1.2)
    ax_stage3.add_patch(box_dt)
    ax_stage3.text(0.50, 0.77, "Dual-Task Regularized Modeling", ha='center', fontweight='bold', color='#047857', fontsize=9.5)
    ax_stage3.text(0.10, 0.57, "• Task A (Binary): L2-Logistic Regression (≥15 ETDRS letters gain)\n• Task B (Continuous): L2-Ridge Regression (ΔVA continuous letters)\n• Task C (Staging): 3-Tier Ordinal Clinical Staging Matrix",
                   fontsize=8, color='#334155', linespacing=1.35)

    box_cp = patches.FancyBboxPatch((0.05, 0.06), 0.90, 0.36, boxstyle="round,pad=0.015",
                                   facecolor='#F0FDF4', edgecolor='#10B981', linewidth=1.2)
    ax_stage3.add_patch(box_cp)
    ax_stage3.text(0.50, 0.35, "Cross-Conformal Uncertainty Calibration", ha='center', fontweight='bold', color='#065F46', fontsize=9.5)
    ax_stage3.text(0.10, 0.17, "• Non-Conformity Score: s_i = |ΔVA_i - f(x_i)|\n• Calibrated 90% Bound: [ f(x_i) ± q̂ ] at q̂ = ±14.05 letters\n• Out-of-Fold Coverage: 85.34 ± 8.48% (Dev) | 13/17 = 76.5% (Test)",
                   fontsize=8, color='#14532D', linespacing=1.3)

    # -------------------------------------------------------------------------
    # BOTTOM SECTION: 2x2 GRID OF ACTUAL RESULT FIGURES
    # -------------------------------------------------------------------------
    # Banner Separator for Results
    rect_res_banner = patches.Rectangle((0.02, 0.60), 0.96, 0.032, transform=fig.transFigure,
                                        facecolor='#1E293B', edgecolor='none', zorder=5)
    fig.add_artist(rect_res_banner)
    fig.text(0.50, 0.616, "VERIFIED STUDY RESULTS & PROGNOSTIC PERFORMANCE (CHU DE QUÉBEC BENCHMARK)",
             ha='center', va='center', color='#FFFFFF', fontweight='bold', fontsize=12, zorder=6)

    gs_bot = GridSpec(2, 2, figure=fig, left=0.02, right=0.98, top=0.59, bottom=0.02,
                      hspace=0.08, wspace=0.04)

    ax_res_a = fig.add_subplot(gs_bot[0, 0])
    ax_res_b = fig.add_subplot(gs_bot[0, 1])
    ax_res_c = fig.add_subplot(gs_bot[1, 0])
    ax_res_d = fig.add_subplot(gs_bot[1, 1])

    # Result Panels Setup
    res_panels = [
        (ax_res_a, "Figure A: Continuous Visual Acuity Gain Regression (ΔVA) & Residuals", "results/figures/fig4_continuous_recovery_scatter.png",
         "Top Multimodal Model: R² = 0.485 ± 0.137, MAE = 8.66 ± 1.73 letters. 76.5% of test errors within ±10 letters."),
        (ax_res_b, "Figure B: Multi-Modal Prognostic Feature Importance & Attribution", "results/figures/fig3_feature_importance_and_shap.png",
         "Preoperative visual acuity and symptom duration are the dominant clinical drivers of functional visual prognosis."),
        (ax_res_c, "Figure C: Binary Classification Accuracy & 3-Tier Clinical Staging", "results/figures/fig1_accuracy_and_staging.png",
         "Baseline VA alone achieves 82.7% AUROC. 3-tier staging achieves Quadratic Weighted Kappa κ_w = 0.684 with 94.1% safety."),
        (ax_res_d, "Figure D: Finite-Sample Cross-Conformal Uncertainty Intervals", "results/figures/fig5_conformal_patient_intervals.png",
         "Patient-by-patient 90% confidence bands (±14.76 let) capturing 13/17 = 76.5% of locked held-out test patients.")
    ]

    for ax, title, img_path, footer in res_panels:
        ax.set_facecolor('#FFFFFF')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color('#CBD5E1')
            sp.set_linewidth(1.5)

        # Title bar inside panel
        ax.text(0.02, 0.97, title, transform=ax.transAxes, ha='left', va='top',
                fontweight='bold', fontsize=10.5, color='#0F172A')

        # Embed Image with generous padding
        if os.path.exists(img_path):
            img = Image.open(img_path)
            ax_in = ax.inset_axes([0.02, 0.06, 0.96, 0.86])
            ax_in.imshow(img)
            ax_in.axis('off')

        # Footer explanation
        ax.text(0.02, 0.015, footer, transform=ax.transAxes, ha='left', va='bottom',
                fontsize=8.5, color='#475569', fontstyle='italic')

    # Master Title
    fig.suptitle("Comprehensive Multimodal Architecture & Verified Visual Recovery Prognosis in Macular Hole Surgery",
                 fontsize=15, fontweight='bold', color='#0F172A', y=0.985)

    out_path = "results/figures/fig_integrated_methodology_and_results.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Integrated methodology & results figure successfully saved to: {out_path}")
    return out_path

if __name__ == "__main__":
    generate_integrated_architecture()
