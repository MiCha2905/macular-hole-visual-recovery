"""
Generate a Master Methodology Pipeline Infographic in the Exact Visual Style
of Top-Tier Medical AI Journals (Clean Modular Stages, 3D CNN Blocks, Real Scans,
Caliper Geometry, Late Fusion Ensemble, and Conformal Uncertainty Bands).
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0

def build_pipeline_diagram():
    fig = plt.figure(figsize=(22, 11.5), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 11.5)
    ax.axis('off')
    
    # Background
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    # Helper for rounded card containers
    def draw_card(x, y, w, h, title, stage_num=None, border_color='#94A3B8', bg_color='#FFFFFF', header_color='#0F172A'):
        # Card box
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.28",
                             facecolor=bg_color, edgecolor=border_color, linewidth=1.6, zorder=1)
        ax.add_patch(box)
        
        # Header text
        if stage_num:
            full_title = f"Stage {stage_num}: {title}"
        else:
            full_title = title
            
        ax.text(x + w/2, y + h + 0.14, full_title, ha='center', va='bottom',
                fontsize=12, fontweight='bold', color=header_color, zorder=3)

    # -------------------------------------------------------------------------
    # STAGE 1: Image Acquisition & Preoperative Data Store (Top Left)
    # -------------------------------------------------------------------------
    draw_card(0.6, 6.0, 6.4, 4.7, "Image Acquisition and Cohort Data Store", stage_num="1", border_color='#64748B')
    
    # Sub-box 1: SD-OCT Acquisition
    box_s1 = FancyBboxPatch((0.85, 6.2), 2.8, 4.25, boxstyle="round,pad=0,rounding_size=0.18",
                            facecolor='#F8FAFC', edgecolor='#CBD5E1', linewidth=1.1, zorder=2)
    ax.add_patch(box_s1)
    ax.text(2.25, 10.15, "Spectral-Domain OCT\n(Heidelberg Spectralis)", ha='center', va='center',
            fontsize=9.0, fontweight='bold', color='#1E293B', zorder=3)
    
    # Load actual baseline OCT images
    h_scan_path = "train/octs/0_baseline_H.tiff"
    v_scan_path = "train/octs/0_baseline_V.tiff"
    
    if os.path.exists(h_scan_path):
        img_h = Image.open(h_scan_path).convert('L')
        img_h = np.array(img_h)
        h_crop = img_h[int(img_h.shape[0]*0.2):int(img_h.shape[0]*0.8), int(img_h.shape[1]*0.25):int(img_h.shape[1]*0.75)]
        ax.imshow(h_crop, cmap='gray', extent=[1.0, 3.5, 8.1, 9.7], zorder=3, aspect='auto')
        ax.text(2.25, 7.9, "Horizontal B-scan (448×448)", ha='center', va='top', fontsize=8.0, color='#475569', zorder=4)
        
    if os.path.exists(v_scan_path):
        img_v = Image.open(v_scan_path).convert('L')
        img_v = np.array(img_v)
        v_crop = img_v[int(img_v.shape[0]*0.2):int(img_v.shape[0]*0.8), int(img_v.shape[1]*0.25):int(img_v.shape[1]*0.75)]
        ax.imshow(v_crop, cmap='gray', extent=[1.0, 3.5, 6.45, 7.65], zorder=3, aspect='auto')
        ax.text(2.25, 6.3, "Vertical B-scan (448×448)", ha='center', va='top', fontsize=8.0, color='#475569', zorder=4)

    # Arrow 1 -> Data Store
    ax.annotate('', xy=(4.1, 8.35), xytext=(3.75, 8.35),
                arrowprops=dict(arrowstyle="simple,head_width=0.45,head_length=0.45", color='#F59E0B', lw=0.5), zorder=4)

    # Data Store Cylinder
    box_store = FancyBboxPatch((4.15, 6.45), 2.65, 3.8, boxstyle="round,pad=0,rounding_size=0.18",
                               facecolor='#E0F2FE', edgecolor='#0284C7', linewidth=1.3, zorder=2)
    ax.add_patch(box_store)
    ax.text(5.47, 9.85, "Preoperative Data Store", ha='center', va='center', fontsize=10.0, fontweight='bold', color='#0369A1', zorder=3)
    ax.text(5.47, 8.85, "CHU de Québec Cohort\nN = 121 Patients", ha='center', va='center', fontsize=9.2, fontweight='bold', color='#0C4A6E', zorder=3)
    ax.text(5.47, 7.45, "• Development: 104 pts\n• Locked Test: 17 pts\n• Excluded: 373 uncurated\n• Target: 6-Mo ΔVA (ETDRS)",
            ha='center', va='center', fontsize=8.0, color='#0369A1', linespacing=1.35, zorder=3)

    # -------------------------------------------------------------------------
    # STAGE 2: Quality Protocol & Clinical Feature Registry (Top Middle)
    # -------------------------------------------------------------------------
    draw_card(7.5, 6.0, 5.0, 4.7, "Quality Protocol & Clinical Priors", stage_num="2", border_color='#64748B')
    
    # Quality Protocol sub-header
    hdr_box = FancyBboxPatch((7.75, 9.75), 4.5, 0.72, boxstyle="round,pad=0,rounding_size=0.12",
                             facecolor='#FEF3C7', edgecolor='#D97706', linewidth=1.1, zorder=2)
    ax.add_patch(hdr_box)
    ax.text(10.0, 10.11, "⚙ Clinical Quality & Verification", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#92400E', zorder=3)
    
    # 7 Clinical Variables Box
    ax.text(7.9, 9.4, "+ Age (years)\n+ Sex (Male / Female)\n+ Lens Status (Pseudophakic)\n+ Symptom Duration (weeks)\n+ Elevated Retinal Edges (0/1)\n+ Macular Hole Diameter (MLD, μm)\n+ Baseline Visual Acuity (ETDRS)",
            ha='left', va='top', fontsize=8.2, color='#334155', linespacing=1.38, zorder=3)
    
    # In-fold isolation & Sentinel processing Badge
    badge = FancyBboxPatch((7.75, 6.25), 4.5, 0.95, boxstyle="round,pad=0,rounding_size=0.12",
                           facecolor='#F1F5F9', edgecolor='#94A3B8', linewidth=1.1, zorder=2)
    ax.add_patch(badge)
    ax.text(10.0, 6.72, "Sentinel (-9 → NaN) Imputation\nStrict In-Fold Scaling: $\\mathbf{x}_i^{\\text{clin}} \\in \\mathbb{R}^7$",
            ha='center', va='center', fontsize=8.2, fontweight='bold', color='#1E293B', zorder=3)

    # Connect Stage 1 -> Stage 2
    ax.annotate('', xy=(7.45, 8.35), xytext=(7.05, 8.35),
                arrowprops=dict(arrowstyle="simple,head_width=0.45,head_length=0.45", color='#0284C7', lw=0.5), zorder=4)

    # -------------------------------------------------------------------------
    # STAGE 3: Macro-Morphometry & Caliper Geometry (Top Right)
    # -------------------------------------------------------------------------
    draw_card(13.0, 6.0, 8.4, 4.7, "OCT Caliper Morphometry & Geometry", stage_num="3", border_color='#64748B')
    
    # Connect Stage 2 -> Stage 3
    ax.annotate('', xy=(12.95, 8.35), xytext=(12.55, 8.35),
                arrowprops=dict(arrowstyle="simple,head_width=0.45,head_length=0.45", color='#0284C7', lw=0.5), zorder=4)

    # Anomaly / Detection Node
    node_circle = plt.Circle((13.6, 8.35), 0.5, facecolor='#DC2626', edgecolor='#991B1B', linewidth=1.3, zorder=3)
    ax.add_patch(node_circle)
    ax.text(13.6, 8.35, "Caliper\nDetect", ha='center', va='center', fontsize=7.2, fontweight='bold', color='#FFFFFF', zorder=4)

    # Diagram of Macular Hole Lumen & Calipers
    caliper_box = FancyBboxPatch((14.4, 6.25), 3.8, 4.2, boxstyle="round,pad=0,rounding_size=0.18",
                                 facecolor='#F8FAFC', edgecolor='#CBD5E1', linewidth=1.1, zorder=2)
    ax.add_patch(caliper_box)
    
    # Draw stylized Macular Hole Profile
    mh_x = np.linspace(14.6, 18.0, 120)
    y_rpe = 6.75
    ax.plot([14.6, 18.0], [y_rpe, y_rpe], color='#E11D48', lw=2.8, label='RPE Band', zorder=3)
    
    # Retinal profile with defect in center
    y_ret = 9.3 - 2.2 * np.exp(-((mh_x - 16.3)**2) / 0.14)
    ax.plot(mh_x, y_ret, color='#0284C7', lw=2.4, zorder=3)
    ax.fill_between(mh_x, y_rpe, y_ret, color='#0284C7', alpha=0.18, zorder=2)
    
    # Caliper lines
    # MLD (at minimum aperture neck)
    ax.plot([15.85, 16.75], [7.85, 7.85], color='#D97706', lw=2.2, marker='|', zorder=4)
    ax.text(16.3, 8.0, "MLD", ha='center', va='bottom', fontsize=8.0, fontweight='bold', color='#B45309', zorder=5)
    
    # Base Diameter (at RPE level)
    ax.plot([15.5, 17.1], [6.85, 6.85], color='#16A34A', lw=2.2, marker='|', zorder=4)
    ax.text(16.3, 7.0, "Basal Diameter (BD)", ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#15803D', zorder=5)
    
    # Height (vertical)
    ax.plot([17.4, 17.4], [6.75, 9.3], color='#7C3AED', lw=2.0, linestyle=':', marker='_', zorder=4)
    ax.text(17.5, 8.0, "Height (H)", ha='left', va='center', fontsize=7.5, fontweight='bold', color='#6D28D9', zorder=5)
    
    # Decision Diamond
    diamond_pts = [[18.6, 8.35], [19.25, 9.15], [19.9, 8.35], [19.25, 7.55]]
    diamond = Polygon(diamond_pts, closed=True, facecolor='#FEF3C7', edgecolor='#D97706', linewidth=1.4, zorder=3)
    ax.add_patch(diamond)
    ax.text(19.25, 8.35, "Scale-Free\nIndices", ha='center', va='center', fontsize=7.2, fontweight='bold', color='#92400E', zorder=4)
    
    # Arrows from diamond
    ax.text(20.15, 8.55, "$\\text{MHI} = H / \\text{BD}$\n$\\text{THI} = H / \\text{MLD}$\n$\\text{DHI} = (\\text{BD} - \\text{MLD})/H$",
            ha='left', va='center', fontsize=8.5, fontweight='bold', color='#1E293B', linespacing=1.35, zorder=4)
    ax.text(20.15, 6.85, "$\\mathbf{x}_i^{\\text{morph}} \\in \\mathbb{R}^6$", fontsize=9.5, fontweight='bold', color='#0284C7', zorder=4)

    # -------------------------------------------------------------------------
    # STAGE 4: Siamese Deep Vision Backbone & In-Fold PCA-8 (Bottom Left)
    # -------------------------------------------------------------------------
    draw_card(0.6, 0.5, 10.8, 4.8, "Siamese Deep Vision Feature Extraction & In-Fold PCA-8", stage_num="4", border_color='#64748B')
    
    # Dual input slices display
    ax.text(1.4, 4.85, "Dual Cross-Hairs", ha='center', va='center', fontsize=9.0, fontweight='bold', color='#1E293B', zorder=3)
    
    if os.path.exists(h_scan_path):
        ax.imshow(h_crop, cmap='bone', extent=[0.85, 1.95, 3.0, 4.45], zorder=3, aspect='auto')
        ax.text(1.4, 2.8, "Horizontal B-Scan", ha='center', va='center', fontsize=7.5, color='#475569', zorder=4)
        
    if os.path.exists(v_scan_path):
        ax.imshow(v_crop, cmap='bone', extent=[0.85, 1.95, 1.05, 2.5], zorder=3, aspect='auto')
        ax.text(1.4, 0.85, "Vertical B-Scan", ha='center', va='center', fontsize=7.5, color='#475569', zorder=4)

    # Arrow to 3D CNN
    ax.annotate('', xy=(2.35, 2.75), xytext=(2.05, 2.75),
                arrowprops=dict(arrowstyle="simple,head_width=0.45,head_length=0.45", color='#0EA5E9', lw=0.5), zorder=4)

    # 3D Isometric CNN Blocks (Encoder Backbone)
    def draw_3d_block(x, y, w, h, depth, face_color, side_color, top_color, label=""):
        # Front face
        front = FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0", facecolor=face_color, edgecolor='#0369A1', linewidth=0.9, zorder=3)
        ax.add_patch(front)
        # Top face
        top_pts = [[x, y + h], [x + depth*0.5, y + h + depth*0.5], [x + w + depth*0.5, y + h + depth*0.5], [x + w, y + h]]
        top_poly = Polygon(top_pts, closed=True, facecolor=top_color, edgecolor='#0369A1', linewidth=0.9, zorder=3)
        ax.add_patch(top_poly)
        # Side face
        side_pts = [[x + w, y], [x + w + depth*0.5, y + depth*0.5], [x + w + depth*0.5, y + h + depth*0.5], [x + w, y + h]]
        side_poly = Polygon(side_pts, closed=True, facecolor=side_color, edgecolor='#0369A1', linewidth=0.9, zorder=3)
        ax.add_patch(side_poly)
        if label:
            ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=7.2, fontweight='bold', color='#FFFFFF', zorder=4, rotation=90)

    # Block 1
    draw_3d_block(2.4, 1.25, 0.5, 3.0, 0.4, '#0284C7', '#0369A1', '#38BDF8', "7×7 Conv")
    # Block 2
    draw_3d_block(3.2, 1.5, 0.6, 2.5, 0.4, '#0EA5E9', '#0284C7', '#7DD3FC', "ResNet-50")
    # Block 3
    draw_3d_block(4.1, 1.8, 0.7, 1.9, 0.4, '#38BDF8', '#0EA5E9', '#BAE6FD', "Layer 1-4")
    # Block 4 (Bottleneck GAP)
    draw_3d_block(5.1, 2.1, 0.55, 1.3, 0.4, '#0284C7', '#0369A1', '#38BDF8', "GAP")

    # Symmetric Mean Pooling Badge
    pool_box = FancyBboxPatch((6.1, 1.7), 2.0, 2.1, boxstyle="round,pad=0,rounding_size=0.18",
                              facecolor='#FEF3C7', edgecolor='#D97706', linewidth=1.3, zorder=3)
    ax.add_patch(pool_box)
    ax.text(7.1, 3.1, "Symmetric\nMean Pooling", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#92400E', zorder=4)
    ax.text(7.1, 2.2, "$\\mathbf{f}_i = \\frac{1}{2}(\\phi_H + \\phi_V)$\n2048-D Vector", ha='center', va='center', fontsize=7.8, color='#B45309', zorder=4)

    # Arrow to In-Fold PCA
    ax.annotate('', xy=(8.5, 2.75), xytext=(8.2, 2.75),
                arrowprops=dict(arrowstyle="simple,head_width=0.45,head_length=0.45", color='#D97706', lw=0.5), zorder=4)

    # In-Fold PCA Module
    pca_patch = FancyBboxPatch((8.55, 1.4), 2.65, 2.7, boxstyle="round,pad=0,rounding_size=0.18",
                               facecolor='#F0FDF4', edgecolor='#16A34A', linewidth=1.3, zorder=3)
    ax.add_patch(pca_patch)
    ax.text(9.87, 3.65, "In-Fold PCA-8 Projection", ha='center', va='center', fontsize=8.8, fontweight='bold', color='#166534', zorder=4)
    ax.text(9.87, 2.75, "• Fit strictly on $\\mathcal{D}_{\\text{train}}$\n• Top 8 Eigenvectors\n• Prevents small-sample\n  overfitting ($N \\approx 100$)",
            ha='center', va='center', fontsize=7.8, color='#15803D', linespacing=1.3, zorder=4)
    ax.text(9.87, 1.75, "$\\mathbf{x}_i^{\\text{vis}} \\in \\mathbb{R}^8$", ha='center', va='center', fontsize=9.2, fontweight='bold', color='#16A34A', zorder=4)

    # -------------------------------------------------------------------------
    # STAGE 5: Multimodal Late Fusion & Conformal Uncertainty (Bottom Right)
    # -------------------------------------------------------------------------
    draw_card(11.8, 0.5, 9.6, 4.8, "Multimodal Late Fusion & Conformal Uncertainty Engine", stage_num="5", border_color='#64748B')
    
    # Multimodal Concatenator Box
    concat_box = FancyBboxPatch((12.05, 1.7), 2.1, 2.9, boxstyle="round,pad=0,rounding_size=0.18",
                                facecolor='#F5F3FF', edgecolor='#7C3AED', linewidth=1.3, zorder=3)
    ax.add_patch(concat_box)
    ax.text(13.1, 4.15, "Multimodal\nVector ($\\mathbb{R}^{15}$)", ha='center', va='center', fontsize=9.0, fontweight='bold', color='#5B21B6', zorder=4)
    ax.text(13.1, 3.0, "Track 0 (Clin: 7)\n+\nTrack 1 (Vis: 8)\n+\n[Track 2 (Morph: 6)]",
            ha='center', va='center', fontsize=7.8, color='#6D28D9', linespacing=1.3, zorder=4)
    ax.text(13.1, 2.05, "$\\mathbf{x}_i^{\\text{multimodal}}$", ha='center', va='center', fontsize=8.8, fontweight='bold', color='#7C3AED', zorder=4)

    # Arrow to Fusion Ensemble
    ax.annotate('', xy=(14.55, 3.15), xytext=(14.2, 3.15),
                arrowprops=dict(arrowstyle="simple,head_width=0.45,head_length=0.45", color='#7C3AED', lw=0.5), zorder=4)

    # Dual Estimator Ensemble Box
    ens_box = FancyBboxPatch((14.6, 1.3), 3.0, 3.5, boxstyle="round,pad=0,rounding_size=0.18",
                             facecolor='#FFF1F2', edgecolor='#E11D48', linewidth=1.3, zorder=3)
    ax.add_patch(ens_box)
    ax.text(16.1, 4.4, "50/50 Late Fusion Ensemble", ha='center', va='center', fontsize=8.8, fontweight='bold', color='#9F1239', zorder=4)
    
    # Sub-estimator 1: Ridge / Logistic
    ax.text(16.1, 3.65, "1. Regularized Linear ($L_2$)\n• Logistic Reg ($C=0.5$)\n• Ridge Reg ($\\alpha=2.0$)",
            ha='center', va='center', fontsize=7.8, color='#BE123C', linespacing=1.25, zorder=4)
    # Sub-estimator 2: HistGBDT
    ax.text(16.1, 2.45, "2. Shallow HistGBDT\n• $\\text{max\\_depth} = 2$\n• $\\text{max\\_iter} = 35$ (trees)\n• $\\text{lr}=0.05, \\text{leaf}=4$",
            ha='center', va='center', fontsize=7.8, color='#9F1239', linespacing=1.25, zorder=4)
    ax.text(16.1, 1.6, "$\\hat{\\mu}(\\mathbf{x}) = \\frac{1}{2}\\mathbf{w}^\\top\\mathbf{x} + \\frac{1}{2}g(\\mathbf{x})$",
            ha='center', va='center', fontsize=8.0, fontweight='bold', color='#E11D48', zorder=4)

    # Arrow to Prognostic Outputs & Conformal Bands
    ax.annotate('', xy=(18.0, 3.15), xytext=(17.65, 3.15),
                arrowprops=dict(arrowstyle="simple,head_width=0.45,head_length=0.45", color='#E11D48', lw=0.5), zorder=4)

    # Final Prognostic Outputs Box
    out_box = FancyBboxPatch((18.05, 0.8), 3.15, 4.15, boxstyle="round,pad=0,rounding_size=0.18",
                             facecolor='#ECFDF5', edgecolor='#059669', linewidth=1.3, zorder=3)
    ax.add_patch(out_box)
    ax.text(19.62, 4.55, "Dual Prognostic Outputs", ha='center', va='center', fontsize=9.2, fontweight='bold', color='#065F46', zorder=4)
    
    # Task 1: Binary Gain
    ax.text(19.62, 3.8, "1. Binary $\\geq$15-Letter Gain\nAUROC: 88.9% (Locked Test)\n$F_1$-Score: 70.2% (CV)",
            ha='center', va='center', fontsize=8.0, fontweight='bold', color='#047857', linespacing=1.25, zorder=4)
    
    # Task 2: Continuous Delta VA
    ax.text(19.62, 2.75, "2. Continuous $\\Delta$VA (Letters)\n$R^2 = 0.557$, MAE = 7.60 let.\n(Locked Single-Pass Test)",
            ha='center', va='center', fontsize=8.0, color='#065F46', linespacing=1.25, zorder=4)
    
    # Conformal Prediction Bands
    conf_band = FancyBboxPatch((18.25, 0.95), 2.75, 1.15, boxstyle="round,pad=0,rounding_size=0.12",
                               facecolor='#D1FAE5', edgecolor='#059669', linewidth=1.1, zorder=4)
    ax.add_patch(conf_band)
    ax.text(19.62, 1.7, "Distribution-Free Conformal Band", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#065F46', zorder=5)
    ax.text(19.62, 1.25, "$\\mathcal{C}_{0.90}(\\mathbf{x}) = [\\hat{y} - \\hat{q},  \\hat{y} + \\hat{q}]$\n(Test Coverage: 13/17 = 76.5%, $\\pm 14.76$ let)",
            ha='center', va='center', fontsize=7.2, color='#047857', linespacing=1.2, zorder=5)

    # Connecting Flow Arrow from Top Row to Bottom Row (Clinical & Morphometry Priors)
    arrow_down = FancyArrowPatch((11.0, 5.9), (12.4, 5.4), connectionstyle="arc3,rad=-0.12",
                                 arrowstyle="simple,head_width=0.45,head_length=0.45", color='#64748B', lw=0.8, zorder=5)
    ax.add_patch(arrow_down)
    ax.text(11.7, 5.55, "Clinical & Morphometry Priors", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#475569', zorder=6)

    # Save outputs
    out_png = "results/figures/fig_master_methodology_architecture.png"
    out_brain = "/Users/sonali/.gemini/antigravity-ide/brain/3fdb2248-49df-4e37-b31d-2bd2f30476ff/fig_master_methodology_architecture.png"
    out_root = "fig_master_methodology_architecture.png"
    
    fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    fig.savefig(out_brain, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    fig.savefig(out_root, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close(fig)
    print(f"Master Methodology Figure successfully updated at: {out_png}, {out_brain}, and {out_root}")

if __name__ == "__main__":
    build_pipeline_diagram()
