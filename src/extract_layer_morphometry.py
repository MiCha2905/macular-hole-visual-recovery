"""
Track 2 Upgrade: Retinal Micro-Structural Layer Biomarker Extractor (EZ & ELM Integrity)
Implements layer integrity profiling based on Baumann et al. (2021) / ophthalmic literature:
1. Ellipsoid Zone (EZ / IS-OS junction) defect diameter (horizontal gap in the hyperreflective EZ band).
2. External Limiting Membrane (ELM) defect width and disruption continuity ratio.
3. Residual photoreceptor outer nuclear layer (ONL) thickness adjacent to the hole margin.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

def analyze_bscan_layer_integrity(img_path):
    """
    Measures microscopic photoreceptor layer biomarkers (EZ and ELM integrity) from a B-scan.
    """
    default_vals = {
        'ez_defect_px': np.nan,
        'elm_defect_px': np.nan,
        'ez_integrity_ratio': np.nan,
        'elm_integrity_ratio': np.nan,
        'photoreceptor_cuff_thickness': np.nan
    }
    if not os.path.exists(img_path):
        return default_vals
    
    try:
        with Image.open(img_path) as pil_img:
            img = np.array(pil_img.convert('L'), dtype=np.float32)
    except Exception:
        return default_vals
        
    if img is None or img.size == 0:
        return default_vals
    
    H_img, W_img = img.shape
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-6)
    
    # 1. Identify RPE hyperreflective band across foveal central 50%
    center_x = W_img // 2
    roi_x_min = max(0, center_x - int(W_img * 0.25))
    roi_x_max = min(W_img, center_x + int(W_img * 0.25))
    
    # Detect RPE vertical trajectory (argmax of intensity profile)
    rpe_y_trajectory = []
    for x in range(roi_x_min, roi_x_max):
        col = img_norm[int(H_img * 0.35):int(H_img * 0.75), x]
        if len(col) > 0:
            rpe_y_trajectory.append(int(H_img * 0.35) + np.argmax(col))
        else:
            rpe_y_trajectory.append(int(H_img * 0.55))
            
    rpe_y_trajectory = np.array(rpe_y_trajectory)
    # Smooth RPE trajectory
    rpe_y_smooth = np.convolve(rpe_y_trajectory, np.ones(5)/5, mode='same').astype(int)
    
    # 2. Extract Intensity Profile along the EZ layer (approx 4-12 pixels above RPE)
    # and ELM layer (approx 14-22 pixels above RPE)
    ez_intensities = []
    elm_intensities = []
    onl_thicknesses = []
    
    tissue_thresh = 0.22
    
    for i, x in enumerate(range(roi_x_min, roi_x_max)):
        y_rpe = rpe_y_smooth[i]
        
        # Sample EZ band (3 to 8 pixels above RPE)
        ez_sample = img_norm[max(0, y_rpe - 8):max(0, y_rpe - 2), x]
        ez_val = np.mean(ez_sample) if len(ez_sample) > 0 else 0.0
        ez_intensities.append(ez_val)
        
        # Sample ELM band (10 to 18 pixels above RPE)
        elm_sample = img_norm[max(0, y_rpe - 18):max(0, y_rpe - 10), x]
        elm_val = np.mean(elm_sample) if len(elm_sample) > 0 else 0.0
        elm_intensities.append(elm_val)
        
        # Measure outer retinal thickness above RPE
        col_above_rpe = img_norm[:y_rpe, x]
        above_th = np.where(col_above_rpe > tissue_thresh)[0]
        thick = (y_rpe - above_th[0]) if len(above_th) > 0 else 0
        onl_thicknesses.append(thick)
        
    ez_intensities = np.array(ez_intensities)
    elm_intensities = np.array(elm_intensities)
    onl_thicknesses = np.array(onl_thicknesses)
    
    # 3. Detect EZ and ELM Defect Horizontal Extents
    # Normal intact EZ is hyperreflective (top 30th percentile in outer retina)
    ez_baseline = np.percentile(ez_intensities, 75) if len(ez_intensities) > 0 else 0.5
    elm_baseline = np.percentile(elm_intensities, 75) if len(elm_intensities) > 0 else 0.4
    
    # Defect condition: where EZ brightness drops below 40% of baseline intact band
    ez_disrupted = np.where(ez_intensities < (ez_baseline * 0.45))[0]
    elm_disrupted = np.where(elm_intensities < (elm_baseline * 0.45))[0]
    
    ez_defect_px = float(len(ez_disrupted)) if len(ez_disrupted) >= 2 else 5.0
    elm_defect_px = float(len(elm_disrupted)) if len(elm_disrupted) >= 2 else 5.0
    
    total_roi_w = float(roi_x_max - roi_x_min)
    ez_integrity_ratio = float(max(0.0, 1.0 - (ez_defect_px / total_roi_w)))
    elm_integrity_ratio = float(max(0.0, 1.0 - (elm_defect_px / total_roi_w)))
    
    # Photoreceptor cuff thickness: average thickness in intact margin (25 pixels on each side)
    intact_margin_thickness = float(np.mean(onl_thicknesses[onl_thicknesses > np.median(onl_thicknesses)])) if len(onl_thicknesses) > 0 else 50.0
    
    return {
        'ez_defect_px': ez_defect_px,
        'elm_defect_px': elm_defect_px,
        'ez_integrity_ratio': ez_integrity_ratio,
        'elm_integrity_ratio': elm_integrity_ratio,
        'photoreceptor_cuff_thickness': intact_margin_thickness
    }

def extract_all_layer_morphometry(df, output_csv="data/layer_morphometry_features.csv"):
    results = []
    
    for _, row in df.iterrows():
        pid = row['id']
        h_path = row['oct_baseline_H_path']
        v_path = row['oct_baseline_V_path']
        
        m_h = analyze_bscan_layer_integrity(h_path)
        m_v = analyze_bscan_layer_integrity(v_path)
        
        # Dual-axis average
        avg_record = {
            'id': pid,
            'ez_defect_mean': float(np.nanmean([m_h['ez_defect_px'], m_v['ez_defect_px']])),
            'elm_defect_mean': float(np.nanmean([m_h['elm_defect_px'], m_v['elm_defect_px']])),
            'ez_integrity_mean': float(np.nanmean([m_h['ez_integrity_ratio'], m_v['ez_integrity_ratio']])),
            'elm_integrity_mean': float(np.nanmean([m_h['elm_integrity_ratio'], m_v['elm_integrity_ratio']])),
            'photoreceptor_cuff_mean': float(np.nanmean([m_h['photoreceptor_cuff_thickness'], m_v['photoreceptor_cuff_thickness']]))
        }
        results.append(avg_record)
        
    layer_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    layer_df.to_csv(output_csv, index=False)
    print(f"Extracted EZ/ELM Layer Morphometry for {len(layer_df)} patients -> {output_csv}")
    print(layer_df.head(5))
    return layer_df

if __name__ == "__main__":
    from src.data_preprocessing import load_and_clean_data
    df = load_and_clean_data(".")
    extract_all_layer_morphometry(df)
