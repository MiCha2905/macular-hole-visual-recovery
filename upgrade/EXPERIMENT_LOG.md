# Upgrade Experiment Log: Small-Sample Architectural & Feature Enhancements

**Author:** Sonali Gupta (Thapar Institute of Engineering and Technology)  
**Dataset:** CHU de Québec HD-OCT Macular Hole Cohort ($N=104$ Dev, 25-Fold CV)  
**Objective:** Empirically evaluate if Supervised PLS Vision, Non-Linear Tree Boosting, Clinical Interaction Terms, and Adaptive CQR can outperform the verified baseline.

---

## 1. Experimental Setup & Hypotheses

| Upgrade Track | Hypothesis | Technique |
|:---|:---|:---|
| **Upgrade T1: Clinical Interactions** | Interaction terms ($\text{VA} \times \text{MLD}$, $\text{Duration} \times \text{Height}$) capture non-linear prognostic dynamics. | ElasticNet ($L_1=0.3, \alpha=0.1$) on 15 engineered features |
| **Upgrade T2: Supervised PLS Vision** | Supervised PLS extracts visual components specifically aligned with visual acuity recovery. | PLSRegression ($k=4$) on 2048-D vision embeddings (in-fold) |
| **Upgrade T3: Multimodal PLS Fusion** | Combining advanced clinical features with supervised PLS vision will boost $R^2$. | ElasticNet on Clinical + PLS-4 |
| **Upgrade T4: Shallow GBDT Ensemble** | Shallow decision trees ($depth=2$) capture non-linear threshold effects. | Blended ElasticNet (70%) + HistGradientBoosting (30%) |
| **Adaptive CQR** | Heteroscedastic residual scaling reduces interval width for confident patients. | Non-conformity score normalized by local residual dispersion model |

---

## 2. 25-Fold Repeated Stratified CV Benchmark Results

| Model / Track | CV $R^2$ | CV MAE (letters) | CV AUROC (%) | Conformal PICP (Avg Radius) |
|:---|:---:---:|:---:|:---:|:---:|
| **Baseline: VA Only (Ridge)** | $\mathbf{0.485 \pm 0.159}$ | $\mathbf{8.58 \pm 1.29}$ | $\mathbf{82.38 \pm 8.26\%}$ | $89.6\%$ ($\pm 16.34\text{ let.}$) |
| **Baseline: Full Clinical (Ridge)** | $0.475 \pm 0.198$ | $8.76 \pm 1.50$ | $80.15 \pm 10.30\%$ | $85.7\%$ ($\pm 15.09\text{ let.}$) |
| **Baseline: Multimodal (Ridge + PCA-8)** | $0.467 \pm 0.206$ | $8.63 \pm 1.62$ | $77.12 \pm 9.23\%$ | $84.8\%$ ($\pm 14.50\text{ let.}$) |
| **Upgrade T1: Clinical Interactions (ElasticNet)** | $0.421 \pm 0.203$ | $8.99 ± 1.68$ | $80.23 \pm 9.45\%$ | $82.5\%$ ($\pm 14.98\text{ let.}$) |
| **Upgrade T2: Supervised PLS Vision** | $-0.298 \pm 0.409$ | $13.10 \pm 2.11$ | $59.93 \pm 8.74\%$ | $36.4\%$ (Collapsed due to overfitting) |
| **Upgrade T3: Multimodal PLS Fusion** | $-0.217 \pm 0.364$ | $12.58 \pm 2.03$ | $59.20 \pm 8.93\%$ | $32.5\%$ (Collapsed) |
| **Upgrade T4: Shallow GBDT Blend** | $-0.099 \pm 0.324$ | $12.02 \pm 2.03$ | $63.21 \pm 8.90\%$ | $28.8\%$ (Collapsed) |

---

## 3. Scientific Discoveries & Key Takeaways

1. **Supervised Dimension Reduction (PLS) Overfits Severely on $N=104$:**
   - In training folds, PLS finds linear combinations of 2048 vision features that fit the training $y$ perfectly, but because $2048 \gg 83$, it memorizes in-sample noise. Out-of-fold generalization collapses to $R^2 = -0.298$.
   - **Conclusion:** Unsupervised PCA-8 (which does not see the training targets) is mathematically safer and superior for small medical imaging cohorts.

2. **Tree Ensembles (GBDT) Degrade in Small-Sample Continuous Recovery:**
   - Decision trees partition the 83 training samples into small leaf buckets ($N \approx 6\text{--}10$), creating step-function approximations that have high variance on continuous visual acuity.
   - Smooth $L_2$-regularized Linear Ridge regression outperforms tree-based boosting by $\Delta R^2 \approx +0.58$.

3. **Feature Multiplicity (15 Interaction Features vs 7 Core Features):**
   - Adding 8 interaction features slightly diluted the primary linear signal of `VA_baseline` ($R^2$ dropped from $0.475 \to 0.421$).

4. **Conformal Prediction Insight:**
   - Heteroscedastic adaptive scaling works well on well-regularized models ($89.6\%$ coverage on Baseline VA), but completely breaks down if the underlying predictor overfits.
