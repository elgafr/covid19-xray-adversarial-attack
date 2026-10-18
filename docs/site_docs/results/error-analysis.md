# Error Analysis: Asymmetric Vulnerability, Semantic Decoupling & Semantic Inversion

This section examines the mechanistic causes of asymmetric feature fragility across the four diagnostic classes and demonstrates how **Grad-CAM** and **Cosine Similarity ($S_c$)** prove **Semantic Decoupling** and **Semantic Inversion** (corresponding to **Section III.B.2 and Section IV of the JEEEMI article**).

---

## 1. Spatial Frequency Governance of Asymmetric Class Vulnerability

Class-wise **F1-Score** comparisons at $\epsilon = 0.1/255$ reveal pronounced asymmetry in adversarial degradation across the four categories:

| Diagnostic Category | Clean F1-Score | FGSM F1-Score ($\downarrow$ Drop) | PGD F1-Score ($\downarrow$ Drop) | Radiological Biomarker & Spatial Frequency Profile |
|:--------------------|:--------------:|:---------------------------------:|:--------------------------------:|:---------------------------------------------------|
| **Lung Opacity** | `0.9355` | `0.4572` (**$\downarrow$ 51.13%**) | `0.1038` (**$\downarrow$ 88.90%**) | **High Frequency, Low Contrast:** Diffuse, non-specific opacities lie immediately adjacent to the decision boundary. |
| **COVID-19** | `0.9747` | `0.4987` (**$\downarrow$ 48.83%**) | `0.1658` (**$\downarrow$ 82.99%**) | **High Frequency, Fine Texture:** Relies on subtle bilateral ground-glass opacities (GGO) that are easily disrupted by $L_\infty$-bounded high-frequency noise. |
| **Normal** | `0.9553` | `0.6863` (**$\downarrow$ 28.16%**) | `0.3647` (**$\downarrow$ 61.82%**) | **Uniform Parenchyma:** Synthetic adversarial textures injected into clear lung fields or rib edges induce false opacity detections. |
| **Viral Pneumonia** | `0.9736` | `0.7562` (**$\downarrow$ 22.33%**) | `0.4488` (**$\downarrow$ 53.91%**) | **Low Spatial Frequency, Macro-Structural:** Characterized by dense macro-structural infiltrates dominating lower spatial frequencies, providing relative resilience against restricted gradient perturbations. |

!!! quote "Mechanistic Explanation (Section IV in JEEEMI)"
    *"Pathologically, COVID-19 and Lung Opacity rely on subtle, low-contrast, high-frequency textural anomalies. These localized features lie close to the decision boundary and are disrupted by $L_\infty$-constrained high-frequency noise. Viral Pneumonia typically presents with macro-structural infiltrates that dominate lower spatial frequencies, which are less susceptible to restricted gradient-based attacks. This demonstrates that adversarial vulnerability is governed by the spatial frequency of the diagnostic biomarkers."*

---

## 2. Structured Misclassification Trends ($\epsilon = 0.1/255$)

Inspection of the normalized confusion matrices in **Fig. 7 (FGSM)** and **Fig. 8 (PGD)** confirms that adversarial errors follow structured topological trajectories rather than uniform random noise:

### A. Under FGSM Attack ($\epsilon = 0.1/255$, Accuracy = 59.28%)
- **COVID-19 ($N=362$):** 186 instances (51.4%) remain correct; **99 instances (27.3%)** shift to *Normal* and **67 instances (18.5%)** shift to *Lung Opacity*.
- **Lung Opacity ($N=602$):** 275 instances (45.7%) remain correct; **172 instances (28.6%)** are misclassified as *Normal* and **149 instances (24.8%)** as *COVID-19*.
- **Normal ($N=1,019$):** 687 instances (67.4%) remain correct; **258 instances (25.3%)** shift to *Lung Opacity*.

### B. Under PGD Attack ($\epsilon = 0.1/255$, Accuracy = 25.32%)
- **Near-Total Collapse in Lung Opacity ($N=602$):** Only **68 instances (11.3%)** are classified correctly; **284 instances (47.2%)** shift to *Normal* and **224 instances (37.2%)** shift to *COVID-19*.
- **Severe False Negatives in COVID-19 ($N=362$):** Only **65 instances (18.0%)** remain correct; **151 instances (41.7%)** are misclassified as *Normal* and **119 instances (32.9%)** as *Lung Opacity*.
- **False Positives in Normal ($N=1,019$):** Over half of healthy scans (**515 instances / 50.5%**) are misdiagnosed as *Lung Opacity* and **121 instances (11.9%)** as *COVID-19*.

---

## 3. Grad-CAM Cosine Similarity ($S_c$): Semantic Decoupling vs. Semantic Inversion

Computing the **Cosine Similarity ($S_c$)** (Equation 12) between flattened clean ($M_{\text{clean}}$) and adversarial ($M_{\text{adv}}$) Grad-CAM activation maps reveals two distinct class-dependent modes of topological shift:

| Diagnostic Category | Mean $S_c$ (PGD Evaluation) | Mean / Representative $S_c$ (FGSM & JEEEMI) | PGD F1-Score Drop | Dominant Spatial Failure Mode |
|:--------------------|:---------------------------:|:-------------------------------------------:|:-----------------:|:------------------------------|
| **Lung Opacity** | **0.32** ($S_c \ll 0.50$) | **0.39** ($S_c \ll 0.50$) | $\downarrow 88.90\%$ | **Extreme Semantic Decoupling:** Attention diverges completely to shoulder, clavicle, and image borders. |
| **Normal** | **0.38** ($S_c \ll 0.50$) | **0.47** ($S_c < 0.50$) | $\downarrow 61.82\%$ | **Semantic Decoupling:** Focus vacates pulmonary parenchyma toward non-diagnostic corners. |
| **COVID-19** | **0.41** (Global Average) | **0.79** (*Semantic Inversion* Regime) | $\downarrow 82.99\%$ | **Semantic Inversion & Decoupling:** In samples with $S_c \approx 0.79$, spatial attention remains anchored on the lungs while synthetic local textures invert the prediction to *Viral Pneumonia*. |
| **Viral Pneumonia** | **0.58** ($S_c > 0.50$) | Highest Spatial Retention | $\downarrow 53.91\%$ | **Partial Retention:** Lower-lobe macro-structural consolidation retains partial attention focus. |

---

[:octicons-arrow-right-24: Continue to Implementation & Source Code Guide](../implementation/index.md){ .md-button .md-button--primary }
