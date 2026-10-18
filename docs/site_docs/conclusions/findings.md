# Key Empirical Findings

The experimental evaluation establishes four primary empirical findings:

---

## 1. Systemic Confidence-Accuracy Misalignment

On clean data, high prediction certainty (**96.05%** average softmax confidence) correlates directly with high diagnostic accuracy (**95.42%**). Under **PGD ($\epsilon = 0.1/255$)**, however, while global accuracy collapses to random-guessing levels (**25.32%**), the model maintains an average confidence score of **85.42%** (frequently reaching **100.00%** on individual misclassified scans). Simple confidence thresholding is therefore ineffective for detecting adversarial failures in clinical pipelines.

---

## 2. Iterative PGD Exploitation vs. Single-Step FGSM Underfitting

- **FGSM (Single-Step Linear Approximation):** Operates under a local linearity hypothesis, taking a single step along the raw gradient sign. While generating structured edge artifacts along ribs and clavicles, it frequently underfits curved adversarial subspaces, leaving **59.28%** accuracy at $\epsilon = 0.1/255$.
- **PGD (Iterative First-Order Optimization):** By executing $N = 10$ steps of size $\alpha = \epsilon/4$ ($N \cdot \alpha = 2.5\epsilon > 2\epsilon$) and recalculating local gradients at each step, PGD navigates high-dimensional curvature in the loss landscape and reduces accuracy to **25.32%** at $\epsilon = 0.1/255$—a **33.96 percentage point** steeper drop than FGSM.

---

## 3. Spatial Frequency Governance of Class-Wise Vulnerability

Adversarial vulnerability across diagnostic categories is governed by the spatial frequency of the underlying radiological biomarkers:

- **High Vulnerability (High-Frequency, Low-Contrast Biomarkers):** **Lung Opacity** ($\downarrow 88.90\%$ F1 under PGD; $\downarrow 51.13\%$ F1 under FGSM) and **COVID-19** ($\downarrow 82.99\%$ F1 under PGD; $\downarrow 48.83\%$ F1 under FGSM) rely on subtle ground-glass opacities (GGO) located close to the decision boundary.
- **Relative Resilience (Low-Frequency, Macro-Structural Biomarkers):** **Viral Pneumonia** ($\downarrow 53.91\%$ F1 under PGD; $\downarrow 22.33\%$ F1 under FGSM) presents with dense macro-structural infiltrates dominating lower spatial frequencies, which are less susceptible to $L_\infty$-restricted high-frequency noise.

---

## 4. Dual Spatial Failure Modes: *Semantic Decoupling* vs. *Semantic Inversion*

Quantitative **Grad-CAM** and **Cosine Similarity ($S_c$)** analyses identify two distinct modes of internal feature disruption:

| Spatial Failure Mode | Dominant Classes | Cosine Similarity ($S_c$) | Internal Mechanism |
|:---------------------|:-----------------|:-------------------------:|:-------------------|
| **Semantic Decoupling** | *Lung Opacity*, *Normal* | $S_c \ll 0.50$ ($0.32 - 0.47$) | Spatial coherence of feature maps collapses; predictive focus detaches from pulmonary parenchyma and anchors onto non-diagnostic background edges. |
| **Semantic Inversion** | *COVID-19* | $S_c \approx 0.79$ | Spatial attention remains anchored on pulmonary structures, but adversarial noise synthesizes deceptive local textures within the ROI that invert the prediction to *Viral Pneumonia*. |

---

[:octicons-arrow-right-24: Continue to Defense Recommendations](recommendations.md){ .md-button .md-button--primary }
