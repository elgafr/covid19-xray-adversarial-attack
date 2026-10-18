# Project Overview & Clinical Background

## Background: Deep Learning Precision vs. Adversarial Fragility in Radiology

The rapid advancement of Deep Learning (DL), particularly **Convolutional Neural Network (CNN)** architectures, has transformed automated diagnosis of thoracic diseases from Chest X-Ray (CXR) and Computed Tomography (CT) images. By automatically learning hierarchical feature representations—ranging from low-level edges and textures to high-level semantic patterns such as bilateral ground-glass opacities (GGO) and pulmonary consolidation—CNNs serve as a core component of modern clinical decision support systems.

Despite achieving high diagnostic accuracy on clean datasets, deep neural networks remain inherently vulnerable to **adversarial attacks**. Mathematically, an adversarial attack applies a small, optimized perturbation $\delta$ to a clean input image $x$ to generate an adversarial example:

$$
x' = x + \delta, \quad \text{subject to } \|\delta\|_\infty \le \epsilon
$$

where $\epsilon$ (epsilon) constrains the maximum allowable pixel modification so that the alteration remains **visually imperceptible to clinicians and radiologists**.

!!! danger "Patient Safety Risks in Clinical Deployment"
    In clinical settings, high-confidence misclassifications caused by adversarial perturbations directly compromise patient safety:
    
    - **Adversarial False Negatives:** Misclassifying a positive *COVID-19* scan as *Normal* with $>85\%$ confidence can result in the premature discharge of contagious patients and delayed critical care.
    - **Adversarial False Positives:** Misclassifying a healthy (*Normal*) scan as *COVID-19* or *Viral Pneumonia* with near-$100\%$ confidence can trigger unnecessary quarantine protocols, inappropriate treatment, and misallocation of medical resources.

---

## Research Gap

Prior studies on adversarial attacks in medical imaging predominantly rely on surface-level quantitative metrics such as accuracy drops, AUC degradation, or Attack Success Rate (ASR). Relying solely on classification scores obscures the underlying cause of model failure:

1. **Absence of Spatial Attention Tracking on the COVID-19 Radiography Dataset:** Existing literature lacks a systematic investigation using **Grad-CAM** to track how a **DenseNet-121** model's spatial attention shifts across the four classes of the *COVID-19 Radiography Dataset* under sub-pixel **FGSM** and **PGD** attacks.
2. **Unresolved Failure Mechanism:** Quantitative accuracy metrics alone cannot distinguish whether a model is merely flipping a label near a decision boundary or fundamentally **looking away from the lungs** toward non-diagnostic background artifacts.

---

## Core Scientific Contributions: Semantic Decoupling & Semantic Inversion

To bridge this gap, this study combines white-box robustness evaluation (**FGSM** and **PGD**) across a fine-grained sub-pixel spectrum ($\epsilon \in \{0.01/255 \dots 0.9/255\}$) with spatial interpretability using **Gradient-weighted Class Activation Mapping (Grad-CAM)** and **Cosine Similarity ($S_c$)**:

### 1. Semantic Decoupling ($S_c \ll 0.50$)
We formally define **Semantic Decoupling** as the systematic, forced separation between the true pathological features present in the image and the model's learned spatial attention. Under this condition (predominantly observed in *Normal* and *Lung Opacity* classes), the model's predictive focus diverges completely from clinically relevant pulmonary parenchyma and anchors onto spurious, non-diagnostic background noise (such as clavicles, rib cages, diaphragmatic borders, and dark image corners).

### 2. Semantic Inversion ($S_c \approx 0.79$)
Conversely, in the *COVID-19* category, we identify **Semantic Inversion**: spatial correlation between clean and adversarial activation maps remains high ($S_c \approx 0.79$), indicating that spatial attention stays anchored within pulmonary structures. However, high-frequency adversarial perturbations synthesize localized textural features within the Region of Interest (ROI), forcing the model to misclassify *COVID-19* cases as *Viral Pneumonia* with high confidence.

---

## Summary of Key Contributions

1. **Comprehensive Robustness Evaluation of DenseNet-121:** Quantifying the resilience of a dense-connectivity CNN against single-step (FGSM) and iterative (PGD) white-box attacks across 21,165 four-class CXR images.
2. **Empirical Proof of Sub-Pixel Fragility:** Demonstrating that at $\epsilon = 0.1/255$, iterative PGD attacks reduce global accuracy from **95.42%** to **25.32%** (ASR **74.68%**), reaching total predictive collapse (**0.05%** accuracy) at $\epsilon = 0.3/255$.
3. **Mechanistic Analysis of Asymmetric Class Vulnerability:** Explaining why classes defined by subtle, low-contrast, high-frequency textural anomalies (*Lung Opacity* $\downarrow 88.90\%$ F1 and *COVID-19* $\downarrow 82.99\%$ F1 under PGD) are disproportionately compromised compared to classes with macro-structural low-frequency infiltrates (*Viral Pneumonia* $\downarrow 53.91\%$ F1).
4. **Topological Quantification via Grad-CAM Cosine Similarity:** Establishing the mathematical link between empirical metric collapse and spatial attention divergence (*Semantic Decoupling*).

---

[:octicons-arrow-right-24: Continue to Research Objectives & Scope](objectives.md){ .md-button .md-button--primary }
