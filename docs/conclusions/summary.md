# Summary of Conclusions

This section presents the primary conclusions from **Section V (Conclusion) of our JEEEMI article**.

---

## Primary Conclusions

This study evaluated the adversarial robustness of a **DenseNet-121** model against **Fast Gradient Sign Method (FGSM)** and **Projected Gradient Descent (PGD)** attacks in classifying four-class COVID-19 chest X-ray images (*COVID-19*, *Lung Opacity*, *Normal*, and *Viral Pneumonia*):

1. **Critical Fragility Behind High Baseline Accuracy:**  
   While the unprotected DenseNet-121 architecture achieved a baseline accuracy of **95.42%** (95% CI: 94.53%–96.31%, with F1-scores exceeding $0.97$ on *COVID-19* and *Viral Pneumonia*) on clean test data, introducing $L_\infty$-constrained sub-pixel perturbations at $\epsilon = 0.1/255$ ($\approx 0.000392$) significantly reduced overall accuracy to **59.28%** under single-step **FGSM** (ASR **40.72%**, $\chi^2 = 763.00, \ p \ll 0.001$) and **25.32%** under iterative **PGD** (ASR **74.68%**, $\chi^2 = 1483.00, \ p \ll 0.001$).
2. **Total Predictive Collapse under Iterative PGD vs. FGSM Plateau:**  
   Escalating PGD perturbation strength to $\epsilon = 0.3/255$ ($\approx 0.12\%$ of dynamic pixel range) forced **total predictive collapse** (**0.05% – 0.14%** accuracy, **99.86% – 99.95%** ASR), whereas single-step FGSM reached a linear approximation plateau around **9.02%** accuracy at $\epsilon = 0.9/255$.
3. **Mechanistic Validation via Grad-CAM (*Semantic Decoupling*):**  
   Interpretability analysis using **Grad-CAM** and **Cosine Similarity ($S_c$)** confirmed that this severe performance collapse is driven by **Semantic Decoupling** ($S_c = 0.32$ on *Lung Opacity* and $S_c = 0.38$ on *Normal*), where the model's predictive focus deterministically shifts from clinically relevant pulmonary features to spurious background noise.
4. **Clinical Deployment Implications:**  
   These findings empirically prove that high baseline accuracy on clinical benchmarks does not provide a guarantee of model reliability or patient safety under adversarial conditions, particularly because mispredictions occur with high average confidence (**85.42%** under PGD and **82.16%** under FGSM).

---

[:octicons-arrow-right-24: Continue to Key Empirical Findings](findings.md){ .md-button .md-button--primary }
