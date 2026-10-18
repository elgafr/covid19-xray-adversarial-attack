# Research Methodology

Comprehensive documentation of the operational framework, mathematical formulations (**Equations 1 to 12**), **DenseNet-121** architecture, white-box **FGSM** and **PGD** attack algorithms, statistical validation tests, and **Grad-CAM** interpretability protocols.

<div class="grid cards" markdown>

-   :material-sitemap: **[Methodology Overview](overview.md)**

    ---
    
    End-to-end operational stages (JEEEMI Fig. 1) and research execution workflow for FGSM and PGD evaluations.

-   :material-image-edit: **[Preprocessing Formulation](preprocessing.md)**

    ---
    
    Input vector standardization and mathematical formulation of ImageNet Z-score normalization (**Equation 1**).

-   :material-network: **[DenseNet-121 Architecture](densenet.md)**

    ---
    
    Dense connectivity propagation (**Equation 2**), four Dense Blocks and Transition Layers, and 4-class linear classification head.

-   :material-school: **[Training Strategy & Loss](training.md)**

    ---
    
    Transfer learning fine-tuning protocol, Categorical Cross-Entropy Loss (**Equation 3**), Adam optimization, and Early Stopping.

-   :material-shield-alert: **[Adversarial Attacks (FGSM & PGD)](adversarial.md)**

    ---
    
    Mathematical formulations of **FGSM (Equation 4)** and **PGD (Equation 5)**, $L_\infty$-ball coverage guarantee ($N\cdot\alpha = 2.5\epsilon > 2\epsilon$), and normalized-space scaling.

-   :material-chart-bell-curve-cumulative: **[Evaluation Metrics, Stats & Grad-CAM](metrics.md)**

    ---
    
    Classification metrics (**Equations 6–9**), ASR, 95% CI, McNemar's test, **Grad-CAM (Equations 10–11)**, and **Cosine Similarity $S_c$ (Equation 12)**.

</div>

---

[:octicons-arrow-right-24: Continue to Methodology Overview](overview.md){ .md-button .md-button--primary }
