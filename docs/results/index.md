# Experimental Results & Analysis

Comprehensive documentation of quantitative evaluation metrics, statistical significance tests, sub-pixel sensitivity curves, head-to-head **FGSM vs. PGD** comparisons, and **Grad-CAM** spatial interpretability (*Semantic Decoupling* vs. *Semantic Inversion*).

<div class="grid cards" markdown>

-   :material-chart-box-outline: **[Baseline Performance Overview](overview.md)**

    ---
    
    DenseNet-121 performance on the *Original Clean Test Set* ($N=2,117$), per-class metrics, training curves, and baseline *Confusion Matrix* (Fig. 5 & Fig. 6).

-   :material-sword-cross: **[Clean vs. FGSM vs. PGD Comparison](comparison.md)**

    ---
    
    Comprehensive comparison at the critical threshold $\epsilon = 0.1/255$ (Tables 2, 3, and 4 in JEEEMI), confidence score analysis, and McNemar's test.

-   :material-chart-timeline-variant-shimmer: **[Sub-Pixel Attack Results](attack-results.md)**

    ---
    
    Robustness profiles across the sub-pixel spectrum ($\epsilon = 0.01/255$ to $0.9/255$, Tables 5 & 6 in JEEEMI) and class-wise ASR breakdown.

-   :material-image-multiple: **[Complete Visualizations Gallery](visualizations.md)**

    ---
    
    Complete gallery of Grad-CAM heatmaps, *Auto-Enhanced Noise* progressions (Fig. 9 & Fig. 10), sub-pixel grids, and 5-sample class blocks.

-   :material-microscope: **[Error Analysis & Semantic Decoupling](error-analysis.md)**

    ---
    
    Mechanistic spatial-frequency biomarker analysis, proof of *Semantic Decoupling* vs. *Semantic Inversion*, and *Cosine Similarity* ($S_c$) evaluation.

</div>

---

[:octicons-arrow-right-24: Continue to Baseline Performance Overview](overview.md){ .md-button .md-button--primary }
