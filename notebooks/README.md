# Experimental Jupyter Notebooks

This directory contains the primary self-contained Jupyter Notebook implementing the complete experimental pipeline:

- **Notebook File:** [`covid19_xray_adversarial_robustness.ipynb`](covid19_xray_adversarial_robustness.ipynb) (45 cells, Parts 1–15)

## Structural Overview of `covid19_xray_adversarial_robustness.ipynb`

| Section | Cells | Description |
|:--------|:-----:|:------------|
| **Step 1 & Part 1: Dataset Ingestion & Stratified Split** | `0 – 8` | Scans the 21,165 CXR images across 4 classes, performs the 80:10:10 stratified split (`16,932` train / `2,116` val / `2,117` test), defines `CovidDataset`, and plots training batch samples. |
| **Part 2: DenseNet-121 Architecture & Training** | `9 – 10` | Initializes `models.densenet121(weights='DEFAULT')` with a 4-class linear head, defines `train_model` (`EPOCHS=50`, `PATIENCE=5`, `lr=1e-4`), and loads `best_densenet121_covid_4class.pth`. |
| **Part 3: White-Box Adversarial Attacks (FGSM & PGD)** | `11 – 12` | Implements `fgsm_attack` (Eq. 4) and `pgd_attack` (Eq. 5) with normalized-space perturbation scaling ($\epsilon / \text{STD\_TENSOR}$). |
| **Parts 4, 5 & 5.5: Clean, FGSM & PGD Evaluations** | `13 – 17` | Evaluates baseline clean accuracy (**95.42%**), FGSM accuracy (**59.28%**), PGD accuracy, classification reports, and confusion matrices. |
| **Parts 6, 7 & 8: Grad-CAM & Amplified Noise Visualization** | `18 – 24` | Computes Grad-CAM on `model.features[-1]`, generating $4 \times 6$ overview grids, 5-sample class grids (6 columns), and 8-column grids with $15\times$ amplified noise. |
| **Parts 9a / 13, 14 & Part 10 (FGSM): Sub-Pixel FGSM** | `25 – 29` | Executes the 9-step FGSM sub-pixel sweep ($0.1/255 - 0.9/255$), 10-column progression grids ($50\times$ noise), and Auto-Enhanced Magma Noise progressions (Fig. 10a–d). |
| **Parts 9b, 10, 11 & 12 (PGD): Sub-Pixel PGD** | `30 – 40` | Executes the custom low-range PGD sweep ($0.01/255 - 0.3/255$ with $\alpha = \epsilon/4, N = 10$), PGD progression grids (Fig. 9a–d), and full sub-pixel PGD sweep ($0.1/255 - 0.9/255$). |
| **Dedicated Micro-Attack Evaluation & Part 15 Final Report** | `41 – 44` | Unified Clean vs. FGSM vs. PGD evaluation at $\epsilon = 0.1/255$ (Tables 2, 3, and 4 in JEEEMI), side-by-side confusion matrices, Fig. 11 robustness profile, and final report synthesis. |
