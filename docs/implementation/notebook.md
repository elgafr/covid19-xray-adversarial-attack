# Jupyter Notebook Guide & Code Walkthrough

All experimental pipelines in this study are consolidated within a single self-contained Jupyter Notebook: **`src/notebook/code3v3-hisyam_v4.ipynb`** (**45 cells** spanning 15 structured experimental sections).

[:fontawesome-solid-book-open: **Open Full Rendered Notebook (`code3v3-hisyam_v4.ipynb`)**](code3v3-hisyam_v4.ipynb){ .md-button .md-button--primary }

---

## Structural Map of the 45 Cells in `code3v3-hisyam_v4.ipynb`

| Section / Stage | Cell Range | Functional Description & Primary Outputs |
|:----------------|:----------:|:-----------------------------------------|
| **Step 1: Directory Inspection & Image Counting** | `Cells 0 – 6` | Retrieves `best_densenet121_covid_4class.pth`, scans the 4 class folders (`COVID`: 3,616, `Lung_Opacity`: 6,012, `Normal`: 10,192, `Viral Pneumonia`: 1,345 $\rightarrow$ Total **21,165**), performs the 80:10:10 stratified split (16,932 / 2,116 / 2,117), and plots `sample_dataset.png`. |
| **Part 1: Imports & Data Preparation** | `Cells 7 – 8` | Self-contained definition of `CovidDataset`, ImageNet normalization and training augmentation transforms, and initialization of `dataloaders['train']`, `dataloaders['val']`, and `dataloaders['test']`. |
| **Part 2: DenseNet-121 Definition & Training** | `Cells 9 – 10` | Initializes `models.densenet121(weights='DEFAULT')`, replaces `model_ft.classifier = nn.Linear(1024, 4)`, defines `train_model` with Early Stopping (`patience=5`), `plot_training_results`, and checkpoint loading logic. |
| **Part 3: White-Box Attack Functions (FGSM & PGD)** | `Cells 11 – 12` | Implements `fgsm_attack(model, images, labels, epsilon)` (Equation 4) and `pgd_attack(model, images, labels, eps, alpha, steps)` (Equation 5) with normalized-space scaling $\epsilon_{\text{norm}} = \epsilon / \text{STD\_TENSOR}$. |
| **Part 4: Complete PGD Evaluation (Clean vs. Adv)** | `Cells 13 – 14` | Evaluates the baseline *Original Test Set* (**95.42%**, `cm_original_test_set.png`) and initial PGD test set (`cm_pgd_test_set.png`). |
| **Parts 5 & 5.5: Complete FGSM Evaluation & Combined Summary** | `Cells 15 – 17` | Evaluates FGSM at $\epsilon = 0.1/255$ (**59.28%**, ASR **40.72%**, `cm_fgsm_test_set.png`) and merges FGSM and PGD summary tables. |
| **Part 6: Complete Grad-CAM Visualization (Clean vs. FGSM vs. PGD)** | `Cells 18 – 20` | Installs `grad-cam`, targets `model.features[-1]`, and renders the $4 \times 6$ comparison grid (`Visualisasi_Grad-CAM_(Clean_vs_FGSM_vs_PGD).png`). |
| **Part 7: 5-Sample Grad-CAM Visualization per Class** | `Cells 21 – 22` | Defines `visualize_class_block` to plot 5 representative test samples per class across 6 columns (`visualisasi_5_sampel_kelas_*.png`). |
| **Part 8: 8-Column Grad-CAM + Amplified Noise Visualization** | `Cells 23 – 24` | Defines `compute_noise_diff` ($15\times$ amplification) and `visualize_class_block_with_noise` across 8 columns (`visualisasi_5_sampel_kelas_*_with_noise.png`). |
| **Parts 9a / 13 & 14: FGSM Sub-Pixel Sensitivity & Progression ($0.1 - 0.9/255$)** | `Cells 25 – 27` | Evaluates FGSM across 9 sub-pixel steps ($0.1/255$ to $0.9/255$), plots `analisis_batas_toleransi_fgsm.png`, and renders 10-column progression grids (`analisis_sub_pixel_fgsm_0.1-0.9_kelas_*.png`). |
| **Part 10 (FGSM & Auto-Enhanced Noise Revision)** | `Cells 28 – 29` | Renders FGSM progression across $\epsilon \in \{0.01, \ 0.025, \ 0.05, \ 0.1\}$ using Auto-Scaled Magma Heatmaps (`analisis_progresi_fgsm_(auto-enhanced-noise)_kelas_*.png` — **Fig. 10a–d in JEEEMI**). |
| **Part 9b (Revised): Custom PGD Sensitivity ($0.01 - 0.3/255$)** | `Cells 30 – 31` | Evaluates PGD with $\alpha = \epsilon/4, N=10$ across $\epsilon \in \{0.01, \ 0.025, \ 0.05, \ 0.1, \ 0.3\}/255$ (**Table 6 in JEEEMI**) and plots `model_sensitivity_pgd_0.01-0.3_255.png`. |
| **Part 10 & Revision: PGD Progression Visualization** | `Cells 32 – 36` | Generates PGD progression grids per class (`analisis_progresi_pgd_kelas_*.png` — **Fig. 9a–d in JEEEMI**) and Auto-Enhanced Magma Heatmap variants (`analisis_progresi_pgd_(auto-enhanced-noise)_kelas_*.png`). |
| **Parts 11 & 12: Sub-Pixel PGD Sensitivity & Progression ($0.1 - 0.9/255$)** | `Cells 37 – 40` | Evaluates PGD across 9 sub-pixel steps ($0.1/255$ to $0.9/255$), plots `analisis_batas_toleransi_pgd.png`, and renders 10-column progression grids (`analisis_sub_pixel_pgd_0.1-0.9_kelas_*.png`). |
| **Dedicated Micro-Attack Evaluation ($\epsilon = 0.1/255$)** | `Cells 41 – 42` | Unified evaluation of Clean vs. FGSM vs. PGD at $\epsilon = 0.1/255$ ($\alpha = \epsilon/4, N=10$) producing **Tables 2, 3, and 4 in JEEEMI** (Accuracy: Clean 95.42%, FGSM 59.28%, PGD 25.32%; Confidence: 96.05%, 82.16%, 85.42%), `cm_fgsm_vs_pgd_0.1.png`, and `SAMPEL_PREDIKSI_VISUAL_(Clean_vs_FGSM_vs_PGD)_0.1.png`. |
| **Part 15: Final Report & Comprehensive Synthesis** | `Cells 43 – 44` | Consolidates all summary tables, plots `head-to-head-fgsm-pgd.png` and **Fig. 11 in JEEEMI** (`robustness_profile_degradasi_akurasi_pada_noise_sub_pixel_dan_mikro.png`), and outputs the final report. |

---

[:octicons-arrow-right-24: Open Full Rendered Notebook (`code3v3-hisyam_v4.ipynb`)](code3v3-hisyam_v4.ipynb){ .md-button .md-button--primary }
