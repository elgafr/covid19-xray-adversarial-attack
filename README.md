# Robustness Under Attack: Assessing Adversarial Fragility in Deep Learning Models for COVID-19 Radiography Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C?style=for-the-badge&logo=pytorch)
![DenseNet121](https://img.shields.io/badge/Model-DenseNet--121-009688?style=for-the-badge)
![JEEEMI](https://img.shields.io/badge/JEEEMI-Vol.8%20No.2%20(2026)-4051B5?style=for-the-badge)
![License](https://img.shields.io/badge/License-CC%20BY--SA%204.0%20%2F%20MIT-green?style=for-the-badge)

**Department of Informatics Engineering — Faculty of Engineering**  
**Universitas Muhammadiyah Malang (UMM)**

[![Documentation](https://img.shields.io/badge/Documentation-MkDocs_Material-526CFE?style=for-the-badge&logo=materialformkdocs&logoColor=white)](https://hisyam99.github.io/covid19-xray-adversarial-attack/)
[![DOI](https://img.shields.io/badge/Journal_DOI-10.35882%2Fjeeemi.v8i2.1506-00629B?style=for-the-badge)](https://doi.org/10.35882/jeeemi.v8i2.1506)
[![GitHub](https://img.shields.io/badge/Source_Code-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/hisyam99/covid19-xray-adversarial-attack)

</div>

---

## Publication & Research Team

This repository contains the official source code, experimental logs, complete visualizations, and technical documentation for the peer-reviewed research article:

> **Kamil, M. H., Farma, E. P. T., & Basuki, S. (2026).** *Robustness Under Attack: Assessing Adversarial Fragility in Deep Learning Models for COVID-19 Radiography Prediction.* **Journal of Electronics, Electromedical Engineering, and Medical Informatics (JEEEMI)**, Vol. 8, No. 2, April 2026, pp. 769–788. DOI: [10.35882/jeeemi.v8i2.1506](https://doi.org/10.35882/jeeemi.v8i2.1506)

| Role | Name | Contribution Focus | Contact Email |
|:-----|:-----|:-------------------|:--------------|
| **First Author** | **Muhammad Hisyam Kamil** | Study conceptualization, primary experiments, FGSM & PGD evaluation, manuscript drafting | `hisyamkamil99@webmail.umm.ac.id` |
| **Co-Author** | **Elga Putri Tri Farma** | Data preprocessing pipeline, iterative PGD sensitivity analysis, Grad-CAM visualization analysis | `elgafarma@webmail.umm.ac.id` |
| **Corresponding Author & Supervisor** | **Setio Basuki, S.T., M.T., Ph.D.** | Project supervision, methodological validation, manuscript review and refinement | `setio_basuki@umm.ac.id` |

---

## Executive Summary

Convolutional Neural Network (CNN) architectures have demonstrated high diagnostic precision in classifying pulmonary diseases from chest X-ray (CXR) images. However, this study empirically demonstrates that unprotected CNN models remain acutely vulnerable to white-box gradient-based adversarial perturbations.

Utilizing the **COVID-19 Radiography Dataset** (**21,165 CXR images** across four diagnostic categories: *COVID-19*, *Lung Opacity*, *Normal*, and *Viral Pneumonia*), a pretrained **DenseNet-121** model was fine-tuned and subjected to **Fast Gradient Sign Method (FGSM)** and **Projected Gradient Descent (PGD)** attacks under varying $L_\infty$-bounded epsilon ($\epsilon$) constraints. Beyond quantitative metrics, this study introduces and quantifies **Semantic Decoupling** and **Semantic Inversion** through **Gradient-weighted Class Activation Mapping (Grad-CAM)** and **Cosine Similarity ($S_c$)** analysis.

<div align="center">
  <img src="images/architecture_diagram_v1.png" alt="Architecture and Adversarial Attacks Pipeline" width="850">
  <p><em>Figure 1: System Architecture and Adversarial Attacks Pipeline (DenseNet-121, FGSM, PGD, and Grad-CAM Evaluation).</em></p>
</div>

---

## Key Experimental Results

### 1. Dataset Distribution (COVID-19 Radiography Dataset)

| Diagnostic Category | Total Images | Percentage | Train (80%) | Validation (10%) | Test (10%) |
|:--------------------|-------------:|-----------:|------------:|-----------------:|-----------:|
| **Normal** | 10,192 | 48.15% | 8,154 | 1,019 | 1,019 |
| **Lung Opacity** | 6,012 | 28.41% | 4,810 | 601 | 602 |
| **COVID-19** | 3,616 | 17.08% | 2,893 | 362 | 362 |
| **Viral Pneumonia** | 1,345 | 6.36% | 1,075 | 134 | 134 |
| **Total** | **21,165** | **100.00%** | **16,932** | **2,116** | **2,117** |

### 2. Performance Comparison at Critical Threshold ($\epsilon = 0.1/255 \approx 0.000392$)

| Category | Metric | Original (Clean) | FGSM ($\epsilon = 0.1/255$) | PGD ($\epsilon = 0.1/255$) |
|:---------|:-------|:----------------:|:---------------------------:|:--------------------------:|
| **COVID-19** | Precision / Recall / **F1** | 0.9914 / 0.9586 / **0.9747** | 0.4844 / 0.5138 / **0.4987** ($\downarrow 48.83\%$) | 0.1540 / 0.1796 / **0.1658** ($\downarrow 82.99\%$) |
| **Lung Opacity** | Precision / Recall / **F1** | 0.9566 / 0.9153 / **0.9355** | 0.4576 / 0.4568 / **0.4572** ($\downarrow 51.13\%$) | 0.0960 / 0.1130 / **0.1038** ($\downarrow 88.90\%$) |
| **Normal** | Precision / Recall / **F1** | 0.9368 / 0.9745 / **0.9553** | 0.6989 / 0.6742 / **0.6863** ($\downarrow 28.16\%$) | 0.4095 / 0.3288 / **0.3647** ($\downarrow 61.82\%$) |
| **Viral Pneumonia** | Precision / Recall / **F1** | 0.9847 / 0.9627 / **0.9736** | 0.7181 / 0.7985 / **0.7562** ($\downarrow 22.33\%$) | 0.4024 / 0.5075 / **0.4488** ($\downarrow 53.91\%$) |
| **Global ($N=2,117$)** | **Accuracy** | **95.42%** | **59.28%** | **25.32%** |
| **Global ($N=2,117$)** | **95% Confidence Interval** | $94.53\% - 96.31\%$ | $57.19\% - 61.37\%$ | $23.42\% - 27.12\%$ |
| **Global ($N=2,117$)** | **Attack Success Rate (ASR)** | — | **40.72%** | **74.68%** |
| **Global ($N=2,117$)** | **Avg. Prediction Confidence** | **96.05%** | **82.16%** | **85.42%** |
| **Global ($N=2,117$)** | **McNemar's Statistical Test** | Baseline | $\chi^2 = 763.00 \ (p \ll 0.001)$ | $\chi^2 = 1483.00 \ (p \ll 0.001)$ |

### 3. Sub-Pixel Sensitivity Profile: FGSM vs PGD

<div align="center">
  <img src="images/robustness_profile_degradasi_akurasi_pada_noise_sub_pixel_dan_mikro.png" alt="Robustness Profile" width="750">
  <p><em>Figure 2: Robustness Profile of DenseNet-121 under Sub-Pixel FGSM and PGD Perturbations.</em></p>
</div>

| Epsilon ($\epsilon$) | Decimal Value | FGSM Accuracy (%) | FGSM ASR (%) | PGD Accuracy (%) | PGD ASR (%) |
|:--------------------:|:-------------:|:-----------------:|:------------:|:----------------:|:-----------:|
| **0 (Clean)** | `0.000000` | **95.42%** | — | **95.42%** | — |
| **0.01/255** | `0.000039` | — | — | **92.82%** | 7.18% |
| **0.025/255** | `0.000098` | — | — | **86.30%** | 13.70% |
| **0.05/255** | `0.000196` | — | — | **66.65%** | 33.35% |
| **0.1/255** | `0.000392` | **59.28%** | 40.72% | **25.32% / 25.51%** | 74.68% / 74.49% |
| **0.2/255** | `0.000784` | **32.88%** | 67.12% | **1.51%** | 98.49% |
| **0.3/255** | `0.001176` | **21.73%** | 78.27% | **0.05% / 0.14%** | 99.95% / 99.86% |
| **0.4/255** | `0.001569` | **15.16%** | 84.84% | **0.05%** | 99.95% |
| **0.5/255** | `0.001961` | **12.14%** | 87.86% | **0.05%** | 99.95% |
| **0.6/255 – 0.9/255** | `0.002353 – 0.003529` | **10.39% → 9.02%** | 89.61% → 90.98% | **0.00%** | 100.00% |

### 4. Grad-CAM Interpretability: Semantic Decoupling vs Semantic Inversion

<div align="center">
  <img src="images/Visualisasi_Grad-CAM_(Clean_vs_FGSM_vs_PGD).png" alt="Visualisasi Grad-CAM Clean vs FGSM vs PGD" width="850">
  <p><em>Figure 3: Spatial Attention Shift (Grad-CAM) across Clean, FGSM ($\epsilon=0.1/255$), and PGD ($\epsilon=0.1/255$) Inputs.</em></p>
</div>

---

## Repository Structure

```text
covid19-xray-adversarial-attack/
├── .devcontainer/
│   └── devcontainer.json               # Development container configuration
├── .github/
│   └── workflows/
│       └── docs.yml                    # GitHub Actions CI/CD workflow for GitHub Pages
├── datasets/
│   └── README.md                       # COVID-19 Radiography Dataset documentation
├── docs/                               # MkDocs Material documentation source
│   ├── about/                          # Research background, objectives, and author profiles
│   ├── dataset/                        # Dataset statistics, stratified split, and preprocessing
│   ├── methodology/                    # Mathematical formulations (DenseNet-121, FGSM, PGD, Grad-CAM)
│   ├── experiments/                    # Training convergence, hyperparameters, and execution logs
│   ├── results/                        # Quantitative tables, confusion matrices, and visualization gallery
│   ├── implementation/                 # Setup guide, code walkthrough, and rendered Jupyter notebook
│   ├── conclusions/                    # Summary, empirical findings, and defense recommendations
│   ├── images/                         # Complete experimental plots and Grad-CAM visualizations
│   ├── publications/                   # Published JEEEMI journal article (PDF)
│   ├── overrides/                      # Material theme template overrides
│   ├── stylesheets/                    # Custom stylesheet for light/dark themes and notebooks
│   ├── javascripts/                    # MathJax 3 configuration for LaTeX equations
│   ├── index.md                        # Documentation home page
│   ├── references.md                   # Complete bibliography (53 references)
│   └── tags.md                         # Documentation tag index
├── images -> docs/images               # Symlink to documentation image assets
├── models/
│   └── README.md                       # Model checkpoint specification (best_densenet121_covid_4class.pth)
├── publications -> docs/publications   # Symlink to published journal PDF
├── results/
│   └── full_hasil.txt                  # Complete execution output log from GPU evaluation
├── src/
│   └── notebook/
│       └── code3v3-hisyam_v4.ipynb     # Main experimental Jupyter Notebook (45 cells)
├── .gitattributes                      # Git LFS tracking for model weights (*.pth)
├── mkdocs.yml                          # MkDocs configuration file
├── requirements-docs.txt               # Documentation build dependencies
├── requirements.txt                    # Full project dependencies (PyTorch, Grad-CAM, MkDocs)
├── sync_docs.sh                        # Script to synchronize notebook into docs/implementation/
└── README.md                           # Main repository documentation
```

---

## Quick Start & Documentation Build

### 1. Environment Setup

```bash
git clone https://github.com/hisyam99/covid19-xray-adversarial-attack.git
cd covid19-xray-adversarial-attack

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Serve Documentation Locally

```bash
./sync_docs.sh
mkdocs serve --dev-addr 127.0.0.1:8000
```

Open `http://127.0.0.1:8000/covid19-xray-adversarial-attack/` in your browser. To build the static production site:

```bash
mkdocs build --strict
```

---

## Citation

If you refer to this research, codebase, or experimental findings, please cite our paper:

```bibtex
@article{kamil2026robustness,
  author    = {Kamil, Muhammad Hisyam and Farma, Elga Putri Tri and Basuki, Setio},
  title     = {Robustness Under Attack: Assessing Adversarial Fragility in Deep Learning Models for COVID-19 Radiography Prediction},
  journal   = {Journal of Electronics, Electromedical Engineering, and Medical Informatics (JEEEMI)},
  volume    = {8},
  number    = {2},
  pages     = {769--788},
  month     = {April},
  year      = {2026},
  issn      = {2656-8632},
  doi       = {10.35882/jeeemi.v8i2.1506},
  url       = {https://doi.org/10.35882/jeeemi.v8i2.1506}
}
```

---

## License

- **Published Journal Article:** Licensed under the *Creative Commons Attribution-ShareAlike 4.0 International License* ([CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)).
- **Source Code & Documentation:** Released under the **MIT License** for academic and research purposes.
