# Dataset Introduction

## Dataset Origin & Background

The primary data source utilized in this study is the **COVID-19 Radiography Dataset**, a publicly accessible benchmark repository compiled by researchers from **Qatar University** (Doha, Qatar) and the **University of Dhaka** (Bangladesh) in collaboration with medical institutions in Pakistan and Malaysia, led by **Muhammad E. H. Chowdhury et al.** [1].

- **Official Kaggle Repository:** [https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- **Total Volume:** **21,165 Chest X-Ray (CXR)** images in `.png` format
- **Native Resolution:** $299 \times 299$ pixels (standardized to $224 \times 224$ pixels for DenseNet-121)
- **Directory Structure:** Each diagnostic category contains `images/` and `masks/` subdirectories. In accordance with our implementation (`covid19_xray_adversarial_robustness.ipynb`), all CXR images are loaded exclusively from the `images/` subdirectory of each category.

---

## Four Clinical Diagnostic Categories

The dataset categorizes chest radiographs into four distinct clinical groups:

1. **COVID-19 (`COVID` — 3,616 images / 17.08%)**  
   CXR scans from patients confirmed positive for SARS-CoV-2 infection via RT-PCR testing. Radiologically characterized by subtle, high-frequency, low-contrast bilateral ground-glass opacities (GGO) with peripheral and posterior distribution.
2. **Lung Opacity (`Lung_Opacity` — 6,012 images / 28.41%)**  
   CXR scans exhibiting non-COVID-19 pulmonary opacities caused by bacterial pneumonia, pulmonary edema, or fibrosis. Characterized by diffuse low-contrast grey-level anomalies that frequently overlap with normal vascular variations or viral opacities near the decision boundary.
3. **Normal (`Normal` — 10,192 images / 48.15%)**  
   CXR scans from healthy individuals without pulmonary pathology. Represents the majority class reflecting routine clinical screening populations, featuring clear and symmetrical lung fields.
4. **Viral Pneumonia (`Viral Pneumonia` — 1,345 images / 6.36%)**  
   CXR scans from patients diagnosed with non-COVID-19 viral pneumonia. Serves as a key differential diagnosis category, typically presenting with denser, macro-structural interstitial infiltrates and consolidation patterns dominating lower spatial frequencies.

---

## Representative Sample Visualizations

![Representative Samples COVID-19 Radiography Database](../images/fig2_dataset_representative_samples_stats.png){: .center width="850" }

*Figure: Figure 1 (Fig. 2 in JEEEMI): Representative samples from the COVID-19 Radiography Database (6 instances per class) showing spatial resolution ($299 \times 299$) and radiometric pixel intensity statistics ($\mu, \sigma$).*

![Batch Sample from Training DataLoader](../images/sample_dataset.png){: .center width="750" }

*Figure: Figure 2: Visualization of the first training mini-batch (4 samples) after un-normalizing ImageNet statistics in PyTorch (Cell 6 of the Notebook).*

---

[:octicons-arrow-right-24: Continue to Dataset Statistics & Partitioning](statistics.md){ .md-button .md-button--primary }
