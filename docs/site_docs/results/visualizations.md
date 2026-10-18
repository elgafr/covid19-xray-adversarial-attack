# Complete Visualizations Gallery (Grad-CAM, Noise & Epsilon Progression)

This gallery presents the complete set of validated experimental visualizations from our JEEEMI article (Figs. 1–11) and `covid19_xray_adversarial_robustness.ipynb`. Click any image to inspect it in full resolution via lightbox zoom.

---

## 1. PGD Progression Analysis per Class (Fig. 9a – 9d in JEEEMI)

Three-row progression grids (**Row 1:** Input Image + Prediction & Confidence, **Row 2:** Amplified Visual Noise $\times 20$, **Row 3:** Grad-CAM Heatmap) across $\epsilon \in \{0.01/255, \ 0.025/255, \ 0.05/255, \ 0.1/255\}$:

=== "Fig. 9(a) — Normal Class"

    <div align="center">
      <img src="../images/analisis_progresi_pgd_kelas_normal.png" alt="PGD Progression Normal" width="900">
      <p><em>Fig. 9(a) in JEEEMI: PGD progression on Normal class. Clean prediction: Normal (99.79%) shifts to Lung_Opacity (100%) at $\epsilon=0.01/255$ and COVID-19 (100%) at $\epsilon \ge 0.05/255$, accompanied by Grad-CAM Semantic Decoupling toward shoulder and clavicle borders.</em></p>
    </div>

=== "Fig. 9(b) — Lung Opacity Class"

    <div align="center">
      <img src="../images/analisis_progresi_pgd_kelas_lung_opacity.png" alt="PGD Progression Lung Opacity" width="900">
      <p><em>Fig. 9(b) in JEEEMI: PGD progression on Lung Opacity class. Clean prediction: Lung_Opacity (100%) shifts immediately to COVID-19 (100%) from $\epsilon=0.01/255$ onward as spatial attention disperses to image corners.</em></p>
    </div>

=== "Fig. 9(c) — COVID-19 Class"

    <div align="center">
      <img src="../images/analisis_progresi_pgd_kelas_covid.png" alt="PGD Progression COVID-19" width="900">
      <p><em>Fig. 9(c) in JEEEMI: PGD progression on COVID-19 class. Clean prediction: COVID (100%) shifts to Normal (100%) at $\epsilon=0.01 - 0.05/255$ and to Viral Pneumonia (100%) at $\epsilon=0.1/255$ via Semantic Inversion.</em></p>
    </div>

=== "Fig. 9(d) — Viral Pneumonia Class"

    <div align="center">
      <img src="../images/analisis_progresi_pgd_kelas_viral_pneumonia.png" alt="PGD Progression Viral Pneumonia" width="900">
      <p><em>Fig. 9(d) in JEEEMI: PGD progression on Viral Pneumonia class. Clean prediction: Viral Pneumonia (99.76%) shifts to Normal (100%) at $\epsilon=0.01 - 0.025/255$ and COVID-19 (100%) at $\epsilon \ge 0.05/255$.</em></p>
    </div>

---

## 2. FGSM Progression Analysis with Auto-Enhanced Noise (Fig. 10a – 10d in JEEEMI)

Progression grids utilizing **Auto-Scaled Noise Heatmaps (`cmap='magma'`)** to expose FGSM's structural edge artifacts alongside Grad-CAM focus displacement:

=== "Fig. 10(a) — Normal Class"

    <div align="center">
      <img src="../images/analisis_progresi_fgsm_(auto-enhanced-noise)_kelas_normal.png" alt="FGSM Progression Normal" width="900">
      <p><em>Fig. 10(a) in JEEEMI: FGSM progression on Normal class. Organ-edge disturbances trigger misclassification into Lung_Opacity (100%) at $\epsilon=0.01$ and COVID-19 (98.9% – 100%) at $\epsilon \ge 0.025$.</em></p>
    </div>

=== "Fig. 10(b) — Lung Opacity Class"

    <div align="center">
      <img src="../images/analisis_progresi_fgsm_(auto-enhanced-noise)_kelas_lung_opacity.png" alt="FGSM Progression Lung Opacity" width="900">
      <p><em>Fig. 10(b) in JEEEMI: FGSM progression on Lung Opacity class. Grad-CAM focus shifts sharply to the upper shoulder region, flipping predictions to COVID-19 (97.3% – 100%).</em></p>
    </div>

=== "Fig. 10(c) — COVID-19 Class"

    <div align="center">
      <img src="../images/analisis_progresi_fgsm_(auto-enhanced-noise)_kelas_covid.png" alt="FGSM Progression COVID-19" width="900">
      <p><em>Fig. 10(c) in JEEEMI: FGSM progression on COVID-19 class. Model exhibits relative resistance to single-step perturbations at higher bounds despite visible edge noise.</em></p>
    </div>

=== "Fig. 10(d) — Viral Pneumonia Class"

    <div align="center">
      <img src="../images/analisis_progresi_fgsm_(auto-enhanced-noise)_kelas_viral_pneumonia.png" alt="FGSM Progression Viral Pneumonia" width="900">
      <p><em>Fig. 10(d) in JEEEMI: FGSM progression on Viral Pneumonia class. Boundary disturbance patterns shift predictions to Normal (85.7%) and COVID-19 (77.8% – 100%).</em></p>
    </div>

---

## 3. PGD Progression with Auto-Enhanced Magma Noise (Part 10 Revised)

=== "PGD Auto-Enhanced — COVID-19"

    <div align="center">
      <img src="../images/analisis_progresi_pgd_(auto-enhanced-noise)_kelas_covid.png" alt="PGD Auto Enhanced COVID" width="900">
    </div>

=== "PGD Auto-Enhanced — Lung Opacity"

    <div align="center">
      <img src="../images/analisis_progresi_pgd_(auto-enhanced-noise)_kelas_lung_opacity.png" alt="PGD Auto Enhanced Lung Opacity" width="900">
    </div>

=== "PGD Auto-Enhanced — Normal"

    <div align="center">
      <img src="../images/analisis_progresi_pgd_(auto-enhanced-noise)_kelas_normal.png" alt="PGD Auto Enhanced Normal" width="900">
    </div>

=== "PGD Auto-Enhanced — Viral Pneumonia"

    <div align="center">
      <img src="../images/analisis_progresi_pgd_(auto-enhanced-noise)_kelas_viral_pneumonia.png" alt="PGD Auto Enhanced Viral Pneumonia" width="900">
    </div>

---

## 4. Sub-Pixel Micro-Progression Grids ($\epsilon = 0.1/255$ to $0.9/255$ — 10 Columns)

### A. Sub-Pixel FGSM ($0.1/255 - 0.9/255$, $50\times$ Noise Amplification)

=== "FGSM Sub-Pixel — COVID-19"

    <div align="center">
      <img src="../images/analisis_sub_pixel_fgsm_0.1-0.9_kelas_covid.png" alt="Sub Pixel FGSM COVID" width="950">
    </div>

=== "FGSM Sub-Pixel — Lung Opacity"

    <div align="center">
      <img src="../images/analisis_sub_pixel_fgsm_0.1-0.9_kelas_lung_opacity.png" alt="Sub Pixel FGSM Lung Opacity" width="950">
    </div>

=== "FGSM Sub-Pixel — Normal"

    <div align="center">
      <img src="../images/analisis_sub_pixel_fgsm_0.1-0.9_kelas_normal.png" alt="Sub Pixel FGSM Normal" width="950">
    </div>

=== "FGSM Sub-Pixel — Viral Pneumonia"

    <div align="center">
      <img src="../images/analisis_sub_pixel_fgsm_0.1-0.9_kelas_viral_pneumonia.png" alt="Sub Pixel FGSM Viral Pneumonia" width="950">
    </div>

### B. Sub-Pixel PGD ($0.1/255 - 0.9/255$, $50\times$ Noise Amplification)

=== "PGD Sub-Pixel — COVID-19"

    <div align="center">
      <img src="../images/analisis_sub_pixel_pgd_0.1-0.9_kelas_covid.png" alt="Sub Pixel PGD COVID" width="950">
    </div>

=== "PGD Sub-Pixel — Lung Opacity"

    <div align="center">
      <img src="../images/analisis_sub_pixel_pgd_0.1-0.9_kelas_lung_opacity.png" alt="Sub Pixel PGD Lung Opacity" width="950">
    </div>

=== "PGD Sub-Pixel — Normal"

    <div align="center">
      <img src="../images/analisis_sub_pixel_pgd_0.1-0.9_kelas_normal.png" alt="Sub Pixel PGD Normal" width="950">
    </div>

=== "PGD Sub-Pixel — Viral Pneumonia"

    <div align="center">
      <img src="../images/analisis_sub_pixel_pgd_0.1-0.9_kelas_viral_pneumonia.png" alt="Sub Pixel PGD Viral Pneumonia" width="950">
    </div>

---

## 5. Multi-Sample Class Blocks: 5 Samples per Class (6-Column & 8-Column Grids)

### A. 6-Column Grids (`Clean`, `CAM`, `FGSM`, `CAM`, `PGD`, `CAM`)

=== "COVID-19 (5 Samples)"

    <div align="center">
      <img src="../images/visualisasi_5_sampel_kelas_covid.png" alt="5 Samples COVID" width="900">
    </div>

=== "Lung Opacity (5 Samples)"

    <div align="center">
      <img src="../images/visualisasi_5_sampel_kelas_lung_opacity.png" alt="5 Samples Lung Opacity" width="900">
    </div>

=== "Normal (5 Samples)"

    <div align="center">
      <img src="../images/visualisasi_5_sampel_kelas_normal.png" alt="5 Samples Normal" width="900">
    </div>

=== "Viral Pneumonia (5 Samples)"

    <div align="center">
      <img src="../images/visualisasi_5_sampel_kelas_viral_pneumonia.png" alt="5 Samples Viral Pneumonia" width="900">
    </div>

### B. 8-Column Grids with Amplified Noise ($\times 15$)

=== "COVID-19 + Noise (8 Columns)"

    <div align="center">
      <img src="../images/visualisasi_5_sampel_kelas_covid_with_noise.png" alt="5 Samples COVID with Noise" width="950">
    </div>

=== "Lung Opacity + Noise (8 Columns)"

    <div align="center">
      <img src="../images/visualisasi_5_sampel_kelas_lung_opacity_with_noise.png" alt="5 Samples Lung Opacity with Noise" width="950">
    </div>

=== "Normal + Noise (8 Columns)"

    <div align="center">
      <img src="../images/visualisasi_5_sampel_kelas_normal_with_noise.png" alt="5 Samples Normal with Noise" width="950">
    </div>

=== "Viral Pneumonia + Noise (8 Columns)"

    <div align="center">
      <img src="../images/visualisasi_5_sampel_kelas_viral_pneumonia_with_noise.png" alt="5 Samples Viral Pneumonia with Noise" width="950">
    </div>

---

[:octicons-arrow-right-24: Continue to Error Analysis & Semantic Decoupling](error-analysis.md){ .md-button .md-button--primary }
