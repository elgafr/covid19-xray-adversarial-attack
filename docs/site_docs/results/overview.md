# Model Performance on Original (Clean) Data

This section presents the evaluation of the **DenseNet-121** model on the unperturbed *Original Clean Test Set* ($N = 2,117$ images), corresponding to **Section III.A of the JEEEMI article**.

---

## Baseline Performance Summary

On the clean test set, the **DenseNet-121** architecture achieved an overall accuracy of **95.42%** ($2,020$ correct predictions out of $2,117$ test instances), with an exact binomial **95% Confidence Interval (Clopper-Pearson) of $94.53\% - 96.31\%$** and an **Average Prediction Confidence of 96.05%**.

### Per-Class Performance Metrics on Clean Data (Table 2 in JEEEMI)

| Diagnostic Category | Precision | Recall (Sensitivity) | F1-Score | Support ($N$) | Baseline Diagnostic Characteristics |
|:--------------------|:---------:|:--------------------:|:--------:|:-------------:|:------------------------------------|
| **COVID-19** | **0.9914** | 0.9586 | **0.9747** | 362 | Highest precision and F1-score; accurately captures bilateral GGO patterns |
| **Lung Opacity** | 0.9566 | 0.9153 | 0.9355 | 602 | Comparatively lower recall due to subtle overlap with *Normal* vascularity |
| **Normal** | 0.9368 | **0.9745** | 0.9553 | 1,019 | Highest recall (97.45%); highly reliable screening of healthy lungs |
| **Viral Pneumonia** | 0.9847 | 0.9627 | 0.9736 | 134 | Strong F1-score (>0.97) despite being the minority class |
| **Overall Accuracy** | — | — | **95.42%** | **2,117** | **$2,020 / 2,117$ test instances correctly classified** |
| **Macro Average** | **0.9674** | **0.9528** | **0.9598** | 2,117 | Balanced performance across all four diagnostic categories |
| **Weighted Average** | **0.9548** | **0.9542** | **0.9541** | 2,117 | Consistent with global accuracy |

---

## Training Curves & Baseline Confusion Matrix

=== "Confusion Matrix — Original Test Set (Fig. 6)"

    ![Confusion Matrix Original Test Set](../images/cm_original_test_set.png){: .center width="550" }

*Figure: Figure 1 (Fig. 6 in JEEEMI): Confusion Matrix on the Original Clean Test Set ($N=2,117$, Accuracy = 95.42%).*

=== "Training & Validation Curves (Fig. 5)"

    ![Training and Validation Accuracy and Loss](../images/training_validation_acc_loss_densenet121.png){: .center width="800" }

*Figure: Figure 2 (Fig. 5 in JEEEMI): Training and Validation (a) Accuracy and (b) Loss convergence profiles.*

---

## Detailed Breakdown of the Clean Confusion Matrix ($N = 2,117$)

From **Figure 1 (Fig. 6)**:

| True Label \ Predicted Label | COVID-19 | Lung Opacity | Normal | Viral Pneumonia | Total Support | Correct Predictions (Recall) |
|:-----------------------------|:--------:|:------------:|:------:|:---------------:|:-------------:|:----------------------------:|
| **COVID-19** | **347** | 1 | 13 | 1 | 362 | $347 / 362$ (**95.86%**) |
| **Lung Opacity** | 2 | **551** | 49 | 0 | 602 | $551 / 602$ (**91.53%**) |
| **Normal** | 1 | 24 | **993** | 1 | 1,019 | $993 / 1,019$ (**97.45%**) |
| **Viral Pneumonia** | 0 | 0 | 5 | **129** | 134 | $129 / 134$ (**96.27%**) |

- **Diagonal Concentration:** A total of **2,020 instances** lie on the principal diagonal, with only **97 instances (4.58%)** misclassified across the entire test set.
- **Natural Clinical Overlap:** The primary source of baseline error occurs between *Lung Opacity* and *Normal* (49 *Lung Opacity* instances predicted as *Normal*, and 24 *Normal* instances predicted as *Lung Opacity*), reflecting radiological similarity between mild non-specific opacities and normal anatomical variations.

---

[:octicons-arrow-right-24: Continue to Clean vs. FGSM vs. PGD Comparison](comparison.md){ .md-button .md-button--primary }
