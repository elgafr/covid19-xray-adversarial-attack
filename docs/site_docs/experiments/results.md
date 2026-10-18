# Experimental Results Summary

This page provides a consolidated overview of all experimental outputs recorded in `results/full_hasil.txt` and reported in the JEEEMI article.

---

## 1. Evaluation Scenarios at $\epsilon = 0.1/255$

Within `code3v3-hisyam_v4.ipynb`, two PGD parameterizations at $\epsilon = 0.1/255$ are evaluated:

1. **Part 4 (Initial PGD Evaluation with Fixed Step $\alpha = 2/255$):** Yields an adversarial accuracy of **53.90%** (ASR **46.10%**).
2. **Parts 9b, 11 & Dedicated Micro-Attack Evaluation (Proportional Step $\alpha = \epsilon/4, N = 10$ — Reported in JEEEMI Tables 2, 3, and 4):** Scaling the step size proportionally to the sub-pixel $\epsilon$-ball ($N \cdot \alpha = 2.5\epsilon$) prevents gradient overshoot and yields a much more severe accuracy drop to **25.32%** (ASR **74.68%**).

### Primary Execution Summary Table ($\epsilon = 0.1/255 \approx 0.000392$)

| Method | Epsilon ($\epsilon$) | Step Size ($\alpha$) | Model Accuracy (%) | Attack Success Rate (ASR %) | Avg. Confidence (%) |
|:-------|:--------------------:|:--------------------:|:------------------:|:---------------------------:|:-------------------:|
| **Clean (No Attack)** | `0` | — | **95.42%** (`95.4180%`) | — | **96.05%** (`96.0468%`) |
| **FGSM (Single-Step)** | `0.1/255` (`0.000392`) | — | **59.28%** (`59.2820%`) | **40.72%** (`40.7180%`) | **82.16%** (`82.1624%`) |
| **PGD (Part 4 — Coarse Step)** | `0.1/255` (`0.000392`) | `2/255` | **53.90%** (`53.8970%`) | **46.10%** (`46.1030%`) | — |
| **PGD (Primary JEEEMI — Proportional Step)** | `0.1/255` (`0.000392`) | $\epsilon / 4$ | **25.32%** (`25.3188%`) | **74.68%** (`74.6812%`) | **85.42%** (`85.4159%`) |

---

## 2. Sub-Pixel Sensitivity Summary

=== "PGD Custom Low-Range Sensitivity (0.01/255 – 0.3/255)"

    | Epsilon Label | Decimal Value ($\epsilon$) | Model Accuracy (%) | Attack Success Rate (ASR %) | Robustness Regime |
    |:-------------:|:--------------------------:|:------------------:|:---------------------------:|:------------------|
    | **0 (Clean)** | `0.000000` | **95.42%** | — | Optimal Baseline |
    | **0.01/255** | `0.000039` | **92.82%** (`92.8200%`) | **7.18%** (`7.1800%`) | Initial Degradation ($\downarrow 2.60\%$) |
    | **0.025/255** | `0.000098` | **86.30%** (`86.3014%`) | **13.70%** (`13.6986%`) | Early Instability ($\downarrow 9.12\%$) |
    | **0.05/255** | `0.000196` | **66.65%** (`66.6509%`) | **33.35%** (`33.3491%`) | Moderate-High Degradation ($\downarrow 28.77\%$) |
    | **0.1/255** | `0.000392` | **25.51% / 25.32%** | **74.49% / 74.68%** | **Critical Degradation Threshold** ($\approx$ Random Guess) |
    | **0.3/255** | `0.001176` | **0.05%** (`0.0472%`) | **99.95%** (`99.9528%`) | **Total Predictive Collapse** |

=== "Sub-Pixel FGSM vs. PGD Comparison (0.1/255 – 0.9/255)"

    | Epsilon Label | Decimal Value ($\epsilon$) | FGSM Accuracy (%) | FGSM ASR (%) | PGD Accuracy (%) | PGD ASR (%) | Performance Gap (FGSM − PGD) |
    |:-------------:|:--------------------------:|:-----------------:|:------------:|:----------------:|:-----------:|:----------------------------:|
    | **0.1/255** | `0.000392` | **59.28%** | 40.72% | **25.60% / 25.32%** | 74.40% / 74.68% | $+33.68\% \text{ to } +33.96\%$ |
    | **0.2/255** | `0.000784` | **32.88%** | 67.12% | **1.51%** | 98.49% | $+31.37\%$ |
    | **0.3/255** | `0.001176` | **21.73%** | 78.27% | **0.14% / 0.05%** | 99.86% / 99.95% | $+21.59\% \text{ to } +21.68\%$ |
    | **0.4/255** | `0.001569` | **15.16%** | 84.84% | **0.05%** | 99.95% | $+15.11\%$ |
    | **0.5/255** | `0.001961` | **12.14%** | 87.86% | **0.05%** | 99.95% | $+12.09\%$ |
    | **0.6/255** | `0.002353` | **10.39%** | 89.61% | **0.00%** | 100.00% | $+10.39\%$ |
    | **0.7/255** | `0.002745` | **9.31%** | 90.69% | **0.00%** | 100.00% | $+9.31\%$ |
    | **0.8/255** | `0.003137` | **9.26%** | 90.74% | **0.00%** | 100.00% | $+9.26\%$ |
    | **0.9/255** | `0.003529` | **9.02%** | 90.98% | **0.00%** | 100.00% | $+9.02\%$ |

---

[:octicons-arrow-right-24: Continue to Detailed Results & Discussion](../results/overview.md){ .md-button .md-button--primary }
