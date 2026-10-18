# Hyperparameter & Attack Configuration

This page documents all configuration parameters used in the experimental notebook (`covid19_xray_adversarial_robustness.ipynb`) and reported in the JEEEMI article.

---

## 1. Dataset & Preprocessing Hyperparameters

| Parameter | Code Variable | Value | Description |
|:----------|:--------------|:------|:------------|
| **Spatial Resolution** | `IMAGE_SIZE` | `224` ($224 \times 224$) | Standard input dimension for DenseNet-121 |
| **Input Channels** | — | `3` (`RGB`) | Converted via `.convert("RGB")` |
| **Batch Size** | `BATCH_SIZE` | `32` | Applied across `train`, `val`, and `test` DataLoaders |
| **DataLoader Workers** | `num_workers` | `2` | Parallel data loading subprocesses |
| **Stratified Split Seed** | `random_state` | `42` | Ensures reproducible 80:10:10 partitioning |
| **ImageNet Mean ($\mu$)** | `norm_mean` | `[0.485, 0.456, 0.406]` | Channel-wise mean for Z-score standardization |
| **ImageNet Std ($\sigma$)** | `norm_std` | `[0.229, 0.224, 0.225]` | Channel-wise std for normalization and $\epsilon_{\text{norm}} = \epsilon / \sigma$ scaling |
| **Training Augmentation** | `data_transforms['train']` | `RandomHorizontalFlip()`, `RandomRotation(10)` | Applied exclusively to the Training subset (`16,932` images) |

---

## 2. DenseNet-121 Architecture & Training Hyperparameters

| Parameter | Code Variable | Value | Description |
|:----------|:--------------|:------|:------------|
| **Pretrained Backbone** | `models.densenet121` | `weights='DEFAULT'` | ImageNet-1K pretrained weights (`densenet121-a639ec97.pth`) |
| **Feature Dimension** | `num_ftrs` | `1024` | Output channels of `denseblock4` after Global Average Pooling |
| **Classifier Head** | `model_ft.classifier` | `nn.Linear(1024, 4)` | Linear projection to 4 diagnostic class logits |
| **Loss Criterion** | `criterion` | `nn.CrossEntropyLoss()` | Categorical Cross-Entropy Loss (Equation 3) |
| **Optimizer** | `optimizer_ft` | `optim.Adam` | Adaptive moment estimation across all model parameters |
| **Learning Rate ($\eta$)** | `LEARNING_RATE` | `1e-4` ($10^{-4}$) | Initial fine-tuning learning rate |
| **Maximum Epochs** | `EPOCHS` | `50` | Upper bound on training epochs |
| **Early Stopping** | `PATIENCE` | `5` | Stops training if validation accuracy does not improve for 5 epochs |

---

## 3. Adversarial Attack & Visualization Configurations

| Notebook Section / Experiment | Attack Method | Epsilon ($\epsilon$) | Step Size ($\alpha$) | Iterations ($N$) | Noise Visualization Scaling |
|:------------------------------|:-------------:|:---------------------|:--------------------:|:----------------:|:----------------------------|
| **Parts 4 & 5 (Initial Evaluation)** | FGSM & PGD | $0.1/255$ (`0.000392`) | $2/255$ (Part 4)<br>$\epsilon/4$ (Micro Eval) | $1$ (FGSM)<br>$10$ (PGD) | — |
| **Parts 6 & 7 (Grad-CAM 1 & 5 Samples)** | FGSM & PGD | $0.1/255$ | $2/255$ | $10$ | Grad-CAM overlay on `model.features[-1]` |
| **Part 8 (8-Column Grad-CAM + Noise)** | FGSM & PGD | $8/255$ | $2/255$ | $10$ | $15\times$ amplification (`compute_noise_diff`) |
| **Parts 9a / 13 & 14 (FGSM Sub-Pixel)** | FGSM | $0.1/255$ to $0.9/255$ (9 steps) | — | $1$ | $50\times$ amplification (`compute_noise_amp`) |
| **Part 9b (PGD Custom Sensitivity)** | PGD | $\{0.01, \ 0.025, \ 0.05, \ 0.1, \ 0.3\}/255$ | $\epsilon / 4$ | $10$ | — |
| **Part 10 (FGSM & PGD Progression)** | FGSM & PGD | $\{0.01, \ 0.025, \ 0.05, \ 0.1\}/255$ | $\epsilon / 4$ | $10$ | $20\times$ amplification & Auto-Scaled Magma Heatmap |
| **Parts 11 & 12 (PGD Sub-Pixel 0.1–0.9)** | PGD | $0.1/255$ to $0.9/255$ (9 steps) | $\epsilon / 4$ | $10$ | $50\times$ amplification |
| **Dedicated Micro-Attack Evaluation** | FGSM & PGD | $0.1/255$ (`0.000392`) | $\epsilon / 4$ | $10$ | Logs Accuracy, ASR, and Average Confidence |

---

[:octicons-arrow-right-24: Continue to Experimental Results Summary](results.md){ .md-button .md-button--primary }
