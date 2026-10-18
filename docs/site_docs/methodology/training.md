# Training Strategy & Optimization Protocol

## Categorical Cross-Entropy Loss — Equation (3)

The **DenseNet-121** model is initialized with pretrained weights from ImageNet to expedite convergence and fine-tuned end-to-end across all network parameters $\theta$ using the categorical Cross-Entropy loss function defined in **Equation (3)** of the JEEEMI article:

$$
J(\theta, x, y) = -\sum_{i=1}^{K} y_i \log(\hat{y}_i) \tag{3}
$$

*(Note: expressed in proportional form $\sum_{i=1}^{K} y_i \log(\hat{y}_i)$ in Eq. 3 of the paper)*, where:

- $K = 4$ represents the number of diagnostic classes (*COVID-19*, *Lung Opacity*, *Normal*, *Viral Pneumonia*).
- $y_i \in \{0, 1\}$ is the binary ground-truth indicator for class $i$.
- $\hat{y}_i$ is the predicted softmax probability for class $i$.

!!! info "Strict White-Box Threat Model Alignment"
    Crucially, in our threat model, both **FGSM** and **PGD** adversarial attacks compute exact input gradients $\nabla_x J(\theta, x, y)$ from this identical loss function $J(\theta, x, y)$, representing a strict white-box attack scenario rather than relying on a surrogate model approximation.

---

## Training Hyperparameters

| Hyperparameter | Configuration | Methodological Rationale |
|:---------------|:--------------|:-------------------------|
| **Execution Device** | PyTorch CUDA GPU | Hardware acceleration for dense block backpropagation |
| **Optimizer** | `torch.optim.Adam` | Adaptive moment estimation across all trainable parameters $\theta$ |
| **Initial Learning Rate ($\eta$)** | $1 \times 10^{-4}$ (`1e-4`) | Preserves pretrained ImageNet feature representations during fine-tuning |
| **Batch Size** | `32` | Balances gradient variance and GPU memory footprint |
| **Maximum Epochs** | `50` | Upper bound on training iterations |
| **Early Stopping Patience** | `5` Epochs | Terminates training when validation accuracy fails to improve for 5 consecutive epochs |
| **Model Checkpoint** | `best_densenet121_covid_4class.pth` | Stores `copy.deepcopy(model.state_dict())` at peak validation accuracy |

---

## Training and Validation Convergence Curves (Fig. 5 in JEEEMI)

<div align="center">
  <img src="../images/training_validation_acc_loss_densenet121.png" alt="Training and Validation Accuracy and Loss" width="850">
  <p><em>Figure 1 (Fig. 5 in JEEEMI): Training and Validation (a) Accuracy and (b) Loss curves across epochs.</em></p>
</div>

As shown in **Figure 1**:

- Training and validation accuracy converge rapidly from ~89.2% in Epoch 1 to over **95.5%** by Epoch 4, while validation loss drops smoothly with minimal oscillation.
- The **Early Stopping (`patience=5`)** mechanism terminates training after **Epoch 9** once validation accuracy peaks at Epoch 4, preventing overfitting and restoring the optimal weights saved in `best_densenet121_covid_4class.pth`.

---

[:octicons-arrow-right-24: Continue to Adversarial Attack Formulations (FGSM & PGD)](adversarial.md){ .md-button .md-button--primary }
