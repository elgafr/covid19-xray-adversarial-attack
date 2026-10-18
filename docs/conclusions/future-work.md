# Future Research Directions

Building upon the findings and operational scope of this study, five directions are prioritized for future work:

---

## 1. Evaluation of Vision Transformers & Hybrid Attention Architectures

Comparing the adversarial resilience of pure convolutional backbones (**DenseNet-121**) against self-attention architectures such as **Vision Transformers (ViT)**, **Swin Transformers**, and **Attention-Augmented DenseNets** (Hassan et al. [26]) to assess whether global self-attention mitigates localized *Semantic Decoupling*.

---

## 2. Geometric & Spectral Regularization (*Lipschitz / Jacobian Constraints*)

Integrating Lipschitz continuity constraints or Jacobian matrix regularization ($\|\nabla_x f_\theta(x)\|_F$) during training to bound logit sensitivity with respect to sub-pixel input perturbations.

---

## 3. Expanded Threat Models (*AutoAttack, Carlini-Wagner & Black-Box Transferability*)

Extending security evaluations beyond $L_\infty$ FGSM and PGD to encompass:

- **Carlini & Wagner ($L_2$ / $L_\infty$ CW)** and **DeepFool** minimum-norm optimization attacks.
- **AutoAttack (Croce & Hein [40]):** Parameter-free ensemble evaluation (*APGD-CE*, *APGD-DLR*, *FAB*, *Square Attack*).
- **Black-Box Surrogate Transferability & Physical Artifacts:** Evaluating cross-architecture attack transferability and physical imaging artifacts.

---

## 4. Multi-Modal & Federated Medical Imaging

Validating whether *Semantic Decoupling* and *Semantic Inversion* generalize across volumetric **3D Chest CT**, **MRI**, and **Federated Learning** environments across distributed hospital nodes (Darzi et al. [16]).

---

## 5. Clinical Validation with Board-Certified Radiologists

Conducting interdisciplinary reader studies with board-certified radiologists to evaluate Grad-CAM localization shifts against clinical diagnostic protocols.

---

[:octicons-arrow-right-24: View Complete Bibliography (53 References)](../references.md){ .md-button .md-button--primary }
