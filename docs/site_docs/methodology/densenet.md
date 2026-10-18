# DenseNet-121 Architecture & Feature Reuse

## Selection of DenseNet-121

We selected **DenseNet-121** (*Densely Connected Convolutional Networks*, Huang et al. [30]) as the primary feature extractor due to its dense connectivity pattern, which mitigates the vanishing gradient problem and encourages aggressive feature reuse across layers.

Unlike traditional CNN architectures or ResNets that combine feature maps via element-wise summation, DenseNet **concatenates the output feature maps of all preceding layers** within each Dense Block.

---

## Mathematical Formulation of Feature Propagation — Equation (2)

Formally, let $x_\ell$ denote the output of the $\ell$-th layer within a Dense Block. Feature propagation is defined in **Equation (2)** of the JEEEMI article as:

$$
x_\ell = H_\ell\left([x_0, x_1, \dots, x_{\ell-1}]\right) \tag{2}
$$

where:

- $[x_0, x_1, \dots, x_{\ell-1}]$ denotes the channel-wise concatenation of feature maps produced in layers $0, 1, \dots, \ell-1$.
- $H_\ell(\cdot)$ represents a composite non-linear transformation comprising **Batch Normalization (BN)**, **Rectified Linear Unit (ReLU)**, a $1 \times 1$ bottleneck convolution, and a $3 \times 3$ spatial convolution.

With a growth rate of $k = 32$, each layer adds 32 new feature maps to the collective knowledge volume of the block.

---

## Layer-by-Layer Architecture & 4-Class Classifier Modification

![DenseNet-121 Architecture Diagram](../images/cropped_architecture_diagram.png){: .center width="850" }

*Figure: Figure 1: Detailed DenseNet-121 Architecture (Stage 2 in Fig. 1 of JEEEMI) showing the 4 Dense Blocks, 3 Transition Layers, Global Average Pooling, and 4-Class Linear Classifier.*

| Architectural Stage | Layer Configuration | Output Spatial & Channel Dimensions |
|:--------------------|:--------------------|:-----------------------------------:|
| **Input Layer** | Normalized RGB CXR Image | $224 \times 224 \times 3$ |
| **Initial Convolution** | $7 \times 7$ Conv (stride 2), BN, ReLU $\rightarrow$ $3 \times 3$ MaxPool (stride 2) | $56 \times 56 \times 64$ |
| **Dense Block 1** | $L = 6$ dense layers ($\text{growth rate } k=32$) | $56 \times 56 \times 256$ |
| **Transition Layer 1** | BN, ReLU, $1 \times 1$ Conv ($\theta = 0.5$) + $2 \times 2$ AvgPool (stride 2) | $28 \times 28 \times 128$ |
| **Dense Block 2** | $L = 12$ dense layers ($\text{growth rate } k=32$) | $28 \times 28 \times 512$ |
| **Transition Layer 2** | BN, ReLU, $1 \times 1$ Conv ($\theta = 0.5$) + $2 \times 2$ AvgPool (stride 2) | $14 \times 14 \times 256$ |
| **Dense Block 3** | $L = 24$ dense layers ($\text{growth rate } k=32$) | $14 \times 14 \times 1024$ |
| **Transition Layer 3** | BN, ReLU, $1 \times 1$ Conv ($\theta = 0.5$) + $2 \times 2$ AvgPool (stride 2) | $7 \times 7 \times 512$ |
| **Dense Block 4** | $L = 16$ dense layers ($\text{growth rate } k=32$) — Grad-CAM Target (`features[-1]`) | $7 \times 7 \times 1024$ |
| **Global Average Pooling** | Spatial pooling $7 \times 7 \rightarrow 1 \times 1$ | $1 \times 1 \times 1024$ |
| **Classification Layer** | `nn.Linear(in_features=1024, out_features=4)` + Softmax | $4$ (`COVID`, `Lung_Opacity`, `Normal`, `Viral Pneumonia`) |

---

## PyTorch Model Initialization (Cell 10)

```python
from torchvision import models
import torch.nn as nn

# Initialize DenseNet-121 with pretrained ImageNet weights
model_ft = models.densenet121(weights='DEFAULT')

# Replace the final linear classifier for 4 diagnostic classes
num_ftrs = model_ft.classifier.in_features
CLASSES = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
model_ft.classifier = nn.Linear(num_ftrs, len(CLASSES))

model_ft = model_ft.to(DEVICE)
```

---

[:octicons-arrow-right-24: Continue to Training Strategy & Loss Function](training.md){ .md-button .md-button--primary }
