# Mathematical Formulation of Data Preprocessing

This section details the mathematical formulation of the input standardization pipeline presented in Section II.A of our JEEEMI paper.

---

## 1. Input Vector Standardization

Let the complete dataset be denoted as $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$ with $N = 21,165$ chest radiographs and diagnostic class labels $y_i \in \{0, 1, 2, 3\}$. Each raw radiograph of resolution $299 \times 299$ undergoes two deterministic spatial transformations:

1. **Grayscale-to-RGB Conversion:** $x_{\text{raw}} \in \mathbb{R}^{H_0 \times W_0 \times 1} \mapsto x_{\text{rgb}} \in \mathbb{R}^{H_0 \times W_0 \times 3}$ by replicating the luminance channel across $c \in \{R, G, B\}$.
2. **Spatial Resizing:** Resampling to $H = 224, W = 224$ pixels and scaling to $x \in [0, 1]^{3 \times 224 \times 224}$.

---

## 2. Z-Score Standardization — Equation (1)

Let $x^{(c)}$ denote the resized pixel intensity in $[0, 1]$ for color channel $c \in \{R, G, B\}$. The normalized pixel value $x_{\text{norm}}^{(c)}$ is computed via Z-score standardization as defined in **Equation (1)** of the JEEEMI article:

$$
x_{\text{norm}}^{(c)} = \frac{x^{(c)} - \mu_c}{\sigma_c} \tag{1}
$$

where $\mu_c$ and $\sigma_c$ represent the ImageNet channel-wise mean and standard deviation:

- $\mu_c = [0.485, \ 0.456, \ 0.406]$
- $\sigma_c = [0.229, \ 0.224, \ 0.225]$

### Inverse Transformation (Denormalization) for Visualization

To render perturbed images, compute pixel-space noise differences, and overlay **Grad-CAM** heatmaps, the normalized tensor $x_{\text{norm}}^{(c)}$ is mapped back to $[0, 1]$ RGB space via:

$$
x^{(c)} = \text{clip}\left(x_{\text{norm}}^{(c)} \cdot \sigma_c + \mu_c, \ 0, \ 1\right)
$$

```python
def denormalize_image(tensor):
    """Maps a normalized tensor (C, H, W) back to a numpy RGB image (H, W, C) in [0, 1]."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    tensor = tensor.cpu().detach()
    img = tensor.permute(1, 2, 0).numpy()
    img = img * std + mean
    return np.clip(img, 0, 1)
```

---

[:octicons-arrow-right-24: Continue to DenseNet-121 Architecture](densenet.md){ .md-button .md-button--primary }
