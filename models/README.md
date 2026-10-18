# Trained Model Checkpoint (`DenseNet-121`)

This directory is designated for storing the trained model weights (`best_densenet121_covid_4class.pth`) obtained from fine-tuning on the **COVID-19 Radiography Dataset**.

## Model Specifications

| Parameter | Configuration / Value |
|:----------|:----------------------|
| **Checkpoint Filename** | `best_densenet121_covid_4class.pth` |
| **File Size** | `28,452,566 bytes` (~28.45 MB) |
| **Base Architecture** | `torchvision.models.densenet121(weights='DEFAULT')` |
| **Classification Head** | `nn.Linear(in_features=1024, out_features=4)` |
| **Class Index Mapping (`CLASSES`)** | `['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']` |
| **Input Resolution** | `3 x 224 x 224` (RGB, ImageNet Z-score Normalized) |
| **Optimizer & Loss** | `Adam (lr = 1e-4)`, `CrossEntropyLoss` |
| **Training Protocol** | Max 50 Epochs, `EarlyStopping (patience = 5)` |
| **Clean Test Set Accuracy** | **95.42%** ($2,020 / 2,117$ test samples correctly classified) |

## Loading the Checkpoint in PyTorch

```python
import torch
import torch.nn as nn
from torchvision import models

CLASSES = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.densenet121(weights=None)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, len(CLASSES))

state_dict = torch.load("best_densenet121_covid_4class.pth", map_location=DEVICE)
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()
```
