# Dataset Statistics & Stratified Partitioning

## Global Class Distribution (Table 1 in JEEEMI)

Scanning the dataset directory in `Cell 4` of the notebook (`full_hasil.txt`) yields **21,165 CXR images** distributed across the four categories as follows:

| Class Category | Dataset Folder Name | Class Index (`class_to_idx`) | Number of Images | Percentage (%) | Dominant Radiological Biomarker Profile |
|:---------------|:--------------------|:----------------------------:|-----------------:|---------------:|:----------------------------------------|
| **Normal** | `Normal/images` | `2` | **10,192** | **48.15%** | Clear pulmonary parenchyma, normal vascularity |
| **Lung Opacity** | `Lung_Opacity/images` | `1` | **6,012** | **28.41%** | Non-specific diffuse opacity, low contrast |
| **COVID-19** | `COVID/images` | `0` | **3,616** | **17.08%** | Subtle bilateral ground-glass opacity (GGO), high spatial frequency |
| **Viral Pneumonia** | `Viral Pneumonia/images` | `3` | **1,345** | **6.36%** | Macro-structural consolidation & infiltrates, low spatial frequency |
| **Total** | — | — | **21,165** | **100.00%** | — |

---

## Stratified Dataset Partitioning (`80 : 10 : 10`)

To preserve exact class proportions across all subsets despite class imbalance (48.15% *Normal* vs. 6.36% *Viral Pneumonia*), the dataset is partitioned using a two-stage stratified split (`stratify=all_labels`, `random_state=42`) via `scikit-learn`:

1. **Stage 1:** Splits the full dataset into **80% Training Set** (`16,932` images) and **20% Temporary Set** (`4,233` images).
2. **Stage 2:** Splits the Temporary Set equally (`test_size=0.5`, `stratify=y_temp`, `random_state=42`) into **10% Validation Set** (`2,116` images) and **10% Testing Set** (`2,117` images).

### Subset Distribution per Class

| Class Category | Training Set (80%) | Validation Set (10%) | Testing Set (10% — Evaluation Support) | Total Images |
|:---------------|-------------------:|---------------------:|---------------------------------------:|-------------:|
| **Normal** | 8,154 | 1,019 | **1,019** | 10,192 |
| **Lung Opacity** | 4,810 | 601 | **602** | 6,012 |
| **COVID-19** | 2,893 | 362 | **362** | 3,616 |
| **Viral Pneumonia** | 1,075 | 134 | **134** | 1,345 |
| **Total Samples** | **16,932** | **2,116** | **2,117** | **21,165** |

---

## Implementation Code (Cell 5 & Cell 8)

```python
DATA_ROOT = '/kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset'
CLASSES = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
BATCH_SIZE = 32
IMAGE_SIZE = 224

all_image_paths = []
all_labels = []
class_to_idx = {cls_name: i for i, cls_name in enumerate(CLASSES)}

for category in CLASSES:
    folder_path = os.path.join(DATA_ROOT, category, 'images')
    files = glob.glob(os.path.join(folder_path, '*.png'))
    for f in files:
        all_image_paths.append(f)
        all_labels.append(class_to_idx[category])

# Stratified Split: 80% Train, 10% Val, 10% Test
X_train, X_temp, y_train, y_temp = train_test_split(
    all_image_paths, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)
```

---

[:octicons-arrow-right-24: Continue to Data Preprocessing Pipeline](preprocessing.md){ .md-button .md-button--primary }
