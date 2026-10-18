# Installation & Environment Setup

## System Prerequisites

- **Python:** Version `3.10` – `3.15` (Kaggle experiments executed on Python `3.11.13`)
- **Hardware Accelerator (Recommended for Experiments):** CUDA-compatible NVIDIA GPU (e.g., Tesla T4 / P100 on Kaggle or Google Colab) for DenseNet-121 training and 10-step PGD generation across 2,117 test images.
- **System Memory:** Minimum 8 GB RAM (16 GB recommended).

---

## 1. Clone the Repository

```bash
git clone https://github.com/hisyam99/covid19-xray-adversarial-attack.git
cd covid19-xray-adversarial-attack
```

---

## 2. Install Python Dependencies

### A. Core Deep Learning & Research Environment (`requirements.txt`)

```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Contents of root `requirements.txt`:

```txt
# Core Deep Learning & Computer Vision
torch>=2.5.0
torchvision>=0.20.0
numpy>=2.0.0
pandas>=2.2.0
scikit-learn>=1.5.0
pillow>=11.0.0
tqdm>=4.66.0
matplotlib>=3.9.0
seaborn>=0.13.0

# Explainable AI (Grad-CAM)
grad-cam>=1.5.0

# Interactive Notebook Environment
jupyter>=1.1.0
ipykernel>=6.29.0
```

### B. Documentation Website Dependencies (`docs/requirements.txt`)

To build or preview the MkDocs documentation website:

```bash
pip install -r docs/requirements.txt
```

---

## 3. Running the Experiments on Kaggle Notebooks

1. Create a new Notebook on [Kaggle](https://www.kaggle.com/) and enable **Accelerator: GPU T4 x2** or **GPU P100**.
2. Attach the **[COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)** (`tawsifurrahman/covid19-radiography-database`) input dataset at:
   ```text
   /kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset
   ```
3. Upload `notebooks/code3v3-hisyam_v4.ipynb` (and optionally place `best_densenet121_covid_4class.pth` in `/kaggle/working/`).
4. Execute all cells sequentially. The `grad-cam` package is installed automatically in `Cell 19` (`!pip install grad-cam`).

---

[:octicons-arrow-right-24: Continue to Jupyter Notebook Code Walkthrough](notebook.md){ .md-button .md-button--primary }
