# Model Training Process & Convergence Dynamics

## Computational Environment

All training and adversarial evaluation experiments were executed using GPU hardware acceleration on Kaggle Notebooks:

- **Programming Language:** Python `3.11.13`
- **Deep Learning Framework:** PyTorch + Torchvision (`cuda` device)
- **Hardware Accelerator:** NVIDIA Tesla T4 / P100 GPU (CUDA Enabled)
- **Dataset Mount Path:** `/kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset`
- **Model Checkpoint File:** `best_densenet121_covid_4class.pth` (`28,452,566 bytes` / ~28.45 MB)

---

## Checkpoint Management: Load or Train Logic (Cell 2 & Cell 10)

To guarantee that both FGSM and PGD evaluations are executed against the exact same network parameters $\theta$, the experimental notebook implements an automated checkpoint verification routine for `best_densenet121_covid_4class.pth`:

1. **Checkpoint Present:** Loads the saved state dictionary (`model_ft.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))`) directly into the modified 4-class DenseNet-121 architecture.
2. **Checkpoint Absent:** Executes `train_model(...)` for up to 50 epochs with Early Stopping (`patience = 5`), saving a deep copy (`copy.deepcopy(model.state_dict())`) whenever validation accuracy improves.

---

## Training and Validation Convergence Dynamics (Fig. 5 in JEEEMI)

<div align="center">
  <img src="../images/training_validation_acc_loss_densenet121.png" alt="Training and Validation Accuracy and Loss" width="850">
  <p><em>Figure 1 (Fig. 5 in JEEEMI): Training and Validation (a) Accuracy and (b) Loss curves for DenseNet-121.</em></p>
</div>

### Convergence Analysis

- **Epochs 1 to 4:** Training accuracy increases rapidly from ~89.2% to >95.5%, while validation accuracy rises from ~89.3% in Epoch 1 to its peak (**>95.6%**) in Epoch 4 with minimum validation loss (~0.125).
- **Epochs 5 to 9 (Early Stopping Trigger):** Following Epoch 4, validation accuracy stabilizes between 94.5% and 95.5% without surpassing the Epoch 4 peak for 5 consecutive epochs (`patience = 5`). Training terminates automatically at the end of **Epoch 9**, and the best weights from Epoch 4 are restored into `best_densenet121_covid_4class.pth`.

---

## Execution Log: Dataset Preparation & Checkpoint Loading (`full_hasil.txt`)

```text
Memeriksa dataset di: /kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset

Kelas 'COVID': ditemukan 3616 gambar.
Kelas 'Lung_Opacity': ditemukan 6012 gambar.
Kelas 'Normal': ditemukan 10192 gambar.
Kelas 'Viral Pneumonia': ditemukan 1345 gambar.

Total semua gambar: 21165
Data Train: 16932 gambar
Data Val  : 2116 gambar
Data Test : 2117 gambar

DataLoader siap! Batch size: 32
Menggunakan perangkat: cuda
Konfigurasi: Epochs=50, Patience=5, Device=cuda
Membangun arsitektur model...
File model ditemukan: 'best_densenet121_covid_4class.pth'
Memuat model yang sudah ada...
Model berhasil dimuat! Melewati proses training.
```

---

[:octicons-arrow-right-24: Continue to Hyperparameter & Attack Configuration](hyperparameters.md){ .md-button .md-button--primary }
