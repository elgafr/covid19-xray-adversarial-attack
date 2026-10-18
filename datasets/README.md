# COVID-19 Radiography Dataset

This research utilizes the **COVID-19 Radiography Database**, developed by researchers from **Qatar University** (Doha, Qatar) and the **University of Dhaka** (Bangladesh) in collaboration with medical partners from Pakistan and Malaysia.

## Official Dataset Repository

- **Kaggle URL:** [https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

## Class Distribution & Stratified Partitioning

A total of **21,165 Chest X-Ray (CXR)** images in `.png` format are extracted from the `images/` subdirectory of each diagnostic category (excluding the `masks/` directory) and partitioned using a two-stage stratified split (`random_state=42`) into **80% Training : 10% Validation : 10% Testing**:

| Diagnostic Class | Label Index | Total Images | Percentage | Train (80%) | Val (10%) | Test (10%) |
|:-----------------|:-----------:|-------------:|-----------:|------------:|----------:|-----------:|
| **COVID** | `0` | 3,616 | 17.08% | 2,893 | 362 | 362 |
| **Lung_Opacity** | `1` | 6,012 | 28.41% | 4,810 | 601 | 602 |
| **Normal** | `2` | 10,192 | 48.15% | 8,154 | 1,019 | 1,019 |
| **Viral Pneumonia** | `3` | 1,345 | 6.36% | 1,075 | 134 | 134 |
| **Total** | — | **21,165** | **100.00%** | **16,932** | **2,116** | **2,117** |

## Expected Directory Layout

```text
COVID-19_Radiography_Dataset/
├── COVID/
│   └── images/*.png
├── Lung_Opacity/
│   └── images/*.png
├── Normal/
│   └── images/*.png
└── Viral Pneumonia/
    └── images/*.png
```

## Dataset Citation

```bibtex
@article{chowdhury2020can,
  title   = {Can AI Help in Screening Viral and COVID-19 Pneumonia?},
  author  = {Chowdhury, Muhammad E. H. and Rahman, Tawsifur and Khandakar, Amith and Mazhar, Rashid and Kadir, Muhammad Abdul and Mahbub, Zaid Bin and Islam, Khandakar Reajul and Khan, Muhammad Salman and Iqbal, Atif and Al Emadi, Nasser and Reaz, Mamun Bin Ibne and Islam, Mohammad Tariqul},
  journal = {IEEE Access},
  volume  = {8},
  pages   = {132665--132676},
  year    = {2020},
  doi     = {10.1109/ACCESS.2020.3010287}
}
```
