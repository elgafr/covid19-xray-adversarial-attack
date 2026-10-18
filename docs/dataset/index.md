# Dataset Documentation

Comprehensive documentation of the **COVID-19 Radiography Dataset** utilized as the primary data source for training the **DenseNet-121** model and evaluating **FGSM** and **PGD** adversarial attacks.

<div class="grid cards" markdown>

-   :material-database-eye: **[Introduction](introduction.md)**

    ---
    
    Dataset origin (Qatar University & University of Dhaka), pathological characteristics of the four diagnostic categories, and representative sample visualizations.

-   :material-chart-pie: **[Dataset Statistics](statistics.md)**

    ---
    
    Distribution of the 21,165 CXR images globally and across stratified partitions (80% Train, 10% Validation, 10% Test) along with radiometric statistics ($\mu, \sigma$).

-   :material-image-filter-center-focus: **[Data Preprocessing](preprocessing.md)**

    ---
    
    Input vector standardization pipeline: grayscale-to-RGB conversion, $224 \times 224$ spatial resizing, ImageNet Z-score normalization (**Equation 1**), and training augmentation.

-   :material-format-quote-close: **[Dataset Citation & Ethics](citation.md)**

    ---
    
    Official BibTeX citation (Chowdhury et al., *IEEE Access* 2020), Kaggle repository link, and ethical compliance statement.

</div>

---

[:octicons-arrow-right-24: Continue to Dataset Introduction](introduction.md){ .md-button .md-button--primary }
