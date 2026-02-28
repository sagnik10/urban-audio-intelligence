# Urban Audio Intelligence

A production-grade machine learning intelligence pipeline for environmental sound classification using the UrbanSound8K metadata dataset.

This repository provides an end-to-end analytical framework including adaptive visualization, dimensionality reduction, clustering validation, feature relevance analysis, supervised classification, and automated PDF reporting. The pipeline is compatible with both local environments and Kaggle notebooks.

---

## Overview

Urban Audio Intelligence is designed to transform structured audio metadata into interpretable insights.

The system performs:

- Data preprocessing and scaling
- Principal Component Analysis (PCA)
- Unsupervised clustering with Silhouette validation
- Feature importance ranking using Mutual Information
- Supervised multi-class classification using Random Forest
- Confusion matrix evaluation
- Adaptive visualization with labeled axes
- Automated PDF intelligence report generation
- Dual compatibility (Local + Kaggle environments)

---

## Repository Structure

```
Urban_Audio_Intelligence_Output/
 ├── charts/
 ├── models/
 └── Urban_Audio_Intelligence_Report.pdf

Analyzer.py
UrbanSound8K.csv
README.md
```

---

## Features

### 1. Adaptive Visualization

- Proper X/Y labels
- Dynamic axis scaling
- PCA variance percentage labeling
- Grid overlays for readability
- Labeled confusion matrix

### 2. Dimensionality Reduction

- PCA with cumulative explained variance
- 2D projection visualization
- Variance-aware component labeling

### 3. Clustering Diagnostics

- KMeans clustering
- Silhouette score evaluation

### 4. Feature Intelligence

- Mutual Information feature ranking
- Horizontal importance visualization

### 5. Supervised Learning

- Stratified train-test split
- Random Forest classifier
- Zero-division-safe classification report

### 6. Automated Reporting

- Structured PDF generation
- Embedded charts
- Performance summary
- Execution time tracking

---

## Installation

### Local Installation (Windows / macOS / Linux)

```bash
pip install numpy pandas matplotlib seaborn scikit-learn reportlab
```

---

### Kaggle Notebook Installation

Only required package:

```python
!pip install reportlab
```

All other dependencies are pre-installed in Kaggle.

---

## Usage

### Local Environment

1. Place `UrbanSound8K.csv` in the same directory as `Analyzer.py`.
2. Run:

```bash
python Analyzer.py
```

---

### Kaggle Environment

1. Add the UrbanSound8K dataset to your notebook.
2. Run the script directly.
3. Output will be saved inside:

```
/kaggle/working/Urban_Audio_Intelligence_Output/
```

---

## Output

After execution, the following artifacts are generated:

```
Urban_Audio_Intelligence_Output/
 ├── charts/
 │    ├── class_distribution.png
 │    ├── pca_variance.png
 │    ├── pca_projection.png
 │    ├── feature_relevance.png
 │    └── confusion_matrix.png
 │
 ├── models/
 │    ├── scaler.pkl
 │    ├── pca.pkl
 │    └── urban_audio_classifier.pkl
 │
 └── Urban_Audio_Intelligence_Report.pdf
```

---

## Dataset

This project is designed for the UrbanSound8K dataset:

- 8,732 labeled sound excerpts
- 10 environmental sound classes
- Structured metadata for each audio sample

Official Dataset:
https://urbansounddataset.weebly.com/urbansound8k.html

---

## Model Details

Classifier: Random Forest  
Feature Scaling: StandardScaler  
Dimensionality Reduction: PCA  
Feature Selection Metric: Mutual Information  
Clustering: KMeans  
Validation: Silhouette Score + Confusion Matrix  

---

## Performance Metrics

- Precision
- Recall
- F1-score
- Support
- Silhouette Score
- Execution Time

---

## Environment Compatibility

| Feature | Local | Kaggle |
|----------|--------|--------|
| CSV Detection | ✓ | ✓ |
| Adaptive Scaling | ✓ | ✓ |
| PDF Generation | ✓ | ✓ |
| Model Persistence | ✓ | ✓ |

---

## Future Enhancements

- Fold-aware evaluation (UrbanSound8K official folds)
- MFCC extraction from raw WAV files
- Deep CNN model integration
- SHAP explainability
- Interactive dashboard (Plotly / Streamlit)
- Cross-validation benchmarking suite

---

## License

MIT License

---

## Author

Sagnik  

---

## Project Vision

Urban Audio Intelligence bridges structured metadata analytics with interpretable machine learning.  
The goal is not just classification, but structural understanding of environmental acoustic data.
