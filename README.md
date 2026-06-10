# Student Graduation vs Dropout Prediction (2-Class)

Predict whether a higher education student will graduate or drop out, using 5 classification models with full visualization.

## Project Structure

```
├── preprocess.py           # Data preprocessing (one-hot encoding + scaling)
├── evaluate.py             # Visualizer class (EDA + model-specific + comparison + SHAP)
├── run_all.py              # Main pipeline
├── README.md
├── models/
│   ├── __init__.py
│   ├── logreg.py           # Logistic Regression
│   ├── rf.py               # Random Forest
│   ├── svm.py              # SVM
│   ├── knn.py              # K-Nearest Neighbors
│   └── mlp.py              # MLP (Deep Learning)
├── results/                # Auto-generated (gitignored)
└── data.csv                # Dataset (gitignored)
```

## Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib shap
```

## Usage

```bash
python run_all.py
```

All figures saved to `results/figures/`, summary CSV to `results/results.csv`.

## Dataset

[UCI Machine Learning Repository - Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

- 4,424 samples → 3,630 after removing `Enrolled`
- Target: `Graduate` (0) vs `Dropout` (1)
- 36 raw features → 155 after one-hot encoding nominal categoricals

## Models

| Model | Type | Special Plots |
|-------|------|---------------|
| Logistic Regression | Linear | Coefficients bar + SHAP |
| Random Forest | Ensemble | Feature importance + SHAP |
| SVM | Kernel | PCA decision boundary |
| KNN | Distance | PCA decision regions + K-value curve |
| MLP | Deep Learning | Loss curve |

## Output Figures (results/figures/)

| Category | Files |
|----------|-------|
| EDA | `target_distribution.png`, `feature_distributions.png`, `correlation_heatmap.png` |
| Model-specific | `logreg_coefficients.png`, `rf_feature_importance.png`, `svm_decision_boundary_pca.png`, `knn_decision_regions_pca.png`, `knn_k_values.png`, `mlp_loss_curve.png` |
| SHAP | `shap_beeswarm_*.png`, `shap_bar_*.png` (RF + LogReg) |
| Per-model | `cm_*.png` (6x confusion matrices), `roc_*.png` (6x ROC curves) |
| Comparison | `comparison_metrics.png`, `comparison_roc_curves.png`, `comparison_pr_curves.png`, `comparison_calibration.png`, `comparison_train_time.png`, `confusion_matrices_grid.png` |
| Learning | `learning_curve_Random_Forest.png`, `learning_curve_MLP.png`, `pairplot_top5_RF.png` |
