# Network Intrusion Detection: 4 Classical ML + 2 Deep Learning Techniques

CMP7239 Applied Machine Learning — Coursework Implementation

Rajesh Babu Salagala — Student ID: 26111961

## Overview

Six techniques implemented and compared for detecting intrusions from
session-level network and authentication behaviour features, per module
requirement: **4 classical ML algorithms + 2 deep learning algorithms**.

- **Classical supervised**: Logistic Regression, Random Forest (tuned)
- **Classical unsupervised**: K-Means clustering, Isolation Forest
- **Deep learning**: MLP classifier (supervised), Autoencoder (unsupervised,
  anomaly detection via reconstruction error)

## Dataset

`Cybersecurity_Intrusion_Detection.csv` — 9,537 sessions, 10 features,
binary target `attack_detected` (44.7% positive class).

## Repository Contents

- `cybersecurity_intrusion_detection_notebook_FINAL.ipynb` — full pipeline:
  EDA → feature engineering → training (all 6 methods) → evaluation → figures
- `logistic_regression.joblib`, `random_forest_tuned.joblib`,
  `kmeans.joblib`, `isolation_forest.joblib` — saved classical models
- `mlp_classifier.keras`, `autoencoder.keras` — saved deep learning models

## Running

Open the notebook in Google Colab, run all cells top to bottom, and upload
`Cybersecurity_Intrusion_Detection.csv` when prompted. Reproducible with
`RANDOM_STATE = 42` set throughout.

## Loading a saved model

```python
import joblib
from tensorflow import keras

rf_model = joblib.load("random_forest_tuned.joblib")
mlp_model = keras.models.load_model("mlp_classifier.keras")
```

## Method Summary

1. **EDA**: class balance, missingness check, correlation analysis, categorical
   attack-rate breakdowns (notably `browser_type == "Unknown"` at 73.1% attack rate).
2. **Feature engineering**: `failed_login_ratio` and `suspicious_browser`.
3. **Classical supervised**: Logistic Regression (baseline) and Random Forest
   (tuned via `RandomizedSearchCV`, 5-fold stratified CV, F1-optimised).
4. **Classical unsupervised**: K-Means (k=2) and Isolation Forest (contamination
   matched to training attack rate) — both trained without label access;
   labels used only for post-hoc external validation (ARI, NMI, Silhouette).
5. **Deep learning**: MLP (64→32→1, dropout, early stopping) and Autoencoder
   (reconstruction-error anomaly detection, threshold from training data only).
6. **Evaluation**: identical held-out 20% stratified test set for all six
   methods; accuracy, precision, recall, F1, ROC-AUC, PR-AUC, plus
   training/testing time for each model.

## Key Findings

- **Learning paradigm matters far more than technique sophistication.**
  Random Forest (F1 0.855) and MLP (F1 0.853) are statistically indistinguishable;
  K-Means (0.478), Isolation Forest (0.534), and Autoencoder (0.539) are all
  similarly weak — a deep learning unsupervised method fails for the same
  structural reason as classical unsupervised methods.
- K-Means' near-zero ARI (0.047), NMI (0.030), and Silhouette (0.131) confirm
  this is a genuine structural gap, not a metric artefact.
- `failed_logins`, `login_attempts`, and `ip_reputation_score` are the
  strongest predictors (confirmed via both Random Forest feature importance
  and SHAP), consistent with brute-force/credential-stuffing intrusion patterns.
- **Testing time**: the two deep learning models are 7–1,000x slower at
  inference than the four classical methods, with no accuracy advantage over
  Random Forest — for real-time deployment, Random Forest dominates on both
  accuracy and latency simultaneously.
- Random Forest achieves zero false positives on the held-out test set.
  Genuine and reproducible, but treated with appropriate caution given the
  dataset's clean, low-noise structure.
