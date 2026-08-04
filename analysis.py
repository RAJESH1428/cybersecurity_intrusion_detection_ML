"""
Network Intrusion Detection: Comparative Evaluation of Classical Supervised,
Classical Unsupervised, and Deep Learning Techniques on Session-Level
Behavioural Features

Author: Rajesh
Module: CMP7239 Applied Machine Learning

Six techniques compared, per module requirement (2 supervised + 2 unsupervised
+ 2 deep learning):
- Supervised (classical): Logistic Regression, Random Forest (tuned)
- Unsupervised (classical): K-Means clustering, Isolation Forest
- Deep Learning: MLP classifier (supervised), Autoencoder (unsupervised,
  anomaly detection via reconstruction error)

Structure
---------
1. Data loading & integrity checks
2. Exploratory Data Analysis (EDA)
3. Preprocessing & feature engineering
4. Classical supervised model training + hyperparameter tuning
4b. Classical unsupervised methods (trained without labels; labels used only
    for post-hoc external validation, not during fitting)
4c. Deep learning models (MLP supervised; Autoencoder unsupervised, threshold
    for anomaly flagging chosen from training reconstruction error only)
5. Unified evaluation across all six methods (Accuracy, Precision, Recall,
   F1, ROC-AUC, PR-AUC), all on the identical held-out test set
6. Visualisation (saved to /figures)
7. Results export (saved to /results)

Run: python src/analysis.py
"""

import os
import json
import time
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, adjusted_rand_score,
    normalized_mutual_info_score, classification_report, silhouette_score
)
from sklearn.decomposition import PCA
import shap

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)

RANDOM_STATE = 42
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Cybersecurity_Intrusion_Detection.csv")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=0.95)


# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=["session_id"])
    return df


# ---------------------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------------------
def run_eda(df: pd.DataFrame) -> dict:
    findings = {}

    findings["shape"] = df.shape
    findings["class_balance"] = df["attack_detected"].value_counts(normalize=True).to_dict()
    findings["missing_values"] = df.isnull().sum().to_dict()

    # Class balance plot
    plt.figure(figsize=(4.5, 4))
    df["attack_detected"].value_counts().sort_index().plot(
        kind="bar", color=["#4C72B0", "#C44E52"]
    )
    plt.xticks([0, 1], ["Benign (0)", "Attack (1)"], rotation=0)
    plt.ylabel("Session count")
    plt.title("Class Distribution: attack_detected")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "01_class_distribution.png"), dpi=150)
    plt.close()

    # Missingness check: is encryption_used missing MCAR, or does it leak signal?
    df_tmp = df.copy()
    df_tmp["encryption_missing"] = df_tmp["encryption_used"].isnull().astype(int)
    miss_rates = df_tmp.groupby("encryption_missing")["attack_detected"].mean()
    findings["attack_rate_by_encryption_missingness"] = miss_rates.to_dict()

    # Numeric feature correlation with target
    num_cols = ["network_packet_size", "login_attempts", "session_duration",
                "ip_reputation_score", "failed_logins"]
    corr = df[num_cols + ["attack_detected"]].corr()
    findings["numeric_correlation_with_target"] = corr["attack_detected"].drop("attack_detected").to_dict()

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Correlation Matrix: Numeric Features vs Target")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "02_correlation_heatmap.png"), dpi=150)
    plt.close()

    # Categorical breakdown vs target
    cat_cols = ["protocol_type", "encryption_used", "browser_type", "unusual_time_access"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col in zip(axes.flat, cat_cols):
        rates = df.groupby(col)["attack_detected"].mean().sort_values(ascending=False)
        rates.plot(kind="bar", ax=ax, color="#55A868")
        ax.set_title(f"Attack rate by {col}")
        ax.set_ylabel("Attack rate")
        ax.axhline(df["attack_detected"].mean(), color="black", linestyle="--", linewidth=1)
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "03_categorical_attack_rates.png"), dpi=150)
    plt.close()

    findings["attack_rate_by_browser"] = df.groupby("browser_type")["attack_detected"].mean().to_dict()

    # Distribution plots for top numeric predictors, split by class
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, ["failed_logins", "login_attempts", "ip_reputation_score"]):
        sns.kdeplot(data=df, x=col, hue="attack_detected", fill=True, common_norm=False, ax=ax, alpha=0.4)
        ax.set_title(f"{col} by class")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "04_top_predictor_distributions.png"), dpi=150)
    plt.close()

    return findings


# ---------------------------------------------------------------------------
# 3. PREPROCESSING & FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # encryption_used missingness was tested against the target during EDA and
    # showed no meaningful association (see findings.json). It is therefore
    # treated as a genuine "unknown/unspecified" category rather than dropped,
    # to avoid discarding ~20% of rows on a feature with no informative gap.
    df["encryption_used"] = df["encryption_used"].fillna("Unknown")

    # Engineered ratio feature: failed_logins relative to total login_attempts.
    # Rationale: a high failed/attempt ratio is a stronger brute-force signal
    # than either raw count alone (e.g. 4 failures out of 4 attempts vs
    # 4 failures out of 20 attempts represent very different risk profiles).
    df["failed_login_ratio"] = df["failed_logins"] / df["login_attempts"].replace(0, 1)

    # Engineered flag: browser_type == "Unknown" showed a markedly higher
    # attack rate in EDA (~73% vs ~42-44% for named browsers), consistent with
    # spoofed or non-standard user-agent strings used by automated/malicious
    # clients. Isolated as its own binary flag so tree models can split on it
    # directly and it is not diluted inside a five-level one-hot feature.
    df["suspicious_browser"] = (df["browser_type"] == "Unknown").astype(int)

    return df


def build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    numeric_pipeline = Pipeline([("scaler", StandardScaler())])
    categorical_pipeline = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])
    return preprocessor


# ---------------------------------------------------------------------------
# 4-5. MODEL TRAINING & TUNING
# ---------------------------------------------------------------------------
def train_models(X_train, X_test, y_train, y_test, preprocessor):
    models = {}
    results = {}
    timing = {}

    # --- Baseline: Logistic Regression (interpretable linear benchmark) ---
    lr_pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])
    t0 = time.perf_counter()
    lr_pipe.fit(X_train, y_train)
    timing["Logistic Regression_train_seconds"] = time.perf_counter() - t0
    models["Logistic Regression"] = lr_pipe
    joblib.dump(lr_pipe, os.path.join(MODELS_DIR, "logistic_regression.joblib"))

    # --- Random Forest, tuned via RandomizedSearchCV ---
    rf_pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE))
    ])
    rf_param_dist = {
        "clf__n_estimators": [100, 200, 300, 400],
        "clf__max_depth": [None, 6, 10, 14, 20],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", "log2"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rf_search = RandomizedSearchCV(
        rf_pipe, rf_param_dist, n_iter=25, scoring="f1", cv=cv,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    t0 = time.perf_counter()
    rf_search.fit(X_train, y_train)
    timing["Random Forest (tuned)_train_seconds"] = time.perf_counter() - t0
    models["Random Forest (tuned)"] = rf_search.best_estimator_
    results["rf_best_params"] = rf_search.best_params_
    results["rf_best_cv_f1"] = rf_search.best_score_
    joblib.dump(rf_search.best_estimator_, os.path.join(MODELS_DIR, "random_forest_tuned.joblib"))

    # --- XGBoost removed: report requires 2 supervised + 2 unsupervised ---

    return models, results, timing


# ---------------------------------------------------------------------------
# 4b. UNSUPERVISED METHODS: K-Means and Isolation Forest
# ---------------------------------------------------------------------------
def train_unsupervised(X_train, X_test, y_train, y_test, preprocessor):
    """
    Both methods are trained WITHOUT using y_train (true unsupervised learning).
    y_test is used only afterward, for external validation -- to check how well
    the unsupervised structure the model found lines up with the real labels.
    This is standard practice for evaluating unsupervised methods against a
    dataset that happens to have ground truth available.
    """
    unsupervised_models = {}
    unsupervised_results = {}
    timing = {}

    # Fit preprocessing on training data only (same convention as supervised models)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # --- K-Means (k=2, unsupervised clustering) ---
    kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10)
    t0 = time.perf_counter()
    kmeans.fit(X_train_proc)
    timing["K-Means_train_seconds"] = time.perf_counter() - t0
    train_clusters = kmeans.predict(X_train_proc)
    t0 = time.perf_counter()
    test_clusters = kmeans.predict(X_test_proc)
    timing["K-Means_test_seconds"] = time.perf_counter() - t0

    # Align cluster IDs to labels via majority vote on the training set only
    # (the mapping itself must not see test labels, to avoid leakage)
    cluster_to_label = {}
    for c in [0, 1]:
        mask = train_clusters == c
        if mask.sum() > 0:
            majority_label = int(round(y_train.values[mask].mean()))
            cluster_to_label[c] = majority_label
    test_pred_kmeans = np.array([cluster_to_label[c] for c in test_clusters])

    unsupervised_models["K-Means"] = {
        "model": kmeans, "preprocessor": preprocessor,
        "cluster_to_label": cluster_to_label, "predict_fn": lambda Xp: np.array(
            [cluster_to_label[c] for c in kmeans.predict(Xp)])
    }
    joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans.joblib"))
    unsupervised_results["kmeans_ari"] = adjusted_rand_score(y_test, test_clusters)
    unsupervised_results["kmeans_nmi"] = normalized_mutual_info_score(y_test, test_clusters)
    # Silhouette score: judges the clusters on their own terms (geometric separation
    # in feature space), independent of the true labels. Computed on the held-out
    # test set, consistent with every other metric in this report -- not on the
    # full dataset, which would mix in points the clustering was fit on.
    unsupervised_results["kmeans_silhouette"] = float(
        silhouette_score(X_test_proc, test_clusters)
    )

    # --- Isolation Forest (unsupervised anomaly detection) ---
    # contamination set to match the dataset's actual attack rate, so the model
    # flags roughly the right proportion of sessions as anomalous
    contamination_rate = y_train.mean()
    iso = IsolationForest(n_estimators=200, contamination=contamination_rate,
                           random_state=RANDOM_STATE)
    t0 = time.perf_counter()
    iso.fit(X_train_proc)
    timing["Isolation Forest_train_seconds"] = time.perf_counter() - t0
    # IsolationForest.predict returns -1 (anomaly) or 1 (normal); map to 1=attack, 0=benign
    t0 = time.perf_counter()
    test_pred_iso = (iso.predict(X_test_proc) == -1).astype(int)
    timing["Isolation Forest_test_seconds"] = time.perf_counter() - t0
    train_pred_iso_raw = iso.predict(X_train_proc)

    unsupervised_models["Isolation Forest"] = {"model": iso, "preprocessor": preprocessor}
    joblib.dump(iso, os.path.join(MODELS_DIR, "isolation_forest.joblib"))

    return (unsupervised_models, unsupervised_results, test_pred_kmeans, test_pred_iso,
            test_clusters, X_test_proc, timing)


# ---------------------------------------------------------------------------
# 4c. DEEP LEARNING MODELS (1 supervised, 1 unsupervised)
# ---------------------------------------------------------------------------
def train_deep_learning(X_train, X_test, y_train, y_test, preprocessor):
    """
    Two deep learning models, mirroring the classical supervised/unsupervised
    split so the report can compare not just "supervised vs unsupervised" but
    also "classical vs deep learning" within each paradigm.

    - MLP classifier: standard feedforward network, trained on labels like
      the classical supervised pair. Same train/test split, same preprocessor
      pattern (fit on training data only).
    - Autoencoder: trained ONLY to reconstruct its own input (no labels used
      at any point during fitting). A session is flagged as an attack if its
      reconstruction error is unusually high -- the network has learned what
      "normal" looks like and struggles to reproduce sessions that deviate
      from that pattern. The anomaly threshold is chosen from the TRAINING
      set's own reconstruction-error distribution (matched to the training
      attack rate), never from the test set, to avoid leaking test
      information into a decision that is supposed to be label-free.
    """
    dl_models = {}
    dl_results = {}
    timing = {}

    # Preprocess once, shared by both deep learning models (fit on train only)
    dl_preprocessor = preprocessor
    X_train_proc = dl_preprocessor.fit_transform(X_train)
    X_test_proc = dl_preprocessor.transform(X_test)
    n_features = X_train_proc.shape[1]

    # --- MLP classifier (supervised deep learning) ---
    mlp = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ], name="MLP_classifier")
    mlp.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )
    t0 = time.perf_counter()
    mlp.fit(
        X_train_proc, y_train.values,
        validation_split=0.15, epochs=100, batch_size=64,
        callbacks=[early_stop], verbose=0
    )
    timing["MLP (Deep Learning)_train_seconds"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    mlp_proba = mlp.predict(X_test_proc, verbose=0).ravel()
    timing["MLP (Deep Learning)_test_seconds"] = time.perf_counter() - t0
    mlp_pred = (mlp_proba >= 0.5).astype(int)
    dl_models["MLP (Deep Learning)"] = mlp
    mlp.save(os.path.join(MODELS_DIR, "mlp_classifier.keras"))

    # --- Autoencoder (unsupervised deep learning, anomaly detection) ---
    bottleneck_dim = max(2, n_features // 4)
    encoder_input = keras.Input(shape=(n_features,))
    encoded = layers.Dense(16, activation="relu")(encoder_input)
    encoded = layers.Dense(bottleneck_dim, activation="relu", name="bottleneck")(encoded)
    decoded = layers.Dense(16, activation="relu")(encoded)
    decoded = layers.Dense(n_features, activation="linear")(decoded)
    autoencoder = keras.Model(encoder_input, decoded, name="Autoencoder")
    autoencoder.compile(optimizer="adam", loss="mse")

    ae_early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )
    # Trained to reconstruct its own input -- y_train is never referenced below.
    t0 = time.perf_counter()
    autoencoder.fit(
        X_train_proc, X_train_proc,
        validation_split=0.15, epochs=100, batch_size=64,
        callbacks=[ae_early_stop], verbose=0
    )
    timing["Autoencoder (Deep Learning)_train_seconds"] = time.perf_counter() - t0

    train_reconstructed = autoencoder.predict(X_train_proc, verbose=0)
    train_recon_error = np.mean(np.square(X_train_proc - train_reconstructed), axis=1)
    t0 = time.perf_counter()
    test_reconstructed = autoencoder.predict(X_test_proc, verbose=0)
    timing["Autoencoder (Deep Learning)_test_seconds"] = time.perf_counter() - t0
    test_recon_error = np.mean(np.square(X_test_proc - test_reconstructed), axis=1)

    # Threshold chosen from TRAINING error distribution only (matched to the
    # training attack rate, same convention used for Isolation Forest's
    # contamination parameter), then applied unchanged to the test set.
    contamination_rate = float(y_train.mean())
    threshold = np.percentile(train_recon_error, 100 * (1 - contamination_rate))
    ae_pred = (test_recon_error >= threshold).astype(int)
    autoencoder.save(os.path.join(MODELS_DIR, "autoencoder.keras"))
    dl_models["Autoencoder (Deep Learning)"] = autoencoder

    dl_results["mlp_val_accuracy_last"] = float(mlp.history.history.get("val_accuracy", [None])[-1] or 0)
    dl_results["autoencoder_threshold"] = float(threshold)
    dl_results["autoencoder_train_recon_error_mean"] = float(train_recon_error.mean())
    dl_results["autoencoder_test_recon_error_mean"] = float(test_recon_error.mean())

    return dl_models, dl_results, mlp_pred, mlp_proba, ae_pred, test_recon_error, timing


# ---------------------------------------------------------------------------
# 6. EVALUATION (unified across supervised and unsupervised methods)
# ---------------------------------------------------------------------------
def evaluate_all(predictions: dict, y_test) -> dict:
    """
    predictions: {name: {"y_pred": array of 0/1, "y_score": continuous array
                          where higher = more likely attack}}
    Works identically for supervised classifiers (score = predict_proba) and
    unsupervised methods (score = a continuous stand-in, e.g. distance-based
    for K-Means or the inverted anomaly score for Isolation Forest), so all
    four methods are compared on exactly the same footing.
    """
    metrics_table = {}
    roc_data = {}
    pr_data = {}

    for name, preds in predictions.items():
        y_pred = preds["y_pred"]
        y_score = preds["y_score"]

        metrics_table[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_score),
            "pr_auc": average_precision_score(y_test, y_score),
        }

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_data[name] = (fpr, tpr)
        prec, rec, _ = precision_recall_curve(y_test, y_score)
        pr_data[name] = (rec, prec)

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"])
        plt.title(f"Confusion Matrix: {name}")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(os.path.join(FIG_DIR, f"05_confusion_matrix_{safe_name}.png"), dpi=150)
        plt.close()

    plt.figure(figsize=(6, 5.5))
    for name, (fpr, tpr) in roc_data.items():
        auc_val = metrics_table[name]["roc_auc"]
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves: All Six Methods")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "06_roc_curves.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5.5))
    for name, (rec, prec) in pr_data.items():
        pr_auc_val = metrics_table[name]["pr_auc"]
        plt.plot(rec, prec, label=f"{name} (AP={pr_auc_val:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves: All Six Methods")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "07_pr_curves.png"), dpi=150)
    plt.close()

    return metrics_table


def plot_feature_importance(models, numeric_features, categorical_features):
    rf_model = models["Random Forest (tuned)"]
    ohe = rf_model.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(ohe.get_feature_names_out(categorical_features))
    all_feature_names = numeric_features + cat_names

    importances = rf_model.named_steps["clf"].feature_importances_
    imp_series = pd.Series(importances, index=all_feature_names).sort_values(ascending=False).head(12)

    plt.figure(figsize=(7, 5))
    imp_series.sort_values().plot(kind="barh", color="#8172B2")
    plt.title("Random Forest: Top 12 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "08_feature_importance.png"), dpi=150)
    plt.close()

    return imp_series.to_dict()


def plot_pca_comparison(X_test_proc, y_test, kmeans_pred, iso_pred):
    """
    The feature matrix has too many dimensions to plot directly. PCA compresses
    it to the 2 directions carrying the most variance so the cluster structure
    can be visually inspected. Three panels, same points, three colourings:
    true labels, K-Means clusters, Isolation Forest flags -- side by side shows
    exactly how far each unsupervised method's structure diverges from ground
    truth. PCA discards information, so points that overlap here may still be
    separable in the full feature space; this is illustrative, not a substitute
    for the quantitative metrics (ARI, NMI, silhouette) reported alongside it.
    """
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_test_proc)
    var_explained = pca.explained_variance_ratio_.sum()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        (y_test.values, "Actual labels (ground truth)"),
        (kmeans_pred, "K-Means clusters"),
        (iso_pred, "Isolation Forest anomalies"),
    ]
    for ax, (colouring, title) in zip(axes, panels):
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colouring, cmap="coolwarm", s=8, alpha=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.suptitle(f"PCA Projection (test set, {var_explained:.1%} variance explained)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "09_pca_comparison.png"), dpi=150)
    plt.close()

    return float(var_explained)


def plot_shap_summary(rf_pipeline, X_test, numeric_features, categorical_features):
    """
    Random Forest's feature_importances_ gives an aggregate ranking but no
    direction. SHAP shows, per session, how much each feature pushed the
    prediction toward "attack" vs "benign" -- extending the aggregate ranking
    into individually-verifiable, directional evidence.
    """
    rf_clf = rf_pipeline.named_steps["clf"]
    X_test_transformed = rf_pipeline.named_steps["prep"].transform(X_test)
    ohe = rf_pipeline.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
    feature_names = numeric_features + list(ohe.get_feature_names_out(categorical_features))

    explainer = shap.TreeExplainer(rf_clf)
    shap_values = explainer.shap_values(X_test_transformed)

    plt.figure()
    # index 1 selects the "attack" class so positive SHAP values push toward attack
    shap.summary_plot(shap_values[:, :, 1], X_test_transformed,
                       feature_names=feature_names, show=False, max_display=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "10_shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    df = load_data(DATA_PATH)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    eda_findings = run_eda(df)
    print("EDA complete. Figures saved to /figures")

    df_fe = engineer_features(df)

    target = "attack_detected"
    numeric_features = [
        "network_packet_size", "login_attempts", "session_duration",
        "ip_reputation_score", "failed_logins", "failed_login_ratio",
        "unusual_time_access", "suspicious_browser"
    ]
    categorical_features = ["protocol_type", "encryption_used", "browser_type"]

    X = df_fe[numeric_features + categorical_features]
    y = df_fe[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    models, tuning_results, supervised_timing = train_models(X_train, X_test, y_train, y_test, preprocessor)
    print("Supervised models trained:", list(models.keys()))

    # Unsupervised methods use their own preprocessor instance (fit independently,
    # since KMeans/IsolationForest are never shown y_train during fitting)
    unsup_preprocessor = build_preprocessor(numeric_features, categorical_features)
    (unsupervised_models, unsupervised_results, kmeans_pred, iso_pred,
     test_clusters, X_test_proc_unsup, unsupervised_timing) = train_unsupervised(
        X_train, X_test, y_train, y_test, unsup_preprocessor
    )
    print("Unsupervised methods trained:", list(unsupervised_models.keys()))
    print("K-Means external validation -- ARI:", round(unsupervised_results["kmeans_ari"], 4),
          "NMI:", round(unsupervised_results["kmeans_nmi"], 4),
          "Silhouette:", round(unsupervised_results["kmeans_silhouette"], 4))

    # Build a unified predictions dict: supervised (predict_proba) + unsupervised
    # (continuous stand-in scores) so all four methods are evaluated identically
    predictions = {}
    for name, model in models.items():
        t0 = time.perf_counter()
        y_pred_supervised = model.predict(X_test)
        supervised_timing[f"{name}_test_seconds"] = time.perf_counter() - t0
        predictions[name] = {
            "y_pred": y_pred_supervised,
            "y_score": model.predict_proba(X_test)[:, 1]
        }

    X_test_proc = unsup_preprocessor.transform(X_test)
    kmeans_model = unsupervised_models["K-Means"]["model"]
    cluster_to_label = unsupervised_models["K-Means"]["cluster_to_label"]
    # Continuous K-Means score: relative distance to the "benign" vs "attack" centroid
    distances = kmeans_model.transform(X_test_proc)
    benign_cluster = [c for c, lbl in cluster_to_label.items() if lbl == 0][0]
    attack_cluster = [c for c, lbl in cluster_to_label.items() if lbl == 1][0]
    kmeans_score = distances[:, benign_cluster] - distances[:, attack_cluster]
    predictions["K-Means"] = {"y_pred": kmeans_pred, "y_score": kmeans_score}

    iso_model = unsupervised_models["Isolation Forest"]["model"]
    # Invert Isolation Forest's decision_function so higher = more anomalous/attack-like
    iso_score = -iso_model.decision_function(X_test_proc)
    predictions["Isolation Forest"] = {"y_pred": iso_pred, "y_score": iso_score}

    # Deep learning: its own preprocessor instance, same independence rule as
    # the classical unsupervised pair -- the autoencoder must never be shown
    # y_train, so it cannot share a preprocessor object that was fit inside a
    # supervised pipeline.
    dl_preprocessor = build_preprocessor(numeric_features, categorical_features)
    dl_models, dl_results, mlp_pred, mlp_proba, ae_pred, ae_recon_error, dl_timing = train_deep_learning(
        X_train, X_test, y_train, y_test, dl_preprocessor
    )
    print("Deep learning models trained:", list(dl_models.keys()))
    print("Autoencoder anomaly threshold (from training data only):",
          round(dl_results["autoencoder_threshold"], 4))

    predictions["MLP (Deep Learning)"] = {"y_pred": mlp_pred, "y_score": mlp_proba}
    predictions["Autoencoder (Deep Learning)"] = {"y_pred": ae_pred, "y_score": ae_recon_error}

    metrics_table = evaluate_all(predictions, y_test)
    feature_importance = plot_feature_importance(models, numeric_features, categorical_features)
    pca_variance_explained = plot_pca_comparison(X_test_proc_unsup, y_test, kmeans_pred, iso_pred)
    plot_shap_summary(models["Random Forest (tuned)"], X_test, numeric_features, categorical_features)
    print("SHAP summary figure saved.")

    # Persist all results for report writing
    output = {
        "eda_findings": eda_findings,
        "tuning_results": tuning_results,
        "unsupervised_results": unsupervised_results,
        "deep_learning_results": dl_results,
        "timing_seconds": {**supervised_timing, **unsupervised_timing, **dl_timing},
        "pca_variance_explained": pca_variance_explained,
        "metrics_table": metrics_table,
        "feature_importance_rf": feature_importance,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "attack_rate_train": float(y_train.mean()),
    }

    def _clean(obj):
        if isinstance(obj, dict):
            return {str(k): _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(_clean(output), f, indent=2)

    # Human-readable metrics table
    metrics_df = pd.DataFrame(metrics_table).T.round(4)
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "metrics_table.csv"))
    print("\n=== Final Test-Set Metrics ===")
    print(metrics_df)

    print("\nAll results saved to /results, all figures saved to /figures")


if __name__ == "__main__":
    main()
