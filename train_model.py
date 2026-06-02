"""
Script to train the pointing gesture classifier.
================================================
Reads   : data/gesture_data.csv
Features: 20 Euclidean distances from landmark 1-20 to the wrist.
Labels  : 0 = no pointing  |  1 = pointing (index finger up)

Outputs : model/gesture_model.pkl   (best pipeline: RF or SVM)
          model/scaler.pkl          (saved for real-time inference compatibility)
          model/confusion_matrix.png
          model/feature_importance.png
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection  import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing    import StandardScaler
from sklearn.ensemble         import RandomForestClassifier
from sklearn.svm              import SVC
from sklearn.pipeline         import Pipeline
from sklearn.metrics          import (classification_report, accuracy_score,
                                      ConfusionMatrixDisplay)

# ── Load data ──────────────────────────────────────────────────────────────────

CSV_PATH = "data/gesture_data.csv"
assert os.path.exists(CSV_PATH), "Run prepare_dataset.py first."

df = pd.read_csv(CSV_PATH)
print(f"Dataset: {len(df):,} samples, {df.shape[1]-1} fitur")
print(df["label"].value_counts()
      .rename({0: "Class 0 – no pointing", 1: "Class 1 – pointing"}))

counts    = df["label"].value_counts()
imbalance = counts.min() / counts.max()
if imbalance < 0.7:
    print(f"\nWARNING: Class imbalance detected (ratio {imbalance:.2f})")
    print("  Consider using class_weight='balanced' or oversampling\n")

X = df.drop("label", axis=1).values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Pipeline 1: Random Forest ──────────────────────────────────────────────────
# Ensemble of decision trees. Does not strictly require scaling, but wrapped in
# a Pipeline to ensure consistency and prevent data leakage during CV.

print("\n— Training Random Forest …")
rf_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])
rf_pipe.fit(X_train, y_train)

rf_preds = rf_pipe.predict(X_test)
rf_acc   = accuracy_score(y_test, rf_preds)
print(f"RF  test accuracy : {rf_acc:.4f}")
print(classification_report(y_test, rf_preds,
                             target_names=["No pointing", "Pointing"]))

rf_cv = cross_val_score(rf_pipe, X, y, cv=cv_strategy, scoring="accuracy")
print(f"RF  5-fold CV     : {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")

# ── Pipeline 2: SVM (RBF kernel) ──────────────────────────────────────────────
# Support Vector Machine with RBF kernel. Not a neural network.
# Requires scaling — StandardScaler inside the Pipeline prevents data leakage.

print("\n— Training SVM (RBF) …")
svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced",
        probability=False,
        random_state=42,
    )),
])
svm_pipe.fit(X_train, y_train)

svm_preds = svm_pipe.predict(X_test)
svm_acc   = accuracy_score(y_test, svm_preds)
print(f"SVM test accuracy : {svm_acc:.4f}")
print(classification_report(y_test, svm_preds,
                             target_names=["No pointing", "Pointing"]))

svm_cv = cross_val_score(svm_pipe, X, y, cv=cv_strategy, scoring="accuracy")
print(f"SVM 5-fold CV     : {svm_cv.mean():.4f} ± {svm_cv.std():.4f}")

# ── Select the best model based on CV mean ────────────────────────────────────

print("\n── Comparison ───────────────────────────────────────────────────")
print(f"  Random Forest CV : {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")
print(f"  SVM (RBF)  CV    : {svm_cv.mean():.4f} ± {svm_cv.std():.4f}")

if rf_cv.mean() >= svm_cv.mean():
    best_pipe  = rf_pipe
    best_name  = "Random Forest"
    best_preds = rf_preds
    best_acc   = rf_acc
    print(f"\nSelected Random Forest (higher CV score)")
else:
    best_pipe  = svm_pipe
    best_name  = "SVM (RBF)"
    best_preds = svm_preds
    best_acc   = svm_acc
    print(f"\nSelected SVM (higher CV score)")

# ── Save ───────────────────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)

joblib.dump(best_pipe, "model/gesture_model.pkl")
print(f"Saved: model/gesture_model.pkl  ({best_name})")

# Save scaler separately for compatibility or manual testing
scaler_fitted = best_pipe.named_steps["scaler"]
joblib.dump(scaler_fitted, "model/scaler.pkl")
print("Saved: model/scaler.pkl")

# ── Confusion matrix ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, best_preds,
    display_labels=["No pointing", "Pointing"],
    colorbar=False, ax=ax,
)
ax.set_title(f"Gesture classifier — confusion matrix ({best_name})")
fig.tight_layout()
fig.savefig("model/confusion_matrix.png", dpi=120)
print("Saved: model/confusion_matrix.png")

# ── Feature importance (RF) or hyperplane comparison (SVM) ─────────────────────
fig, ax = plt.subplots(figsize=(7, 3))

if best_name == "Random Forest":
    rf_clf      = best_pipe.named_steps["rf"]
    importances = rf_clf.feature_importances_
    feat_names  = df.drop("label", axis=1).columns.tolist()
    sorted_idx  = importances.argsort()[::-1][:10]
    ax.bar(range(10), importances[sorted_idx], color="#4C72B0")
    ax.set_xticks(range(10))
    ax.set_xticklabels([feat_names[i] for i in sorted_idx], rotation=45, ha="right")
    ax.set_ylabel("Importance")
    ax.set_title("Random Forest — top 10 feature importances")
else:
    # SVM: Show accuracy comparison of RF vs SVM as a bar chart
    ax.bar(["Random Forest", "SVM (RBF)"],
           [rf_cv.mean(), svm_cv.mean()],
           yerr=[rf_cv.std(), svm_cv.std()],
           color=["#4C72B0", "#DD8452"], capsize=6)
    ax.set_ylim(max(0, min(rf_cv.mean(), svm_cv.mean()) - 0.05), 1.0)
    ax.set_ylabel("CV Accuracy")
    ax.set_title("Model comparison — 5-fold CV accuracy")

fig.tight_layout()
fig.savefig("model/feature_importance.png", dpi=120)
print("Saved: model/feature_importance.png")
plt.close("all")

print(f"\nDone. Model: {best_name}, test accuracy: {best_acc:.4f}")