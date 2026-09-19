from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "model" / "stubbleai_final_model.pkl"
TEST_PATH = PROJECT_ROOT / "ml_test.csv"

OUTPUT_DIR = PROJECT_ROOT / "frontend" / "public" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading model...")
model = joblib.load(MODEL_PATH)

print("Loading test dataset...")
df = pd.read_csv(TEST_PATH)

FEATURES = [
    "T2M",
    "RH2M",
    "WS2M",
    "PRECTOTCORR",
    "fire_lag_1d",
    "fire_lag_3d",
    "fire_lag_7d",
    "fire_mean_3d",
    "fire_mean_7d",
    "season_sin",
    "season_cos",
    "state",
    "district",
]

X = df[FEATURES]
y = df["elevated_next_day"].astype(int)

print(f"Test rows: {len(df)}")

# ---------------------------------------------------------
# MODEL PREDICTIONS
# ---------------------------------------------------------

probabilities = model.predict_proba(X)[:, 1]

THRESHOLD = 0.40
predictions = (probabilities >= THRESHOLD).astype(int)

# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

metrics = {
    "accuracy": accuracy_score(y, predictions),
    "balanced_accuracy": balanced_accuracy_score(y, predictions),
    "precision": precision_score(y, predictions),
    "recall": recall_score(y, predictions),
    "f1": f1_score(y, predictions),
    "roc_auc": roc_auc_score(y, probabilities),
    "pr_auc": average_precision_score(y, probabilities),
    "threshold": THRESHOLD,
    "training_year": 2023,
    "validation_year": 2024,
    "test_year": 2025,
    "test_rows": len(df),
}

with open(OUTPUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nFinal metrics:")
for key, value in metrics.items():
    print(f"{key}: {value}")

# ---------------------------------------------------------
# ROC CURVE
# ---------------------------------------------------------

fpr, tpr, _ = roc_curve(y, probabilities)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    linewidth=2.5,
    label=f"Random Forest (AUC = {metrics['roc_auc']:.3f})",
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.5,
    label="Random baseline",
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — 2025 Unseen Test Set")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "roc_curve.png",
    dpi=180,
    bbox_inches="tight",
)

plt.close()

# ---------------------------------------------------------
# PRECISION-RECALL CURVE
# ---------------------------------------------------------

precision, recall, _ = precision_recall_curve(y, probabilities)

plt.figure(figsize=(7, 6))

plt.plot(
    recall,
    precision,
    linewidth=2.5,
    label=f"Random Forest (PR-AUC = {metrics['pr_auc']:.3f})",
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve — 2025 Unseen Test Set")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "pr_curve.png",
    dpi=180,
    bbox_inches="tight",
)

plt.close()

# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------

cm = confusion_matrix(y, predictions)

fig, ax = plt.subplots(figsize=(6, 5))

image = ax.imshow(cm)

ax.set_title("Confusion Matrix — 2025 Unseen Test Set")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["Normal", "Elevated"])
ax.set_yticklabels(["Normal", "Elevated"])

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=16,
        )

fig.colorbar(image, ax=ax)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=180,
    bbox_inches="tight",
)

plt.close()

# ---------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------

try:
    rf = model.named_steps["classifier"]

    preprocessor = model.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()

    importances = rf.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    importance_df.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False,
    )

    top = importance_df.head(12).sort_values("importance")

    plt.figure(figsize=(8, 6))

    plt.barh(
        top["feature"],
        top["importance"],
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "feature_importance.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

except Exception as e:
    print("Feature importance generation skipped:", e)

# ---------------------------------------------------------
# DONE
# ---------------------------------------------------------

print("\nEvaluation artifacts generated:")
print(OUTPUT_DIR)