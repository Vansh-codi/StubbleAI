import os
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


# ============================================================
# STUBBLEAI DEPLOYMENT MODEL
# Uses all historically labeled data: 2023 + 2024 + 2025
# ============================================================

print("=" * 70)
print("STUBBLEAI DEPLOYMENT MODEL TRAINING")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load historical datasets
# ------------------------------------------------------------

train_df = pd.read_csv("ml_train.csv")
test_df = pd.read_csv("ml_test.csv")

df = pd.concat([train_df, test_df], ignore_index=True)

print(f"\nTotal historical rows: {len(df)}")
print(f"Years available: {sorted(df['date'].str[:4].unique())}")


# ------------------------------------------------------------
# 2. Target
# ------------------------------------------------------------

target_column = "elevated_next_day"

if target_column not in df.columns:
    raise ValueError(f"Missing target column: {target_column}")

y = df[target_column].astype(int)


# ------------------------------------------------------------
# 3. Features
# ------------------------------------------------------------

numeric_features = [
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
]

categorical_features = [
    "state",
    "district",
]

features = numeric_features + categorical_features

X = df[features].copy()


# ------------------------------------------------------------
# 4. Check data
# ------------------------------------------------------------

print("\nMissing values:")
print(X.isnull().sum())

if X.isnull().sum().sum() > 0:
    raise ValueError("Missing values detected. Fix data before training.")


# ------------------------------------------------------------
# 5. Preprocessing
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features,
        ),
    ]
)


# ------------------------------------------------------------
# 6. Random Forest
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    class_weight="balanced",
    min_samples_leaf=2,
    n_jobs=-1,
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", model),
    ]
)


# ------------------------------------------------------------
# 7. Train
# ------------------------------------------------------------

print("\nTraining deployment model...")

pipeline.fit(X, y)

print("✅ Training completed.")


# ------------------------------------------------------------
# 8. Save model
# ------------------------------------------------------------

os.makedirs("model", exist_ok=True)

model_path = "model/stubbleai_deployment_model.pkl"

joblib.dump(pipeline, model_path)

print(f"\n✅ Model saved:")
print(model_path)


# ------------------------------------------------------------
# 9. Save deployment configuration
# ------------------------------------------------------------

config = {
    "model_type": "Random Forest",
    "n_estimators": 400,
    "class_weight": "balanced",
    "min_samples_leaf": 2,
    "random_state": 42,
    "training_period": "2023-2025",
    "target": "elevated_next_day",
    "target_definition": "next_day_fire_count > 2",
    "threshold": 0.40,
    "purpose": "Future/live inference",
    "research_model": "model/stubbleai_final_model.pkl",
}

with open("model/stubbleai_deployment_config.json", "w") as f:
    json.dump(config, f, indent=4)

print("\n✅ Configuration saved:")
print("model/stubbleai_deployment_config.json")

print("\n" + "=" * 70)
print("DEPLOYMENT MODEL READY")
print("=" * 70)