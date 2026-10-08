from pathlib import Path
import pandas as pd
import json

BASE = Path(".")
V2   = BASE / "research" / "v2"

# ============================================================
# 1. ALL V2 BASELINE CSVs — RF TEST ROWS
# ============================================================

print("=" * 80)
print("1. V2 BASELINE CSVs — RF TEST ROWS")
print("=" * 80)

for p in sorted(V2.glob("plus*/v2_baseline_results.csv")):
    df = pd.read_csv(p)
    print(f"\nFILE: {p}")
    print(f"  methods:  {df['method'].unique().tolist()}")
    print(f"  splits:   {df['split'].unique().tolist()}")
    print(f"  horizons: {df['horizon'].unique().tolist()}")
    rf = df[(df["method"] == "Random Forest") & (df["split"] == "test")]
    if rf.empty:
        print("  RF TEST ROW: NONE")
    else:
        print(rf[["horizon","threshold","accuracy","precision",
                   "recall","f1","roc_auc","pr_auc"]].to_string(index=False))


# ============================================================
# 2. SPATIAL RESULTS FILE — EXACT SCHEMA
# ============================================================

print("\n" + "=" * 80)
print("2. SPATIAL RESULTS — EXACT SCHEMA AND HORIZON VALUES")
print("=" * 80)

spatial_files = list((V2 / "v2_3" / "spatial").glob("v2_3_spatial_results*"))
print(f"Files found: {[f.name for f in spatial_files]}")

for f in spatial_files:
    df = pd.read_csv(f)
    print(f"\nFILE: {f.name}")
    print(f"  columns:  {df.columns.tolist()}")
    print(f"  horizons: {df['horizon'].unique().tolist()}")
    if "arm" in df.columns:
        print(f"  arms:     {df['arm'].unique().tolist()}")
    if "split" in df.columns:
        print(f"  splits:   {df['split'].unique().tolist()}")
    print("\n  Full contents:")
    print(df.to_string(index=False))


# ============================================================
# 3. THRESHOLD FILES — WHERE DO THEY LIVE?
# ============================================================

print("\n" + "=" * 80)
print("3. THRESHOLD FILES — ALL LOCATIONS")
print("=" * 80)

all_thresh = sorted(V2.glob("**/*threshold*"))
print(f"All threshold files under research/v2:")
for f in all_thresh:
    print(f"  {f}")
    if f.suffix == ".json":
        with open(f) as fh:
            data = json.load(fh)
        print(f"    keys: {list(data.keys())}")
        if "models" in data:
            print(f"    models: {list(data['models'].keys())}")
        if "horizon" in data:
            print(f"    horizon: {data['horizon']}")


# ============================================================
# 4. TREND RESULTS — SCHEMA CHECK
# ============================================================

print("\n" + "=" * 80)
print("4. TREND RESULTS — EXACT SCHEMA AND RF ROWS")
print("=" * 80)

trend_file = V2 / "v2_3" / "v2_3_trend_results.csv"

if trend_file.exists():
    df = pd.read_csv(trend_file)

    print(f"Columns: {df.columns.tolist()}")
    print(f"Horizons: {df['horizon'].unique().tolist()}")
    print(f"Rows: {len(df)}")

    print("\nTrend result rows:")
    print(df.to_string(index=False))
else:
    print(f"NOT FOUND: {trend_file}")

# ============================================================
# 5. V2.7 — CONFIRM RF +2d ROW EXISTS
# ============================================================

print("\n" + "=" * 80)
print("5. V2.7 LEARNER RESULTS — ALL RF ROWS")
print("=" * 80)

v27_file = V2 / "v2_7" / "results" / "v2_7_learner_results.csv"
if v27_file.exists():
    df = pd.read_csv(v27_file)
    print(f"Columns: {df.columns.tolist()}")
    print(f"Models:  {df['model'].unique().tolist()}")
    print(f"Horizons: {df['horizon'].unique().tolist()}")
    rf_rows = df[df["model"] == "rf"]
    print(f"\nAll RF rows:")
    print(rf_rows[["horizon","test_f1","test_roc_auc",
                   "test_accuracy","test_precision","test_recall"]].to_string(index=False))
else:
    print(f"NOT FOUND: {v27_file}")