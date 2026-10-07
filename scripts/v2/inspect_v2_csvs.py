import pandas as pd
from pathlib import Path

base = Path("research/v2")

for h in [1, 2, 3, 5, 7]:
    path = base / f"plus{h}d" / "v2_baseline_results.csv"

    print()
    print("=" * 60)
    print(f"PLUS{h}d")
    print("=" * 60)

    df = pd.read_csv(path)

    print("ROWS:", len(df))
    print("HORIZONS:", df["horizon"].unique().tolist())

    print("\nRF ROWS:")
    print(
        df[df["method"] == "Random Forest"][
            [
                "method",
                "split",
                "horizon",
                "threshold",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
            ]
        ].to_string(index=False)
    )

    print("\nPERSISTENCE ROWS:")
    print(
        df[df["method"] == "Persistence"][
            ["method", "split", "horizon", "f1"]
        ].to_string(index=False)
    )
