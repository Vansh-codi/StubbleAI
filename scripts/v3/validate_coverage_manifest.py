from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "live" / "v3" / "district_day_coverage_manifest.csv"

REQUIRED = [
    "date", "state", "district", "observation_status",
    "source", "evidence_file",
]

ALLOWED = {"OBSERVED", "NOT_OBSERVED", "UNKNOWN"}


def main():
    print("V3 District-Day Coverage Manifest Validator")
    print("=" * 48)

    if not MANIFEST.exists():
        print("BLOCKED: no coverage manifest exists yet.")
        print(f"Expected path: {MANIFEST}")
        print("\nRequired CSV columns:")
        print(", ".join(REQUIRED))
        print("\nAllowed observation_status values:")
        print(", ".join(sorted(ALLOWED)))
        print(
            "\nImportant: a FIRMS detection count or successful API "
            "request is not proof of district-day observation coverage."
        )
        print(
            "Do not mark a district OBSERVED unless the referenced "
            "evidence actually supports that claim."
        )
        raise SystemExit(2)

    df = pd.read_csv(MANIFEST, dtype=str).fillna("")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"BLOCKED: missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    invalid_dates = int(df["date"].isna().sum())
    invalid_status = ~df["observation_status"].isin(ALLOWED)
    blank_identity = (
        df["state"].str.strip().eq("")
        | df["district"].str.strip().eq("")
        | df["source"].str.strip().eq("")
        | df["evidence_file"].str.strip().eq("")
    )
    duplicate_keys = int(
        df.duplicated(["date", "state", "district"]).sum()
    )

    print(f"Manifest rows: {len(df)}")
    print(f"Invalid dates: {invalid_dates}")
    print(f"Invalid statuses: {int(invalid_status.sum())}")
    print(f"Rows missing identity/evidence fields: {int(blank_identity.sum())}")
    print(f"Duplicate date-state-district keys: {duplicate_keys}")
    print("\nStatus counts:")
    print(df["observation_status"].value_counts(dropna=False).to_string())

    errors = (
        invalid_dates
        + int(invalid_status.sum())
        + int(blank_identity.sum())
        + duplicate_keys
    )
    if errors:
        raise SystemExit(
            "\nBLOCKED: manifest validation failed. "
            "No forecast-readiness flags were changed."
        )

    observed = df["observation_status"].eq("OBSERVED").sum()
    print(f"\nStructurally valid rows: {len(df)}")
    print(f"Rows claiming observation: {observed}")
    print(
        "STRUCTURE ONLY: this validator checks format, not the truth "
        "or completeness of satellite coverage evidence."
    )


if __name__ == "__main__":
    main()
