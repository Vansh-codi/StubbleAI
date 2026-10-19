from pathlib import Path
import json
import pandas as pd
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "data" / "live" / "v3"
OUT = LIVE / "v3_coverage_evidence_audit.json"

def newest(pattern):
    files = sorted(LIVE.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None

def main():
    LIVE.mkdir(parents=True, exist_ok=True)
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COVERAGE_UNVERIFIED",
        "coverage_verified_rows": 0,
        "zero_fire_days_certified": 0,
        "evidence": {},
        "conclusion": (
            "Detection presence and successful API requests do not establish "
            "per-district satellite observation coverage."
        ),
    }

    assigned_path = newest("firms_live_six_state_assigned_*.csv")
    manifest_path = newest("firms_live_manifest_*.csv")
    feature_path = newest("v3_live_feature_candidates_*.csv")

    if assigned_path:
        df = pd.read_csv(assigned_path)
        date_col = "date" if "date" in df.columns else "acq_date"
        result["evidence"]["assigned_detections"] = {
            "file": assigned_path.name,
            "rows": int(len(df)),
            "districts": int(df["district"].nunique()) if "district" in df else None,
            "states": int(df["state"].nunique()) if "state" in df else None,
            "date_min": str(df[date_col].min()) if date_col in df and len(df) else None,
            "date_max": str(df[date_col].max()) if date_col in df and len(df) else None,
            "meaning": "Detections do not prove complete coverage."
        }

    if manifest_path:
        mf = pd.read_csv(manifest_path)
        result["evidence"]["request_manifest"] = {
            "file": manifest_path.name,
            "rows": int(len(mf)),
            "columns": list(mf.columns),
            "successful_requests": int((mf["status"] == "SUCCESS").sum())
                if "status" in mf else None,
            "meaning": "Request success is not per-district observation coverage."
        }

    if feature_path:
        df = pd.read_csv(feature_path)
        result["evidence"]["feature_candidates"] = {
            "file": feature_path.name,
            "rows": int(len(df)),
            "coverage_verified_true": int(
                df["satellite_coverage_verified"].fillna(False).astype(bool).sum()
            ) if "satellite_coverage_verified" in df else 0,
            "forecast_ready_true": int(
                df["forecast_ready"].fillna(False).astype(bool).sum()
            ) if "forecast_ready" in df else 0,
        }

    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("\nAudit saved:", OUT)

if __name__ == "__main__":
    main()
