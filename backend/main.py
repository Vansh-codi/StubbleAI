from pathlib import Path
import logging
import os
import secrets
import subprocess
import sys

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


# ============================================================
# Environment
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load local .env when running locally.
# In Railway/Render, actual environment variables take precedence.
load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=os.getenv("STUBBLEAI_LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("stubbleai")


# ============================================================
# Configuration
# ============================================================

FRONTEND_ORIGINS = os.getenv(
    "FRONTEND_ORIGIN",
    "http://localhost:5173",
)

# Allow multiple origins:
# FRONTEND_ORIGIN=http://localhost:5173,https://your-app.vercel.app
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in FRONTEND_ORIGINS.split(",")
    if origin.strip()
]

# Shared secret required to trigger the live prediction pipeline.
# Set this in your deployment platform's environment variables
# (Railway/Render), never in frontend code, never committed.
PREDICTION_TRIGGER_KEY = os.getenv("PREDICTION_TRIGGER_KEY")

if not PREDICTION_TRIGGER_KEY:
    logger.warning(
        "PREDICTION_TRIGGER_KEY is not set. "
        "/api/run-prediction will reject all requests until it is configured."
    )


PREDICTIONS_FILE = PROJECT_ROOT / "stubbleai_2026_predictions.csv"
PREDICTION_HISTORY_FILE = (
    PROJECT_ROOT / "stubbleai_2026_prediction_history.csv"
)
PREDICTION_SCRIPT = PROJECT_ROOT / "predict_2026.py"
TRACKER_FILE = PROJECT_ROOT / "prediction_tracker.csv"

# ============================================================
# Rate limiting
# ============================================================
# Protects read endpoints from casual abuse and the (auth-gated)
# prediction trigger from being hammered even by a holder of the key.
# In-memory limiter is fine for a single-instance deployment
# (Railway/Render single service). For multi-instance deployments,
# back this with Redis instead (slowapi supports a storage_uri).

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="StubbleAI API",
    description=(
        "Backend API for the StubbleAI crop-residue "
        "fire-risk dashboard."
    ),
    version="1.0.1",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Trigger-Key"],
)


# ============================================================
# Helpers
# ============================================================

def load_predictions() -> pd.DataFrame:
    """
    Load and validate the latest generated district predictions.
    """
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            "Prediction file is not available."
        )

    df = pd.read_csv(PREDICTIONS_FILE)

    required_columns = [
        "prediction_date",
        "state",
        "district",
        "risk_probability",
        "risk_label",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Prediction data is missing required columns: "
            f"{missing_columns}"
        )

    if df.empty:
        return df

    df["prediction_date"] = pd.to_datetime(
        df["prediction_date"],
        errors="coerce"
    )

    if df["prediction_date"].isna().any():
        raise ValueError(
            "Prediction data contains invalid dates."
        )

    df["risk_probability"] = pd.to_numeric(
        df["risk_probability"],
        errors="coerce"
    )

    if df["risk_probability"].isna().any():
        raise ValueError(
            "Prediction data contains invalid risk probabilities."
        )

    if (
        (df["risk_probability"] < 0)
        | (df["risk_probability"] > 1)
    ).any():
        raise ValueError(
            "Prediction probabilities must be between 0 and 1."
        )

    valid_labels = {"Normal", "Elevated"}

    invalid_labels = set(
        df["risk_label"].dropna().astype(str).unique()
    ) - valid_labels

    if invalid_labels:
        raise ValueError(
            f"Invalid risk labels found: {invalid_labels}"
        )

    return df
def load_prediction_history() -> pd.DataFrame:
    """
    Load and validate persistent 2026 prediction history.
    """

    if not PREDICTION_HISTORY_FILE.exists():
        raise FileNotFoundError(
            "Prediction history file is not available."
        )

    df = pd.read_csv(
        PREDICTION_HISTORY_FILE
    )

    required_columns = [
        "prediction_date",
        "state",
        "district",
        "risk_probability",
        "risk_label",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Prediction history is missing required columns: "
            f"{missing_columns}"
        )

    if df.empty:
        return df

    df["prediction_date"] = pd.to_datetime(
        df["prediction_date"],
        errors="coerce"
    )

    if df["prediction_date"].isna().any():
        raise ValueError(
            "Prediction history contains invalid dates."
        )

    if not df["prediction_date"].dt.year.eq(2026).all():
        raise ValueError(
            "Prediction history contains non-2026 records."
        )

    df["risk_probability"] = pd.to_numeric(
        df["risk_probability"],
        errors="coerce"
    )

    if df["risk_probability"].isna().any():
        raise ValueError(
            "Prediction history contains invalid "
            "risk probabilities."
        )

    if (
        (df["risk_probability"] < 0)
        | (df["risk_probability"] > 1)
    ).any():
        raise ValueError(
            "Prediction history probabilities must "
            "be between 0 and 1."
        )

    valid_labels = {
        "Normal",
        "Elevated"
    }

    invalid_labels = set(
        df["risk_label"]
        .dropna()
        .astype(str)
        .unique()
    ) - valid_labels

    if invalid_labels:
        raise ValueError(
            "Invalid risk labels found in prediction "
            f"history: {invalid_labels}"
        )

    duplicate_rows = df.duplicated(
        subset=[
            "prediction_date",
            "state",
            "district",
        ],
        keep=False
    )

    if duplicate_rows.any():
        raise ValueError(
            "Prediction history contains duplicate "
            "district/date records."
        )

    return df

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert pandas values to JSON-safe Python values.
    """
    result = df.copy()

    for column in result.columns:
        if pd.api.types.is_datetime64_any_dtype(result[column]):
            result[column] = result[column].dt.strftime(
                "%Y-%m-%d"
            )

    return result.astype(object).where(
        pd.notna(result),
        None
    )


def verify_trigger_key(x_trigger_key: str | None) -> None:
    """
    Constant-time check of the shared secret required to run the
    live prediction pipeline. Raises 401/503 on failure.
    """
    if not PREDICTION_TRIGGER_KEY:
        raise HTTPException(
            status_code=503,
            detail="Prediction trigger is not configured on this server.",
        )

    if not x_trigger_key or not secrets.compare_digest(
        x_trigger_key, PREDICTION_TRIGGER_KEY
    ):
        raise HTTPException(status_code=401, detail="Unauthorized")


# ============================================================
# Basic endpoints
# ============================================================

@app.get("/")
def root():
    return {
        "project": "StubbleAI",
        "status": "running",
        "service": "FastAPI backend",
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "service": "StubbleAI FastAPI backend",
    }


# ============================================================
# Prediction endpoints
# ============================================================

@app.get("/api/predictions")
@limiter.limit("60/minute")
def get_predictions(
    request: Request,
    date: str | None = None,
):
    try:
        if date is None:
            df = load_predictions()

        else:
            try:
                requested_date = pd.to_datetime(
                    date,
                    format="%Y-%m-%d",
                    errors="raise"
                )
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Invalid date format. "
                        "Use YYYY-MM-DD."
                    ),
                )

            history = load_prediction_history()

            df = history[
                history["prediction_date"] == requested_date
            ].copy()

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No predictions are available "
                    f"for date {date}."
                    if date
                    else "No predictions are currently available."
                ),
            )

        df = clean_dataframe(df)

        if "prediction_date" not in df.columns:
            raise HTTPException(
                status_code=500,
                detail="Prediction output is missing prediction_date.",
            )

        if df["prediction_date"].nunique() != 1:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Prediction output contains "
                    "inconsistent prediction dates."
                ),
            )

        return {
            "prediction_date": df["prediction_date"].iloc[0],
            "total_districts": len(df),
            "predictions": df.to_dict(
                orient="records"
            ),
        }

    except HTTPException:
        raise

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Prediction data is not available.",
        )

    except Exception:
        logger.exception(
            "Failed to load predictions."
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to load prediction data.",
        )
@app.get("/api/prediction-dates")
@limiter.limit("60/minute")
def get_prediction_dates(request: Request):
    try:
        history = load_prediction_history()

        if history.empty:
            return {
                "dates": []
            }

        dates = sorted(
            history["prediction_date"]
            .dt.strftime("%Y-%m-%d")
            .unique(),
            reverse=True,
        )

        return {
            "dates": dates
        }

    except FileNotFoundError:
        return {
            "dates": []
        }

    except Exception:
        logger.exception(
            "Failed to load prediction dates."
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to load prediction dates.",
        )
@app.get("/api/predictions/{district}")
@limiter.limit("60/minute")
def get_district_prediction(request: Request, district: str):
    try:
        df = load_predictions()
        if df.empty:
          raise HTTPException(
        status_code=404,
        detail="No predictions are currently available.",
    )

        if "district" not in df.columns:
            raise HTTPException(
                status_code=500,
                detail="Prediction data is missing the district field.",
            )

        result = df[
            df["district"].astype(str).str.lower()
            == district.strip().lower()
        ]

        if result.empty:
            raise HTTPException(
                status_code=404,
                detail=f"District '{district}' not found.",
            )

        row = result.iloc[0]

        return clean_dataframe(
            row.to_frame().T
        ).iloc[0].to_dict()

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Failed to load prediction for district: %s",
            district,
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to load district prediction.",
        )


# ============================================================
# Run live prediction (protected)
# ============================================================

@app.post("/api/run-prediction")
@limiter.limit("6/hour")
def run_prediction(
    request: Request,
    x_trigger_key: str | None = Header(
        default=None,
        alias="X-Trigger-Key",
        max_length=256,
    ),
):
    """
    Execute the trusted local prediction pipeline.

    Protected by a shared-secret header (X-Trigger-Key) so the
    pipeline cannot be triggered by arbitrary visitors, and rate
    limited on top of that so even a key holder can't hammer it.

    This endpoint should be exposed only through the backend.
    The prediction script path is fixed and is never supplied
    by the client.
    """

    verify_trigger_key(x_trigger_key)

    if not PREDICTION_SCRIPT.exists():
        logger.error(
            "Prediction script not found: %s",
            PREDICTION_SCRIPT,
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction pipeline is unavailable.",
        )

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(PREDICTION_SCRIPT),
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )

        if result.returncode != 0:
            logger.error(
                "Prediction pipeline failed with return code %s",
                result.returncode,
            )

            raise HTTPException(
                status_code=500,
                detail="Prediction pipeline failed. Check backend logs.",
            )

        df = load_predictions()

        if df.empty:
            raise HTTPException(
                status_code=500,
                detail="Prediction pipeline returned no results.",
            )       

        df = clean_dataframe(df)

        if "prediction_date" not in df.columns:
            raise HTTPException(
                status_code=500,
                detail="Prediction output is missing prediction_date.",
            )

        if df["prediction_date"].nunique() != 1:
            raise HTTPException(
                status_code=500,
                detail="Prediction data contains inconsistent prediction dates.",
            )

        elevated = int(
            (df["risk_label"] == "Elevated").sum()
        )

        normal = int(
            (df["risk_label"] == "Normal").sum()
        )

        logger.info(
            "Live prediction completed successfully: %s districts",
            len(df),
        )

        

        return {
            "status": "success",
            "message": "Live prediction completed successfully.",
            "total_districts": len(df),
            "normal_count": normal,
            "elevated_count": elevated,
            "prediction_date": df["prediction_date"].iloc[0],
            "predictions": df.to_dict(orient="records"),
        }

    except subprocess.TimeoutExpired:
        logger.error(
            "Prediction pipeline timed out after 180 seconds."
        )

        raise HTTPException(
            status_code=504,
            detail="Prediction pipeline timed out.",
        )

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Unexpected error while running prediction."
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to complete live prediction.",
        )

# ============================================================
# Prediction tracker
# ============================================================

@app.get("/api/tracker")
@limiter.limit("60/minute")
def get_tracker(request: Request):
    if not TRACKER_FILE.exists():
        return {
            "rows": [],
            "summary": {
                "total": 0,
                "pending": 0,
                "verified": 0,
                "correct": 0,
                "incorrect": 0,
            },
        }

    try:
        df = pd.read_csv(TRACKER_FILE)
        df = clean_dataframe(df)

        rows = df.to_dict(orient="records")

        if "verification_status" in df.columns:
            verification = (
                df["verification_status"]
                .fillna("Pending")
                .astype(str)
            )
        else:
            verification = pd.Series(
                ["Pending"] * len(df),
                index=df.index,
            )

        verified_count = int(
            (verification != "Pending").sum()
        )

        correct_count = int(
            (verification == "Correct").sum()
        )

        incorrect_count = int(
            (verification == "Incorrect").sum()
        )

        summary = {
            "total": len(df),
            "pending": int(
                (verification == "Pending").sum()
            ),
            "verified": verified_count,
            "correct": correct_count,
            "incorrect": incorrect_count,
            "verification_coverage": (
                round(
                    verified_count / len(df),
                    4
                )
                if len(df) > 0
                else 0.0
            ),
            "verified_accuracy": (
                round(
                    correct_count / verified_count,
                    4
                )
                if verified_count > 0
                else None
            ),
        }

        return {
            "rows": rows,
            "summary": summary,
        }

    except Exception:
        logger.exception(
            "Failed to load prediction tracker."
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to load prediction tracker.",
        )
    

# ============================================================
# Dashboard summary
# ============================================================

@app.get("/api/summary")
@limiter.limit("60/minute")
def get_summary(request: Request):
    try:
        df = load_predictions()

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail="No predictions are currently available.",
            )

        if "district" not in df.columns:
            raise HTTPException(
                status_code=500,
                detail="Prediction data is missing the district field.",
            )

        total = len(df)

        elevated = int(
            (df["risk_label"] == "Elevated").sum()
        )

        normal = int(
            (df["risk_label"] == "Normal").sum()
        )

        highest_risk = df.loc[
            df["risk_probability"].idxmax()
        ]

        return {
            "prediction_date": df["prediction_date"].iloc[0],
            "total_districts": total,
            "normal_count": normal,
            "elevated_count": elevated,
            "highest_risk_district": highest_risk["district"],
            "highest_risk_state": highest_risk["state"],
            "highest_risk_probability": float(
                highest_risk["risk_probability"]
            ),
            "highest_risk_label": highest_risk["risk_label"],
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Failed to generate dashboard summary."
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to generate dashboard summary.",
        )