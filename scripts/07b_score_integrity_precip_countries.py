from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# repo imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trust_bi.common.settings import Settings  # noqa: E402
from trust_bi.warehouse.duckdb_io import DuckDBClient  # noqa: E402

# -----------------------------
# Config
# -----------------------------
# Country KPI (ISO3 level)
KPI_ID = "precip_country_monthly_sum"
SOURCE_ID = "openmeteo"
GEO_LIKE = "ISO3:%"


# -----------------------------
# Helpers (identical logic)
# -----------------------------
def robust_std(x: pd.Series) -> float:
    """Robust std using MAD."""
    x = x.dropna()
    if len(x) < 3:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad


def grade(x: float) -> str:
    if x >= 0.85:
        return "A"
    if x >= 0.70:
        return "B"
    if x >= 0.50:
        return "C"
    return "D"



def blocking_reason(row) -> str | None:
    if row["coverage_score"] < 0.6:
        return "low_coverage"
    if row["missingness_score"] < 0.6:
        return "high_missingness"
    if row["uncertainty_score"] < 0.6:
        return "high_volatility"
    return None


def score_one_geo(g: pd.DataFrame, geo_id: str) -> pd.DataFrame:
    """
    g: DataFrame with columns [date, value, coverage_pct] for a single geo_id.
    Returns rows aligned to fact_integrity_scores schema.
    """
    g = g.copy()
    g["date"] = pd.to_datetime(g["date"])
    g = g.sort_values("date").set_index("date")

    # Coverage score
    g["coverage_score"] = pd.to_numeric(g["coverage_pct"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    # Missingness score (per-month presence)
    g["missingness_score"] = 1.0 - g["value"].isna().astype(float)

    # Uncertainty score (robust volatility on level)
    rstd = robust_std(g["value"])
    if np.isnan(rstd) or rstd == 0:
        g["uncertainty_score"] = 1.0
    else:
        z = np.abs(g["value"] - g["value"].median()) / rstd
        g["uncertainty_score"] = np.exp(-z).clip(lower=0.0, upper=1.0)

    # Source stability score (single source → stable by definition)
    g["source_stability_score"] = 1.0

    # Confidence score (mean of components)
    g["confidence_score"] = (
        g["coverage_score"]
        + g["missingness_score"]
        + g["uncertainty_score"]
        + g["source_stability_score"]
    ) / 4.0

    g["confidence_grade"] = g["confidence_score"].apply(grade)

    # Blocking reason (diagnostic only)
    g["blocking_reason"] = g.apply(blocking_reason, axis=1)

    out = pd.DataFrame(
        {
            "kpi_id": KPI_ID,
            "geo_id": geo_id,
            "date": g.index.date,
            "coverage_score": g["coverage_score"].astype(float),
            "missingness_score": g["missingness_score"].astype(float),
            "uncertainty_score": g["uncertainty_score"].astype(float),
            "source_stability_score": g["source_stability_score"].astype(float),
            "confidence_score": g["confidence_score"].astype(float),
            "confidence_grade": g["confidence_grade"].astype(str),
            "blocking_reason": g["blocking_reason"].astype(object),
            "created_at": datetime.utcnow(),
        }
    )
    return out


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    s = Settings.load(".env")
    db = DuckDBClient(s.duckdb_path)

    print(f"DuckDB path: {s.duckdb_path}")
    print(f"Scoring integrity for {KPI_ID} @ ISO3 countries")

    df = db.query_df(
        f"""
        SELECT
            geo_id,
            date,
            value,
            coverage_pct
        FROM fact_kpi_timeseries
        WHERE kpi_id = '{KPI_ID}'
          AND geo_id LIKE '{GEO_LIKE}'
          AND source_id = '{SOURCE_ID}'
        ORDER BY geo_id, date;
        """
    )

    if df.empty:
        raise RuntimeError(f"No data found for {KPI_ID} @ ISO3:* (source={SOURCE_ID}).")

    outs = []
    for geo_id, g in df.groupby("geo_id", sort=False):
        outs.append(score_one_geo(g, geo_id=str(geo_id)))

    out = pd.concat(outs, ignore_index=True)

    # Upsert (ISO3:* rows)
    db.upsert_df(
        out.drop(columns=["created_at"]),
        "fact_integrity_scores",
        pk_cols=["kpi_id", "geo_id", "date"],
    )

    print(f"Wrote {len(out):,} integrity rows to fact_integrity_scores.")
    print(f"Countries scored: {out['geo_id'].nunique():,}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
