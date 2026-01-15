from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

# repo imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trust_bi.common.settings import Settings  # noqa: E402
from trust_bi.warehouse.duckdb_io import DuckDBClient  # noqa: E402

# -----------------------------
# Config / constants
# -----------------------------
KPI_ID = "pm25_monthly_mean_global_popw"
SOURCE_ID = "cams"
GEO_ID = "GLOBAL"

@dataclass(frozen=True)
class GlobalIntegrityWeights:
    # for GLOBAL heavily trust population coverage.
    coverage: float = 0.55
    missingness: float = 0.20
    stability: float = 0.10   # coverage jumps
    volatility: float = 0.15  # value jump spikes

def grade(score: float) -> str:
    if score >= 0.92:
        return "A"
    if score >= 0.85:
        return "B"
    if score >= 0.75:
        return "C"
    return "D"

def blocking_reason(
        conf_grade: str,
        missing_rate_12m: float,
        coverage_pct: float,
        stability_flag: bool,
        volatility_flag: bool,
    ) -> Optional[str]: 
    reasons = []
    if conf_grade in ("C", "D"):
        reasons.append("low_confidence")
    if missing_rate_12m >= 0.10:
        reasons.append("missingness_12m_high")
    if coverage_pct < 0.85:
        reasons.append("population_coverage_low")
    if stability_flag:
        reasons.append("coverage_instability_spike")
    if volatility_flag:
        reasons.append("volatility_spike")
    return ";".join(reasons) if reasons else None

def robust_std(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if x.size < 6:
        return float("nan")
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    s = 1.4826 * mad
    if s == 0:
        s = float(np.std(x, ddof = 1))
    return float(s)

def main() -> None:
    s = Settings.load(".env")
    db = DuckDBClient(s.duckdb_path)
    print(f"DuckDB path: {s.duckdb_path}")

    q = f"""
    SELECT
      date,
      value,
      coverage_pct,
      data_quality_flag
    FROM fact_kpi_timeseries
    WHERE kpi_id='{KPI_ID}'
      AND source_id='{SOURCE_ID}'
      AND geo_id='{GEO_ID}'
    ORDER BY date;
    """
    df = db.query_df(q)
    if df.empty:
        raise RuntimeError("No GLOBAL PM2.5 rows found. Run 06_build_kpi_pm25_global.py first.")


    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors = "coerce")
    df["coverage_pct"] = pd.to_numeric(df["coverage_pct"], errors = "coerce").fillna(0.0).clip(0.0, 1.0)

    # full monthly index to capture missing months
    start = pd.Timestamp(df["date"].min().year, df["date"].min().month, 1)
    end = pd.Timestamp(df["date"].max().year, df["date"].max().month, 1)
    full_idx = pd.date_range(start = start, end= end, freq = "MS")

    g2 = df.set_index("date").reindex(full_idx)
    g2.index.name = "date"


    weights = GlobalIntegrityWeights()

    # missingness
    g2["is_missing"] = g2["value"].isna().astype(int)
    g2["missing_rate_12m"] = g2["is_missing"].rolling(12, min_periods=1).mean()
    g2["missingness_score"] = (1.0 - g2["missing_rate_12m"]).clip(0.0, 1.0)

    # coverage score (already calculated; population coverage)
    g2["coverage_score"] = g2["coverage_pct"].fillna(0.0).clip(0.0, 1.0)

    # coverage stability (penalize abrupt changes in coverage score)
    cov = g2["coverage_score"]
    cov_delta = cov.diff().abs()
    mu = cov_delta.rolling(12, min_periods=3).mean()
    sd = cov_delta.rolling(12, min_periods=3).std(ddof=0).replace({0: np.nan})
    z_cov = ((cov_delta - mu) / sd).fillna(0.0)
    g2["coverage_delta_z"] = z_cov
    g2["stability_flag"] = (g2["coverage_delta_z"].abs() >= 3.0).astype(int)
    g2["source_stability_score"] = (1.0 - 0.6 * g2["stability_flag"]).clip(0.0, 1.0)

    # volatility on values (seasonality agnostic, just spikes are detected)
    dv = g2["value"].diff()
    sdv = robust_std(dv.to_numpy(dtype=float))
    if not np.isfinite(sdv) or sdv == 0:
        sdv = float(np.nanstd(dv.to_numpy(dtype=float)))

    z_v = (dv / sdv) if (np.isfinite(sdv) and sdv > 0) else dv * 0.0
    z_v = pd.Series(z_v, index=g2.index).fillna(0.0)
    g2["value_delta_z"] = z_v
    # using sensitive threshold 
    z_abs = g2["value_delta_z"].abs()
    g2["volatility_flag"] = (z_abs >= 2.5).astype(int)
    g2["volatility_score"] = (1.0 - ((z_abs - 1.5) / (4.0 - 1.5)).clip(0.0, 1.0)).clip(0.0, 1.0)

    # uncertainty_score unknown -> neutral
    g2["uncertainty_score"] = 1.0

    # final confidence
    g2["confidence_score"] = (
        weights.coverage * g2["coverage_score"]
        + weights.missingness * g2["missingness_score"]
        + weights.stability * g2["source_stability_score"]
        + weights.volatility * g2["volatility_score"]
    ).clip(0.0, 1.0)

    g2["confidence_grade"] = g2["confidence_score"].apply(grade)

    g2["blocking_reason"] = [
        blocking_reason(
            conf_grade=cg,
            missing_rate_12m=float(mr),
            coverage_pct=float(cp),
            stability_flag=bool(sf),
            volatility_flag=bool(vf),
        )
        for cg, mr, cp, sf, vf in zip(
            g2["confidence_grade"],
            g2["missing_rate_12m"].fillna(1.0),
            g2["coverage_score"].fillna(0.0),
            g2["stability_flag"].fillna(0),
            g2["volatility_flag"].fillna(0),
        )
    ]

    # Building rows for integrity table (Global only)
    out = pd.DataFrame(
        {
            "kpi_id": KPI_ID,
            "geo_id": GEO_ID,
            "date": pd.to_datetime(g2.index).date,
            "coverage_score": g2["coverage_score"].astype(float),
            "missingess_score": g2["missingness_score"].astype(float), 
            "uncertainty_score": g2["uncertainty_score"].astype(float),
            "source_stability_score": g2["source_stability_score"].astype(float),
            "confidence_score": g2["confidence_score"].astype(float),
            "confidence_grade": g2["confidence_grade"].astype(str),
            "blocking_reason": g2["blocking_reason"].astype(object),
            "created_at": datetime.utcnow(),
        }
    )
    # upsert integrity (GLOBAL)
    db.upsert_df(
        out.drop(columns=["created_at"]),
        "fact_integrity_scores",
        pk_cols=["kpi_id", "geo_id", "date"],
    )
    
    # updating KPI table quality flag for GLOBAL 
    # warn if grade below B or pop coverage < 0.90 or volatility_score low
    updates = out.copy()
    updates["source_id"] = SOURCE_ID
    updates["coverage_pct"] = out["coverage_score"].astype(float)
    updates["data_quality_flag"] = np.where(
        (updates["confidence_grade"].isin(["A", "B"])) & (updates["coverage_pct"] >= 0.90) & (g2["volatility_score"].values >= 0.85),
        "pass",
        "warn",
    )

    con = duckdb.connect(str(s.duckdb_path))
    try:
        con.register("global_updates", updates[["kpi_id", "geo_id", "date", "source_id", "coverage_pct", "data_quality_flag"]])
        con.execute(
            """
            UPDATE fact_kpi_timeseries AS f
            SET
              coverage_pct = u.coverage_pct,
              data_quality_flag = u.data_quality_flag
            FROM global_updates AS u
            WHERE f.kpi_id = u.kpi_id
              AND f.geo_id = u.geo_id
              AND f.date = u.date
              AND f.source_id = u.source_id;
            """
        )
    finally:
        con.close()

    print(f"GLOBAL ingetrity scored for {len(out):,} month.")
    print(out[["date", "coverage_score", "confidence_score", "confidence_grade"]].tail(8).to_string(index=False))

if __name__ == "__main__":
    main()