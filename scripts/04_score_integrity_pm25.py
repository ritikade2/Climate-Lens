# scripts/04_score_integrity_pm25.py
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
KPI_ID = "pm25_monthly_mean"
SOURCE_ID = "cams"


@dataclass(frozen=True)
class IntegrityWeights:
    coverage: float = 0.40
    missingness: float = 0.25
    stability: float = 0.15
    volatility: float = 0.20


def grade(score: float) -> str:
    if score >= 0.90:
        return "A"
    if score >= 0.80:
        return "B"
    if score >= 0.70:
        return "C"
    return "D"


def blocking_reason(
    conf_grade: str,
    missing_rate_12m: float,
    stability_flag: bool,
    volatility_flag: bool,
    coverage_score: float,
) -> Optional[str]:
    reasons = []
    if conf_grade in ("C", "D"):
        reasons.append("low_confidence")
    if missing_rate_12m >= 0.20:
        reasons.append("high_missingness_12m")
    if stability_flag:
        reasons.append("coverage_instability_spike")
    if volatility_flag:
        reasons.append("volatility_spike")
    if coverage_score < 0.60:
        reasons.append("low_station_coverage_proxy")
    return ";".join(reasons) if reasons else None


def build_month_index(min_date: pd.Timestamp, max_date: pd.Timestamp) -> pd.DatetimeIndex:
    start = pd.Timestamp(min_date.year, min_date.month, 1)
    end = pd.Timestamp(max_date.year, max_date.month, 1)
    return pd.date_range(start=start, end=end, freq="MS")


def main() -> None:
    s = Settings.load(".env")
    db = DuckDBClient(s.duckdb_path)
    print(f"DuckDB path: {s.duckdb_path}")

    q = f"""
    SELECT
        kpi_id, geo_id, date, source_id, value, unit, n_obs
    FROM fact_kpi_timeseries
    WHERE kpi_id = '{KPI_ID}' AND source_id = '{SOURCE_ID}'
    ORDER BY geo_id, date;
    """
    df = db.query_df(q)
    if df.empty:
        raise RuntimeError("No PM2.5 KPI rows found. Run 03_build_kpis_pm25.py first.")

    df["date"] = pd.to_datetime(df["date"])
    df["n_obs"] = pd.to_numeric(df["n_obs"], errors="coerce")

    weights = IntegrityWeights()
    out_rows = []
    upd_rows = []

    # coverage proxy per country: max n_obs observed
    expected_nobs = df.groupby("geo_id")["n_obs"].max().rename("expected_n_obs").reset_index()
    df = df.merge(expected_nobs, on="geo_id", how="left")
    df["expected_n_obs"] = df["expected_n_obs"].replace({0: np.nan})
    df["coverage_score"] = (df["n_obs"] / df["expected_n_obs"]).clip(lower=0.0, upper=1.0)

    created_at = datetime.utcnow()

    for geo_id, g in df.groupby("geo_id", sort=False):
        g = g.sort_values("date").copy()
        full_idx = build_month_index(g["date"].min(), g["date"].max())

        g2 = g.set_index("date").reindex(full_idx)
        g2.index.name = "date"

        g2["kpi_id"] = KPI_ID
        g2["geo_id"] = geo_id

        # coverage proxy: for missing months, 0
        g2["coverage_score"] = g2["coverage_score"].fillna(0.0)

        # missingness
        g2["is_missing"] = g2["value"].isna().astype(int)
        g2["missing_rate_12m"] = g2["is_missing"].rolling(12, min_periods=1).mean()
        g2["missingness_score"] = (1.0 - g2["missing_rate_12m"]).clip(0.0, 1.0)

        # stability proxy from coverage jumps
        cov = g2["coverage_score"]
        cov_delta = cov.diff().abs()
        mu = cov_delta.rolling(12, min_periods=3).mean()
        sd = cov_delta.rolling(12, min_periods=3).std(ddof=0).replace({0: np.nan})
        z = (cov_delta - mu) / sd
        g2["coverage_delta_z"] = z.fillna(0.0)
        g2["stability_flag"] = (g2["coverage_delta_z"] >= 3.0).astype(int)
        g2["source_stability_score"] = (1.0 - 0.5 * g2["stability_flag"]).clip(0.0, 1.0)

        # volatility proxy from value jumps (stricter)
        v = g2["value"]
        dv = v.diff()
        mu_v = dv.rolling(12, min_periods=6).mean()
        sd_v = dv.rolling(12, min_periods=6).std(ddof=0).replace({0: np.nan})
        z_v = ((dv - mu_v) / sd_v).fillna(0.0)
        g2["value_delta_z"] = z_v
        z_abs = g2["value_delta_z"].abs()
        g2["volatility_flag"] = (z_abs >= 2.5).astype(int)
        g2["volatility_score"] = (
            1.0 - ((z_abs - 1.5) / (4.0 - 1.5)).clip(0.0, 1.0)
        ).clip(0.0, 1.0)

        # uncertainty_score: not available -> neutral
        g2["uncertainty_score"] = 1.0

        # confidence score
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
                stability_flag=bool(sf),
                volatility_flag=bool(vf),
                coverage_score=float(cs),
            )
            for cg, mr, sf, vf, cs in zip(
                g2["confidence_grade"],
                g2["missing_rate_12m"].fillna(1.0),
                g2["stability_flag"].fillna(0),
                g2["volatility_flag"].fillna(0),
                g2["coverage_score"].fillna(0.0),
            )
        ]

        # Integrity rows (include missing months)
        g2_reset = g2.reset_index()
        for _, row in g2_reset.iterrows():
            out_rows.append(
                {
                    "kpi_id": KPI_ID,
                    "geo_id": geo_id,
                    "date": pd.to_datetime(row["date"]).date(),
                    "coverage_score": float(row["coverage_score"]),
                    "missingness_score": float(row["missingness_score"]), 
                    "uncertainty_score": float(row["uncertainty_score"]),
                    "source_stability_score": float(row["source_stability_score"]),
                    "confidence_score": float(row["confidence_score"]),
                    "confidence_grade": row["confidence_grade"],
                    "blocking_reason": row["blocking_reason"],
                    "created_at": created_at,  
                }
            )

        # KPI updates only where KPI exists
        g_exist = g.set_index("date")
        g2_exist = g2.loc[g_exist.index]

        for dt in g2_exist.index:
            cg = g2_exist.loc[dt, "confidence_grade"]
            vs = float(g2_exist.loc[dt, "volatility_score"])
            flag = "pass" if (cg in ("A", "B") and vs >= 0.85) else "warn"

            upd_rows.append(
                {
                    "kpi_id": KPI_ID,
                    "geo_id": geo_id,
                    "date": dt.date(),
                    "source_id": SOURCE_ID,
                    "coverage_pct": float(g2_exist.loc[dt, "coverage_score"]),
                    "data_quality_flag": flag,
                }
            )

    integrity = pd.DataFrame(out_rows)
    updates = pd.DataFrame(upd_rows)

    # 1) Upsert integrity scores
    db.upsert_df(
        integrity,
        "fact_integrity_scores",
        pk_cols=["kpi_id", "geo_id", "date"],
    )

    # 2) Partial update KPI table via SQL UPDATE ... FROM
    con = duckdb.connect(str(s.duckdb_path))
    try:
        con.register("kpi_updates", updates)
        con.execute(
            """
            UPDATE fact_kpi_timeseries AS f
            SET
              coverage_pct = u.coverage_pct,
              data_quality_flag = u.data_quality_flag
            FROM kpi_updates AS u
            WHERE f.kpi_id = u.kpi_id
              AND f.geo_id = u.geo_id
              AND f.date = u.date
              AND f.source_id = u.source_id;
            """
        )
    finally:
        con.close()

    print(f"Scored integrity for {integrity['geo_id'].nunique()} countries across {integrity['date'].nunique()} months.")
    print(f"Upserted {len(integrity):,} rows into fact_integrity_scores.")
    print(f"Updated {len(updates):,} KPI rows in fact_kpi_timeseries (coverage_pct, data_quality_flag).")


if __name__ == "__main__":
    main()
