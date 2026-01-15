from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import duckdb

# repo imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trust_bi.common.settings import Settings
from trust_bi.warehouse.duckdb_io import DuckDBClient

# -----------------------------
# Config / constants
# -----------------------------
GLOBAL_KPI = "precip_country_monthly_sum_global_popw"
COUNTRY_KPI = "precip_country_monthly_sum"
SOURCE_ID = "openmeteo"
GLOBAL_GEO = "GLOBAL"


@dataclass(frozen=True)
class Params:
    pre_window_months: int = 12
    post_window_months: int = 6
    top_n: int = 25
    min_global_grade: str = "C"  # gate threshold


GRADE_W: Dict[str, float] = {"A": 1.0, "B": 0.8, "C": 0.5, "D": 0.0}


def grade_ok(grade: Optional[str], min_grade: str) -> bool:
    order = {"A": 4, "B": 3, "C": 2, "D": 1}
    if grade is None:
        return False
    return order.get(grade, 0) >= order.get(min_grade, 0)


def main() -> None:
    s = Settings.load(".env")
    db = DuckDBClient(s.duckdb_path)
    p = Params()

    # Latest GLOBAL drift event
    ev = db.query_df(
        f"""
        SELECT event_id, cp_start, drift_type, robustness_score
        FROM fact_drift_events
        WHERE kpi_id='{GLOBAL_KPI}' AND geo_id='{GLOBAL_GEO}'
        ORDER BY detected_at DESC
        LIMIT 1;
        """
    )
    if ev.empty:
        raise RuntimeError(
            "No GLOBAL drift event found. Run scripts/07_detect_drift_global_popw_kpis.py first."
        )

    event_id = ev.loc[0, "event_id"]
    cp = pd.to_datetime(ev.loc[0, "cp_start"]).date()
    cp_ts = pd.Timestamp(cp)

    pre_start = (cp_ts - pd.DateOffset(months=p.pre_window_months)).to_period("M").to_timestamp()
    pre_end = (cp_ts - pd.DateOffset(months=1)).to_period("M").to_timestamp()
    post_start = cp_ts.to_period("M").to_timestamp()
    post_end = (cp_ts + pd.DateOffset(months=p.post_window_months - 1)).to_period("M").to_timestamp()
    cp_year = int(cp_ts.year)

    # 1) GLOBAL trust gate (hard)
    g = db.query_df(
        f"""
        SELECT confidence_grade, confidence_score, blocking_reason
        FROM fact_integrity_scores
        WHERE kpi_id='{GLOBAL_KPI}' AND geo_id='{GLOBAL_GEO}' AND date='{cp}'
        LIMIT 1;
        """
    )
    if g.empty:
        raise RuntimeError(
            "GLOBAL integrity missing at cp_start. Run scripts/07b_score_integrity_precip_global.py first."
        )

    g_grade = str(g.loc[0, "confidence_grade"])
    g_block = g.loc[0, "blocking_reason"]

    if (not grade_ok(g_grade, p.min_global_grade)) or (pd.notna(g_block) and str(g_block).strip() != ""):
        print("Contribution analysis refused due to GLOBAL integrity gate.")
        print(f"GLOBAL grade at {cp}: {g_grade} | blocking_reason: {g_block}")
        return

    print(f"GLOBAL gate passed at {cp}: grade={g_grade}")

    # 2) GLOBAL direction (sign) for contribution filtering
    q_gdelta = f"""
    WITH g AS (
      SELECT date, value
      FROM fact_kpi_timeseries
      WHERE kpi_id='{GLOBAL_KPI}' AND geo_id='{GLOBAL_GEO}' AND source_id='{SOURCE_ID}'
        AND date BETWEEN '{pre_start.date()}' AND '{post_end.date()}'
    ),
    pre AS (
      SELECT AVG(value) AS pre_mean FROM g
      WHERE date BETWEEN '{pre_start.date()}' AND '{pre_end.date()}'
    ),
    post AS (
      SELECT AVG(value) AS post_mean FROM g
      WHERE date BETWEEN '{post_start.date()}' AND '{post_end.date()}'
    )
    SELECT (post.post_mean - pre.pre_mean) AS global_delta
    FROM pre, post;
    """
    gd = db.query_df(q_gdelta)
    global_delta = float(gd.loc[0, "global_delta"])
    global_sign = 1 if global_delta >= 0 else -1
    print(f"GLOBAL delta sign={global_sign} (delta={global_delta:.4f})")

    # 3) Base contributions (country deltas * population)
    q_base = f"""
    WITH k AS (
      SELECT
        REPLACE(geo_id,'ISO3:','') AS iso3,
        date,
        value
      FROM fact_kpi_timeseries
      WHERE kpi_id='{COUNTRY_KPI}'
        AND source_id='{SOURCE_ID}'
        AND geo_id LIKE 'ISO3:%'
    ),
    pre AS (
      SELECT iso3, AVG(value) AS pre_mean
      FROM k
      WHERE date BETWEEN '{pre_start.date()}' AND '{pre_end.date()}'
      GROUP BY 1
    ),
    post AS (
      SELECT iso3, AVG(value) AS post_mean
      FROM k
      WHERE date BETWEEN '{post_start.date()}' AND '{post_end.date()}'
      GROUP BY 1
    ),
    pop AS (
      SELECT country_iso3 AS iso3, population
      FROM dim_country_population_yearly
      WHERE year = {cp_year}
    )
    SELECT
      pre.iso3,
      pre.pre_mean,
      post.post_mean,
      (post.post_mean - pre.pre_mean) AS delta,
      pop.population
    FROM pre
    JOIN post ON pre.iso3 = post.iso3
    JOIN pop  ON pre.iso3 = pop.iso3
    WHERE pre.pre_mean IS NOT NULL
      AND post.post_mean IS NOT NULL
      AND pop.population IS NOT NULL;
    """
    d = db.query_df(q_base)
    if d.empty:
        raise RuntimeError("No country rows found for contributions. Check country KPI + population joins.")

    d["contribution_value"] = d["delta"] * d["population"]

    # sign-filter: keep only countries moving in same direction as GLOBAL delta
    # treat exact 0 as global_sign
    d = d[np.sign(d["contribution_value"]).replace({0: global_sign}) == global_sign].copy()
    if d.empty:
        raise RuntimeError("After sign-filtering, no countries move in the same direction as GLOBAL delta.")

    # 4) Country integrity grades over pre+post window
    q_grades = f"""
    WITH iq AS (
      SELECT
        REPLACE(geo_id,'ISO3:','') AS iso3,
        MIN(confidence_grade) AS min_grade
      FROM fact_integrity_scores
      WHERE kpi_id='{COUNTRY_KPI}'
        AND date BETWEEN '{pre_start.date()}' AND '{post_end.date()}'
        AND geo_id LIKE 'ISO3:%'
      GROUP BY 1
    ),
    wq AS (
      SELECT
        REPLACE(geo_id,'ISO3:','') AS iso3,
        SUM(CASE WHEN data_quality_flag='warn' THEN 1 ELSE 0 END) AS warn_months
      FROM fact_kpi_timeseries
      WHERE kpi_id='{COUNTRY_KPI}'
        AND source_id='{SOURCE_ID}'
        AND date BETWEEN '{pre_start.date()}' AND '{post_end.date()}'
        AND geo_id LIKE 'ISO3:%'
      GROUP BY 1
    )
    SELECT
      iq.iso3,
      iq.min_grade,
      COALESCE(wq.warn_months, 0) AS warn_months
    FROM iq
    LEFT JOIN wq ON iq.iso3 = wq.iso3;
    """
    grades = db.query_df(q_grades)
    if grades.empty:
        raise RuntimeError("No integrity grades available for country window; refusing strict attribution.")

    d = d.merge(grades, on="iso3", how="left")

    # STRICTEST RULE (binary):
    # Any WARN month in the window => exclude completely (weight=0).
    d["warn_months"] = d["warn_months"].fillna(0).astype(int)
    d["conf_weight"] = (d["warn_months"] == 0).astype(float)

    # debug label
    d["grade_for_weight"] = np.where(d["conf_weight"] == 1.0, "A", "D")

    # Apply weight
    d["contribution_value_conf"] = d["contribution_value"] * d["conf_weight"]

    # Keep only included rows for confidence-weighted ranking
    d_conf = d[d["conf_weight"] > 0].copy()
    if d_conf.empty:
        raise RuntimeError("All countries excluded by strict rule (warn_months>0 everywhere).")

    # IMPORTANT: use ABS sums for denominators because already sign-filtered.
    total_base = float(np.abs(d["contribution_value"]).sum())
    total_conf = float(np.abs(d_conf["contribution_value_conf"]).sum())

    if not np.isfinite(total_base) or total_base == 0:
        raise RuntimeError("Base total contribution is zero/invalid after sign-filter.")
    if not np.isfinite(total_conf) or total_conf == 0:
        raise RuntimeError("Conf total contribution is zero/invalid after strict + sign-filter.")

    d["contribution_pct"] = np.abs(d["contribution_value"]) / total_base
    d_conf["contribution_pct_conf"] = np.abs(d_conf["contribution_value_conf"]) / total_conf

    top_base = d.sort_values("contribution_pct", ascending=False).head(p.top_n).copy()
    top_conf = d_conf.sort_values("contribution_pct_conf", ascending=False).head(p.top_n).copy()

    # clear previous rows for this event_id + these methods
    con = duckdb.connect(str(s.duckdb_path))
    try:
        con.execute(
            """
            DELETE FROM fact_contributions
            WHERE event_id = ?
              AND method IN ('pop_weighted_country_delta', 'pop_weighted_conf_weighted_delta');
            """,
            [event_id],
        )
    finally:
        con.close()

    out_base = pd.DataFrame(
        {
            "event_id": event_id,
            "contributor_geo_id": "ISO3:" + top_base["iso3"].astype(str),
            "contribution_value": top_base["contribution_value"].astype(float),
            "contribution_pct": top_base["contribution_pct"].astype(float),
            "method": "pop_weighted_country_delta",
            "created_at": datetime.utcnow(),
        }
    )

    out_conf = pd.DataFrame(
        {
            "event_id": event_id,
            "contributor_geo_id": "ISO3:" + top_conf["iso3"].astype(str),
            "contribution_value": top_conf["contribution_value_conf"].astype(float),
            "contribution_pct": top_conf["contribution_pct_conf"].astype(float),
            "method": "pop_weighted_conf_weighted_delta",
            "created_at": datetime.utcnow(),
        }
    )

    db.upsert_df(
        out_base.drop(columns=["created_at"]),
        "fact_contributions",
        pk_cols=["event_id", "contributor_geo_id", "method"],
    )
    db.upsert_df(
        out_conf.drop(columns=["created_at"]),
        "fact_contributions",
        pk_cols=["event_id", "contributor_geo_id", "method"],
    )

    print("Strict rule (binary): any WARN month in window => excluded (weight=0)")
    print(f"Wrote contributions for event_id={event_id}")
    print("Top 10 (base):")
    print(out_base.head(10).to_string(index=False))
    print("\nTop 10 (confidence-weighted):")
    print(out_conf.head(10).to_string(index=False))
    print(f"Base pct sum (top_n only) = {out_base['contribution_pct'].sum():.3f}")
    print(f"Conf pct sum (top_n only) = {out_conf['contribution_pct'].sum():.3f}")


if __name__ == "__main__":
    main()
