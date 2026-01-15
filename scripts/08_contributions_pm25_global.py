"""
GLOBAL Drift Event (KPI: PM25)
- Compute country pre mean (12 m before cp)
- Compute country post mean (6 m after cp)
- Compute each country's delta (i.e. post - pre)
- Contribution Value = Population Weight (Year = CP Year) * Delta
- Contribution % = Contribution Value / sum(Contribution Value)
- Write top Contributions to fact_contributions table.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

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
GLOBAL_KPI = "pm25_monthly_mean_global_popw"
COUNTRY_KPI = "pm25_monthly_mean"
SOURCE_ID = "cams"
GLOBAL_GEO = "GLOBAL"


@dataclass(frozen=True)
class ContributionParams:
    pre_window_months: int = 12
    post_window_months: int = 6
    top_n: int = 25
    method: str = "pop_weighted_country_delta"


def main() -> None:
    s = Settings.load(".env")
    db = DuckDBClient(s.duckdb_path)
    p = ContributionParams()

    # Picking latest GLOBAL drift event. True_drift preferred.
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
        raise RuntimeError("No GLOBAL drift event found. Run scripts/07_detect_drift_pm25_global.py first.")

    event_id = ev.loc[0, "event_id"]
    cp = pd.to_datetime(ev.loc[0, "cp_start"]).date()

    cp_ts = pd.Timestamp(cp)
    pre_start = (cp_ts - pd.DateOffset(months=p.pre_window_months)).to_period("M").to_timestamp()
    pre_end = (cp_ts - pd.DateOffset(months=1)).to_period("M").to_timestamp()
    post_start = cp_ts.to_period("M").to_timestamp()
    post_end = (cp_ts + pd.DateOffset(months=p.post_window_months - 1)).to_period("M").to_timestamp()

    cp_year = int(cp_ts.year)

    print(f"Event: {event_id}")
    print(f"Change point: {cp}  | pre: {pre_start.date()}..{pre_end.date()}  post: {post_start.date()}..{post_end.date()}")
    print(f"Population year used: {cp_year}")

    # Country pre/post deltas
    q = f"""
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
    -- Defines who is eligibal
    pre AS (
      SELECT iso3, AVG(value) AS pre_mean
      FROM k
      WHERE date BETWEEN '{pre_start.date()}' AND '{pre_end.date()}'
      GROUP BY 1
    ),
    -- Defines direction & magnitude
    post AS (
      SELECT iso3, AVG(value) AS post_mean
      FROM k
      WHERE date BETWEEN '{post_start.date()}' AND '{post_end.date()}'
      GROUP BY 1
    ),
    -- Defines weight
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
    JOIN post USING (iso3)
    JOIN pop  USING (iso3)
    WHERE pre.pre_mean IS NOT NULL
      AND post.post_mean IS NOT NULL
      AND pop.population IS NOT NULL;
    """
    d = db.query_df(q)
    if d.empty:
        raise RuntimeError("No country rows found for contribution analysis. Check country KPI + population joins.")

    # pop-weighted contribution values. 
    d["contribution_value"] = d["delta"] * d["population"]
    total = float(d["contribution_value"].sum())

    if total == 0 or not np.isfinite(total):
        raise RuntimeError("Total contribution_value is zero/invalid; cannot compute contribution pct.")

    d["contribution_pct"] = d["contribution_value"] / total
    d = d.sort_values("contribution_pct", ascending=False).reset_index(drop=True)

    # writing top contributors to fact_contributions table
    top = d.head(p.top_n).copy()
    out = pd.DataFrame(
        {
            "event_id": event_id,
            "contributor_geo_id": "ISO3:" + top["iso3"].astype(str),
            "contribution_value": top["contribution_value"].astype(float),
            "contribution_pct": top["contribution_pct"].astype(float),
            "method": p.method,
            "created_at": datetime.utcnow(),
        }
    )
    # ----------------------------------
    # Write to fact_contributions
    # ----------------------------------
    db.upsert_df(
        out.drop(columns=["created_at"]),
        "fact_contributions",
        pk_cols=["event_id", "contributor_geo_id"],
    )

    print(f"Wrote top {len(out)} contributors to fact_contributions table for event_id={event_id}")
    print(out.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
