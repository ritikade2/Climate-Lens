from __future__ import annotations

import sys
from dataclasses import dataclass
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
# Config / constants (ONLY changes vs PM2.5)
# -----------------------------
GLOBAL_KPI_ID = "precip_country_monthly_sum_global_popw"
COUNTRY_KPI_ID = "precip_country_monthly_sum"
SOURCE_ID = "openmeteo"
GEO_ID = "GLOBAL"


@dataclass(frozen=True)
class Params:
    pre_window_months: int = 12
    post_window_months: int = 6
    top_n: int = 25
    method: str = "pop_weighted_country_delta"


def main() -> None:
    s = Settings.load(".env")
    db = DuckDBClient(s.duckdb_path)
    p = Params()

    # Load Latest GLOBAL drift event
    ev = db.query_df(
        f"""
        SELECT event_id, cp_start
        FROM fact_drift_events
        WHERE kpi_id='{GLOBAL_KPI_ID}' AND geo_id='{GEO_ID}'
        ORDER BY detected_at DESC
        LIMIT 1;
        """
    )
    if ev.empty:
        raise RuntimeError(
            f"No GLOBAL drift event found for {GLOBAL_KPI_ID}. "
            "Run scripts/07_detect_drift_global_popw_kpis.py first."
        )

    event_id = ev.loc[0, "event_id"]
    cp = pd.to_datetime(ev.loc[0, "cp_start"]).date()

    cp_ts = pd.Timestamp(cp)
    pre_start = (cp_ts - pd.DateOffset(months=p.pre_window_months)).to_period("M").to_timestamp()
    pre_end = (cp_ts - pd.DateOffset(months=1)).to_period("M").to_timestamp()
    post_start = cp_ts.to_period("M").to_timestamp()
    post_end = (cp_ts + pd.DateOffset(months=p.post_window_months - 1)).to_period("M").to_timestamp()

    cp_year = int(cp_ts.year)

    print(f"DuckDB path: {s.duckdb_path}")
    print(f"Event: {event_id}")
    print(
        f"Change point: {cp}  | pre: {pre_start.date()}..{pre_end.date()}  "
        f"post: {post_start.date()}..{post_end.date()}"
    )
    print(f"Population year used: {cp_year}")

    # Country KPI series: compute pre/post means, join population for cp_year
    q = f"""
    WITH k AS (
      SELECT
        REPLACE(geo_id,'ISO3:','') AS iso3,
        date,
        value
      FROM fact_kpi_timeseries
      WHERE kpi_id='{COUNTRY_KPI_ID}'
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

    # ✅ BUGFIX: pass SQL string q (not a dataframe variable)
    d = db.query_df(q)

    if d.empty:
        raise RuntimeError(
            "No country rows found for contribution analysis. "
            "Check country KPI coverage + population joins."
        )

    d["contribution_value"] = d["delta"] * d["population"]

    total = float(d["contribution_value"].sum())
    if total == 0 or (not np.isfinite(total)):
        raise RuntimeError("Total contribution_value is zero/invalid; cannot compute contribution pct.")

    d["contribution_pct"] = d["contribution_value"] / total
    d = d.sort_values("contribution_pct", ascending=False).reset_index(drop=True)

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

    # (kept identical to PM2.5 script)
    db.upsert_df(
        out.drop(columns=["created_at"]),
        "fact_contributions",
        pk_cols=["event_id", "contributor_geo_id"],
    )

    print(f"Wrote top {len(out)} contributors to fact_contributions table for event_id={event_id}")
    print(out.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
