from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import duckdb
import numpy as np
import pandas as pd

# repo imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trust_bi.common.settings import Settings 
from trust_bi.warehouse.duckdb_io import DuckDBClient 


# -----------------------------
# Config
# -----------------------------
SOURCE_ID = "openmeteo"
GEO_GLOBAL = "GLOBAL"

WORLD_BANK_SOURCE_ID = "worldbank"

# Input (country-month) -> Output (global-month popw)
KPI_PAIRS: Dict[str, str] = {
    "temp_country_monthly_mean": "temp_country_monthly_mean_global_popw",
    "precip_country_monthly_sum": "precip_country_monthly_sum_global_popw",
    "rain_country_monthly_sum": "rain_country_monthly_sum_global_popw",
    "snow_country_monthly_sum": "snow_country_monthly_sum_global_popw",
}

# Units to stamp on output (kept consistent with your ingest script)
KPI_UNIT: Dict[str, str] = {
    "temp_country_monthly_mean": "degC",
    "precip_country_monthly_sum": "mm",
    "rain_country_monthly_sum": "mm",
    "snow_country_monthly_sum": "mm",
}


# -----------------------------
# DuckDB helpers
# -----------------------------
def ensure_tables_exist(duckdb_path: Path) -> None:
    """
    Ensures the required pop table exists (should already exist from Script 06).
    """
    con = duckdb.connect(str(duckdb_path))
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS dim_country_population_yearly (
              country_iso3 VARCHAR,
              year INTEGER,
              population DOUBLE,
              source_id VARCHAR,
              created_at TIMESTAMP,
              PRIMARY KEY (country_iso3, year)
            );
            """
        )
    finally:
        con.close()


def build_global_popw_for_kpi(db: DuckDBClient, *, kpi_in: str, kpi_out: str, unit: str) -> pd.DataFrame:
    """
    Pop-weighted global monthly KPI from country-month series.

    coverage_pct = sum(pop of countries included this month) / sum(pop of all WB countries in that year)
    n_obs = #countries included (value != null and pop != null)
    """
    q = f"""
    WITH k AS (
      SELECT
        geo_id,
        date,
        value
      FROM fact_kpi_timeseries
      WHERE kpi_id = '{kpi_in}'
        AND source_id = '{SOURCE_ID}'
        AND geo_id LIKE 'ISO3:%'
    ),
    k2 AS (
      SELECT
        CASE
          WHEN REPLACE(geo_id, 'ISO3:', '') = 'ROM' THEN 'ROU'
          WHEN REPLACE(geo_id, 'ISO3:', '') = 'ZAR' THEN 'COD'
          WHEN REPLACE(geo_id, 'ISO3:', '') = 'TMP' THEN 'TLS'
          ELSE REPLACE(geo_id, 'ISO3:', '')
        END AS iso3,
        date,
        EXTRACT(year FROM date) AS year,
        value
      FROM k
    ),
    p AS (
      SELECT country_iso3, year, population
      FROM dim_country_population_yearly
      WHERE source_id = '{WORLD_BANK_SOURCE_ID}'
        AND population IS NOT NULL
    ),
    joined AS (
      SELECT
        k2.date,
        k2.year,
        k2.iso3,
        k2.value,
        p.population
      FROM k2
      LEFT JOIN p
        ON p.country_iso3 = k2.iso3 AND p.year = k2.year
    ),
    denom AS (
      -- denominator is WB countries-only total population for the year (no month repetition)
      SELECT
        year,
        SUM(population) AS pop_total
      FROM p
      GROUP BY 1
    ),
    num AS (
      SELECT
        date,
        year,
        SUM(population) AS pop_included,
        COUNT(*) AS n_countries
      FROM joined
      WHERE population IS NOT NULL AND value IS NOT NULL
      GROUP BY 1,2
    )
    SELECT
      '{kpi_out}' AS kpi_id,
      '{GEO_GLOBAL}' AS geo_id,
      n.date::DATE AS date,
      '{SOURCE_ID}' AS source_id,
      SUM(j.value * j.population) / NULLIF(SUM(j.population), 0) AS value,
      NULL::DOUBLE AS value_se,
      '{unit}' AS unit,
      n.n_countries::INTEGER AS n_obs,
      (n.pop_included / NULLIF(d.pop_total, 0))::DOUBLE AS coverage_pct
    FROM joined j
    JOIN num n
      ON n.date = j.date AND n.year = j.year
    JOIN denom d
      ON d.year = j.year
    WHERE j.population IS NOT NULL AND j.value IS NOT NULL
    GROUP BY 1,2,3,4,6,7,8,9
    ORDER BY date;
    """
    g = db.query_df(q)
    if g.empty:
        raise RuntimeError(f"Global popw query returned 0 rows for kpi_in={kpi_in}.")
    return g


def upsert_global_kpi(db: DuckDBClient, g: pd.DataFrame) -> None:
    g2 = g.copy()
    g2["data_quality_flag"] = np.where(g2["coverage_pct"].fillna(0.0) >= 0.90, "pass", "warn")
    g2["created_at"] = datetime.utcnow()

    db.upsert_df(
        g2[
            [
                "kpi_id",
                "geo_id",
                "date",
                "source_id",
                "value",
                "value_se",
                "unit",
                "n_obs",
                "coverage_pct",
                "data_quality_flag",
                "created_at",
            ]
        ],
        "fact_kpi_timeseries",
        pk_cols=["kpi_id", "geo_id", "date", "source_id"],
    )


def main() -> None:
    s = Settings.load(".env")
    db = DuckDBClient(s.duckdb_path)

    print(f"DuckDB path: {s.duckdb_path}")
    ensure_tables_exist(s.duckdb_path)

    # sanity: ensure we actually have WB pop
    pop_chk = db.query_df(
        f"""
        SELECT COUNT(*) AS n
        FROM dim_country_population_yearly
        WHERE source_id='{WORLD_BANK_SOURCE_ID}';
        """
    )
    if int(pop_chk.loc[0, "n"]) == 0:
        raise RuntimeError(
            "dim_country_population_yearly has 0 WB rows. Run scripts/06_build_kpi_pm25_global.py first "
            "to populate World Bank population."
        )

    # Build globals for each Open-Meteo KPI
    all_out: List[pd.DataFrame] = []
    for kpi_in, kpi_out in KPI_PAIRS.items():
        unit = KPI_UNIT.get(kpi_in, "")
        print(f"Building global popw for {kpi_in} -> {kpi_out} (unit={unit})")

        g = build_global_popw_for_kpi(db, kpi_in=kpi_in, kpi_out=kpi_out, unit=unit)
        upsert_global_kpi(db, g)

        print(f"  wrote {len(g):,} rows. tail:")
        print(g.tail(3).to_string(index=False))
        all_out.append(g)

    # summary
    print("\nDone. Output KPI IDs:")
    for _, kpi_out in KPI_PAIRS.items():
        print(f"  - {kpi_out}")


if __name__ == "__main__":
    main()
