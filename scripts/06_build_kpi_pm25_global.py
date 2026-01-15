# scripts/06_build_kpi_pm25_global.py
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import duckdb
import numpy as np
import pandas as pd
import requests

# repo imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trust_bi.common.settings import Settings  # noqa: E402
from trust_bi.warehouse.duckdb_io import DuckDBClient  # noqa: E402

# -----------------------------
# Config / constants
# -----------------------------
KPI_IN = "pm25_monthly_mean"
SOURCE_ID = "cams"

# Output KPI (global-month, pop-weighted)
KPI_OUT = "pm25_monthly_mean_global_popw"
GEO_GLOBAL = "GLOBAL"

WORLD_BANK_SOURCE_ID = "worldbank"
WORLD_BANK_BASE = "https://api.worldbank.org/v2"
WORLD_BANK_COUNTRY_INDICATOR_PATH = "country/all/indicator"
WORLD_BANK_COUNTRY_META_PATH = "country"
WB_INDICATOR_POP = "SP.POP.TOTL"


# -----------------------------
# HTTP helpers
# -----------------------------
def _wb_get_json(url: str, *, timeout: int = 60, retries: int = 4, backoff: float = 1.5):
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(backoff**attempt)
    raise RuntimeError(f"World Bank API failed after {retries} retries. Last error: {repr(last_err)}")


# -----------------------------
# WB: allowlist (countries only)
# -----------------------------
def wb_fetch_country_iso3_set() -> set[str]:
    """
    Returns ISO3 codes for *countries only* (excludes aggregates) from WB /country metadata.

    WB /country JSON returns [meta, rows] where each row has:
      - id: ISO3 code
      - region.value: "Aggregates" for aggregate entities
    """
    iso3s: set[str] = set()
    page = 1
    while True:
        url = (
            f"{WORLD_BANK_BASE}/{WORLD_BANK_COUNTRY_META_PATH}"
            f"?format=json&per_page=400&page={page}"
        )
        payload = _wb_get_json(url)

        if not isinstance(payload, list) or len(payload) < 2:
            raise RuntimeError(f"Unexpected WB /country payload: {payload!r}")

        meta = payload[0] or {}
        rows = payload[1] or []

        for r in rows:
            if not r:
                continue
            region_val = (r.get("region") or {}).get("value")
            if region_val == "Aggregates":
                continue

            iso3 = (r.get("id") or "").strip().upper()
            if len(iso3) == 3:
                iso3s.add(iso3)

        pages = int(meta.get("pages", 1))
        if page >= pages:
            break
        page += 1

    if not iso3s:
        raise RuntimeError("WB country ISO3 allowlist is empty. WB /country parsing failed.")

    return iso3s


# -----------------------------
# WB: population fetch + raw cache
# -----------------------------
def wb_fetch_population_all(years: List[int]) -> pd.DataFrame:
    """
    Fetch total population (SP.POP.TOTL) for all countries/aggregates for given years.
    We'll filter to countries-only using wb_fetch_country_iso3_set().
    """
    out = []
    for y in years:
        page = 1
        while True:
            url = (
                f"{WORLD_BANK_BASE}/{WORLD_BANK_COUNTRY_INDICATOR_PATH}/{WB_INDICATOR_POP}"
                f"?format=json&per_page=20000&page={page}&date={y}"
            )

            payload = _wb_get_json(url)

            if not isinstance(payload, list) or len(payload) < 2:
                raise RuntimeError(f"Unexpected WB indicator payload for year={y}: {payload!r}")

            meta = payload[0] or {}
            rows = payload[1] or []

            for row in rows:
                if not row:
                    continue

                iso3 = (row.get("countryiso3code") or "").strip().upper()
                val = row.get("value")
                dt = row.get("date")

                if len(iso3) != 3 or val is None or dt is None:
                    continue

                try:
                    out.append(
                        {
                            "country_iso3": iso3,
                            "year": int(dt),
                            "population": float(val),
                        }
                    )
                except Exception:
                    continue

            pages = int(meta.get("pages", 1))
            if page >= pages:
                break
            page += 1

    df = pd.DataFrame(out).dropna()
    if df.empty:
        return df

    df["country_iso3"] = df["country_iso3"].astype(str).str.strip().str.upper()
    df["year"] = df["year"].astype(int)
    df["population"] = df["population"].astype(float)

    df = df.drop_duplicates(subset=["country_iso3", "year"], keep="last")
    return df


def cache_population_raw(raw_dir: Path, years: List[int]) -> Tuple[pd.DataFrame, Path]:
    """
    Cache WB population as CSV under data/raw/worldbank/.
    Filters out WB aggregates so pop_all isn't inflated.
    """
    wb_dir = raw_dir / "worldbank"
    wb_dir.mkdir(parents=True, exist_ok=True)

    y0, y1 = min(years), max(years)
    cache_path = wb_dir / f"population_{WB_INDICATOR_POP}_{y0}_{y1}.csv"

    valid_iso3 = wb_fetch_country_iso3_set()

    if cache_path.exists():
        pop = pd.read_csv(cache_path)
        needed = {"country_iso3", "year", "population"}
        if not needed.issubset(set(pop.columns)):
            raise RuntimeError(f"Bad cache file schema at {cache_path}. Columns={list(pop.columns)}")

        pop["country_iso3"] = pop["country_iso3"].astype(str).str.strip().str.upper()
        pop["year"] = pop["year"].astype(int)
        pop["population"] = pop["population"].astype(float)

        # enforce countries-only even if cache is stale
        pop = pop[pop["country_iso3"].isin(valid_iso3)].copy()
        return pop, cache_path

    pop = wb_fetch_population_all(years)
    if pop.empty:
        raise RuntimeError("WB population fetch returned 0 rows.")

    pop["country_iso3"] = pop["country_iso3"].astype(str).str.strip().str.upper()
    pop = pop[pop["country_iso3"].isin(valid_iso3)].copy()

    if pop.empty:
        raise RuntimeError("After filtering aggregates, WB population is empty. Check WB /country parsing.")

    pop.to_csv(cache_path, index=False)
    return pop, cache_path


# -----------------------------
# DuckDB tables / upserts
# -----------------------------
def ensure_pop_table(duckdb_path: Path) -> None:
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


def ensure_worldbank_source(duckdb_path: Path) -> None:
    con = duckdb.connect(str(duckdb_path))
    try:
        con.execute(
            f"""
            INSERT INTO dim_source (source_id, source_name, source_version, provider, base_url)
            SELECT '{WORLD_BANK_SOURCE_ID}', 'World Bank Indicators', 'v2', 'World Bank', '{WORLD_BANK_BASE}'
            WHERE NOT EXISTS (
              SELECT 1 FROM dim_source WHERE source_id = '{WORLD_BANK_SOURCE_ID}'
            );
            """
        )
    finally:
        con.close()


def upsert_population(db: DuckDBClient, pop: pd.DataFrame, duckdb_path: Path) -> None:
    """
    Delete-and-replace WB pop rows for the year range to avoid aggregate contamination
    from earlier runs.
    """
    pop2 = pop.copy()
    pop2["source_id"] = WORLD_BANK_SOURCE_ID
    pop2["created_at"] = datetime.utcnow()

    y_min = int(pop2["year"].min())
    y_max = int(pop2["year"].max())

    con = duckdb.connect(str(duckdb_path))
    try:
        con.execute(
            f"""
            DELETE FROM dim_country_population_yearly
            WHERE source_id = '{WORLD_BANK_SOURCE_ID}'
              AND year BETWEEN {y_min} AND {y_max};
            """
        )
    finally:
        con.close()

    db.upsert_df(
        pop2[["country_iso3", "year", "population", "source_id", "created_at"]],
        "dim_country_population_yearly",
        pk_cols=["country_iso3", "year"],
    )


# -----------------------------
# Build global KPI
# -----------------------------
def build_global_popw_kpi(db: DuckDBClient) -> pd.DataFrame:
    """
    Pop-weighted global monthly PM2.5.
    coverage_pct = pop included that month / pop total across ISO3 universe (WB countries-only) for that year.
    """
    q = f"""
    WITH k AS (
      SELECT
        geo_id,
        date,
        value
      FROM fact_kpi_timeseries
      WHERE kpi_id = '{KPI_IN}' AND source_id = '{SOURCE_ID}'
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
    k2_country_only AS (
        SELECT k2.*
        FROM k2
        JOIN (SELECT DISTINCT country_iso3 FROM dim_country_population_yearly WHERE source_id='worldbank') wb
            ON wb.country_iso3 = k2.iso3
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
      FROM k2_country_only as k2
      LEFT JOIN p
        ON p.country_iso3 = k2.iso3 AND p.year = k2.year
    ),
    denom AS (
      -- denominator per year should be WB pop for countries-only, no month repetition
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
      '{KPI_OUT}' AS kpi_id,
      '{GEO_GLOBAL}' AS geo_id,
      n.date::DATE AS date,
      '{SOURCE_ID}' AS source_id,
      SUM(j.value * j.population) / SUM(j.population) AS value,
      NULL::DOUBLE AS value_se,
      'ug/m3' AS unit,
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
        raise RuntimeError("Global KPI query returned no rows. Check population + country KPI tables.")
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


# -----------------------------
# main
# -----------------------------
def main() -> None:
    s = Settings.load(".env")
    db = DuckDBClient(s.duckdb_path)

    print(f"DuckDB path: {s.duckdb_path}")

    yr = db.query_df(
        f"""
        SELECT MIN(EXTRACT(year FROM date)) AS min_y, MAX(EXTRACT(year FROM date)) AS max_y
        FROM fact_kpi_timeseries
        WHERE kpi_id='{KPI_IN}' AND source_id='{SOURCE_ID}';
        """
    )
    min_y = int(yr.loc[0, "min_y"])
    max_y = int(yr.loc[0, "max_y"])
    years = list(range(min_y, max_y + 1))
    print(f"Building population table for years: {years}")

    ensure_worldbank_source(s.duckdb_path)
    ensure_pop_table(s.duckdb_path)

    # cache under data/raw/worldbank/ (countries-only)
    pop, cache_path = cache_population_raw(s.raw_dir, years)
    print(f"Population cache: {cache_path}")
    print(f"Population rows (unique country-year): {len(pop):,}")

    # delete+replace WB rows for the year range
    upsert_population(db, pop, s.duckdb_path)

    # build global KPI
    g = build_global_popw_kpi(db)
    upsert_global_kpi(db, g)

    print(f"Wrote global pop-weighted PM2.5 KPI: {len(g):,} rows into fact_kpi_timeseries")
    print(g.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
