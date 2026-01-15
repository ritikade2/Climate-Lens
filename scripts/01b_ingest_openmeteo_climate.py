"""
Ingest Open-Meteo Climate API daily data:
- Read data at country centroids (lat + long)
- Aggregate to country-month KPIs,
- Upsert into DuckDB fact_kpi_timeseries table

Writes these KPI IDs (country-month):
- temp_country_monthly_mean        unit: degC
- precip_country_monthly_sum       unit: mm
- rain_country_monthly_sum         unit: mm
- snow_country_monthly_sum         unit: mm

Source: openmeteo (climate-api.open-meteo.com)
Model: EC_Earth3P_HR (configurable)
Period: 2019-01-01 to 2024-12-31 by default (configurable)
"""

from __future__ import annotations

import sys
import time
import random
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests

# repo imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trust_bi.common.settings import Settings  # noqa: E402
from trust_bi.warehouse.duckdb_io import DuckDBClient  # noqa: E402

# -----------------------------
# Config
# -----------------------------
OPENMETEO_BASE = "https://climate-api.open-meteo.com/v1/climate"
SOURCE_ID = "openmeteo"
MODEL = "EC_Earth3P_HR"

# default ingestion range (inclusive)
DEFAULT_START = "2019-01-01"
DEFAULT_END = "2024-12-31"

# requested daily variables
DAILY_VARS = [
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
]

# KPI mapping (daily var -> (kpi_id, agg, unit))
KPI_MAP = {
    "temperature_2m_mean": ("temp_country_monthly_mean", "mean", "degC"),
    "precipitation_sum": ("precip_country_monthly_sum", "sum", "mm"),
    "rain_sum": ("rain_country_monthly_sum", "sum", "mm"),
    "snowfall_sum": ("snow_country_monthly_sum", "sum", "mm"),
}

# throttling / batching. For full-universe ingestion, letting script iterate over all countries.
SLEEP_BETWEEN_COUNTRIES_S = 2.0  # if still hitting 429, bump to ~2.0-3.0

# Optional batching: run only a slice of countries to avoid rate limits
# Set END_IDX=None to run all (not recommended on first try).
START_IDX = 0
END_IDX: Optional[int] = None  # run 30 at a time


# -----------------------------
# Natural Earth centroids
# -----------------------------
def load_world_countries(repo_root: Path):
    """
    Loads Natural Earth Admin-0 country polygons (110m) from local disk.
    Auto-downloads the zip once if missing.
    """
    import geopandas as gpd
    import subprocess

    ref_dir = repo_root / "data" / "reference" / "natural_earth"
    ref_dir.mkdir(parents=True, exist_ok=True)

    zip_path = ref_dir / "ne_110m_admin_0_countries.zip"
    shp_inside_zip = "ne_110m_admin_0_countries.shp"

    if not zip_path.exists():
        url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
        subprocess.run(
            ["curl", "-L", "--fail", "--silent", "--show-error", "-o", str(zip_path), url],
            check=True,
        )

    world = gpd.read_file(f"zip://{zip_path}!{shp_inside_zip}")
    world = world.rename(columns={"ISO_A3": "country_iso3", "NAME": "geo_name"})
    world = world[world["country_iso3"].notna() & (world["country_iso3"] != "-99")].copy()
    world = world[["country_iso3", "geo_name", "geometry"]].to_crs(4326)
    return world


def country_centroids_iso3(repo_root: Path) -> pd.DataFrame:
    """
    Returns DataFrame: country_iso3, lat, lon
    Uses representative_point() to ensure point lies within polygon.
    """
    try:
        import geopandas as gpd  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Missing geopandas stack required for centroids.\n"
            "Install: python3 -m pip install geopandas shapely pyproj fiona rtree\n"
            f"Original error: {repr(e)}"
        )

    world = load_world_countries(repo_root)

    # representative_point() avoids centroids falling in ocean for weird shapes
    pts = world.geometry.representative_point()
    out = pd.DataFrame(
        {
            "country_iso3": world["country_iso3"].astype(str).str.upper(),
            "lat": pts.y.astype(float),
            "lon": pts.x.astype(float),
        }
    )

    out = out.dropna().drop_duplicates(subset=["country_iso3"])
    return out


# -----------------------------
# HTTP helpers (429-safe)
# -----------------------------
def _get_json(url: str, *, timeout: int = 60, retries: int = 10, backoff: float = 1.8):
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)

            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                if ra:
                    sleep_s = float(ra)
                else:
                    sleep_s = min(120.0, (backoff**attempt)) + random.uniform(0.0, 0.8)
                time.sleep(sleep_s)
                continue

            r.raise_for_status()
            return r.json()

        except Exception as e:
            last_err = e
            sleep_s = min(60.0, (backoff**attempt)) + random.uniform(0.0, 0.8)
            time.sleep(sleep_s)

    raise RuntimeError(f"Open-Meteo request failed after {retries} retries. Last error: {repr(last_err)}")


def build_openmeteo_url(
    *, lat: float, lon: float, start: str, end: str, model: str, daily_vars: List[str]
) -> str:
    daily_str = ",".join(daily_vars)
    return (
        f"{OPENMETEO_BASE}"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        f"&models={model}"
        f"&daily={daily_str}"
        f"&timezone=UTC"
    )


def fetch_country_daily(*, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    url = build_openmeteo_url(lat=lat, lon=lon, start=start, end=end, model=MODEL, daily_vars=DAILY_VARS)
    payload = _get_json(url)

    daily = payload.get("daily") or {}
    dates = daily.get("time") or []
    if not dates:
        return pd.DataFrame()

    df = pd.DataFrame({"date": pd.to_datetime(dates)})
    for v in DAILY_VARS:
        arr = daily.get(v)
        df[v] = pd.to_numeric(arr, errors="coerce") if arr is not None else np.nan

    return df


# -----------------------------
# Transform: daily -> monthly KPIs
# -----------------------------
def daily_to_monthly_kpis(df_daily: pd.DataFrame) -> pd.DataFrame:
    if df_daily.empty:
        return df_daily

    d = df_daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["month"] = d["date"].dt.to_period("M").dt.to_timestamp()

    rows = []
    for var, (kpi_id, agg, unit) in KPI_MAP.items():
        if var not in d.columns:
            continue

        g = d.groupby("month")[var]

        if agg == "sum":
            val = g.sum(min_count=1)
        elif agg == "mean":
            val = g.mean()
        else:
            raise ValueError(f"Unknown agg={agg} for var={var}")

        # n_obs = number of non-null daily points used that month
        n_obs = g.count()

        tmp = pd.DataFrame(
            {
                "kpi_id": kpi_id,
                "date": val.index.astype("datetime64[ns]").date,
                "value": val.values.astype(float),
                "n_obs": n_obs.values.astype(int),
                "unit": unit,
            }
        )
        rows.append(tmp)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out


def month_days_counts(start: str, end: str) -> pd.DataFrame:
    """
    Returns expected number of days per month in [start,end] range.
    """
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    all_days = pd.date_range(s, e, freq="D")
    m = all_days.to_series().dt.to_period("M").value_counts().sort_index()
    return pd.DataFrame({"month": m.index.to_timestamp(), "days_in_month": m.values})


# -----------------------------
# DuckDB: ensure dim_source row
# -----------------------------
def ensure_openmeteo_source(duckdb_path: Path) -> None:
    import duckdb

    con = duckdb.connect(str(duckdb_path))
    try:
        con.execute(
            f"""
            INSERT INTO dim_source (source_id, source_name, source_version, provider, base_url)
            SELECT '{SOURCE_ID}', 'Open-Meteo Climate API', 'v1', 'Open-Meteo', '{OPENMETEO_BASE}'
            WHERE NOT EXISTS (
              SELECT 1 FROM dim_source WHERE source_id = '{SOURCE_ID}'
            );
            """
        )
    finally:
        con.close()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    s = Settings.load(".env")
    db = DuckDBClient(s.duckdb_path)

    start = DEFAULT_START
    end = DEFAULT_END

    print(f"DuckDB path: {s.duckdb_path}")
    print(f"Open-Meteo model: {MODEL}")
    print(f"Range: {start} to {end}")

    ensure_openmeteo_source(s.duckdb_path)

    # Get country universe from your warehouse (dim_geo)
    geos = db.query_df(
        """
        SELECT DISTINCT country_iso3 AS iso3
        FROM dim_geo
        WHERE geo_level='country' AND country_iso3 IS NOT NULL
        ORDER BY 1;
        """
    )
    if geos.empty:
        raise RuntimeError("dim_geo has no country_iso3 rows. Run Script 03 first (it inserts dim_geo rows).")

    iso3_universe = geos["iso3"].astype(str).str.upper().tolist()

    # Compute centroids from Natural Earth
    cent = country_centroids_iso3(REPO_ROOT)
    cent = cent[cent["country_iso3"].isin(iso3_universe)].copy()

    if cent.empty:
        raise RuntimeError("No centroids matched your dim_geo country ISO3 universe.")

    cent = cent.sort_values("country_iso3").reset_index(drop=True)

    # Optional batching
    if END_IDX is not None:
        cent = cent.iloc[START_IDX:END_IDX].copy()

    print(f"Countries to fetch: {len(cent)}")

    cache_dir = s.raw_dir / "openmeteo"
    cache_dir.mkdir(parents=True, exist_ok=True)

    expected_days = month_days_counts(start, end)
    expected_days["month"] = pd.to_datetime(expected_days["month"])

    all_fact_rows = []

 
    for idx, r in enumerate(cent.itertuples(index=False), start=1):
        iso3 = r.country_iso3
        lat = float(r.lat)
        lon = float(r.lon)

        cache_csv = cache_dir / f"openmeteo_daily_{MODEL}_{iso3}_{start}_{end}.csv"

        if cache_csv.exists():
            df_daily = pd.read_csv(cache_csv, parse_dates=["date"])
        else:
            df_daily = fetch_country_daily(lat=lat, lon=lon, start=start, end=end)
            if df_daily.empty:
                print(f"[WARN] No daily data returned for {iso3}. Skipping.")
                time.sleep(SLEEP_BETWEEN_COUNTRIES_S)
                continue
            df_daily.to_csv(cache_csv, index=False)

        df_m = daily_to_monthly_kpis(df_daily)
        if df_m.empty:
            print(f"[WARN] No monthly rows produced for {iso3}. Skipping.")
            time.sleep(SLEEP_BETWEEN_COUNTRIES_S)
            continue

        # add geo/source + coverage_pct + data_quality_flag
        df_m["geo_id"] = "ISO3:" + iso3
        df_m["source_id"] = SOURCE_ID
        df_m["value_se"] = None

        df_m["date"] = pd.to_datetime(df_m["date"])
        df_m = df_m.merge(expected_days, left_on="date", right_on="month", how="left")

        df_m["coverage_pct"] = np.where(
            df_m["days_in_month"].notna() & (df_m["days_in_month"] > 0),
            df_m["n_obs"].astype(float) / df_m["days_in_month"].astype(float),
            np.nan,
        )
        df_m["coverage_pct"] = pd.to_numeric(df_m["coverage_pct"], errors="coerce")
        df_m["data_quality_flag"] = np.where(df_m["coverage_pct"].ge(0.90), "pass", "warn").astype(str)

        df_m["created_at"] = datetime.utcnow()

        out_cols = [
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
        all_fact_rows.append(df_m[out_cols])

        time.sleep(SLEEP_BETWEEN_COUNTRIES_S)

        if idx % 10 == 0:
            print(f"Fetched+processed {idx}/{len(cent)} countries...")


    if not all_fact_rows:
        print("No KPI rows produced. Nothing to upsert.")
        return

    fact = pd.concat(all_fact_rows, ignore_index=True)

    # Ensure month-start date (defensive)
    fact["date"] = pd.to_datetime(fact["date"]).dt.date
    fact["date"] = [d.replace(day=1) for d in fact["date"]]

    # Upsert to fact_kpi_timeseries
    db.upsert_df(
        fact,
        "fact_kpi_timeseries",
        pk_cols=["kpi_id", "geo_id", "date", "source_id"],
    )

    print(f"Upserted {len(fact):,} rows into fact_kpi_timeseries (Open-Meteo climate KPIs).")
    print(fact.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
