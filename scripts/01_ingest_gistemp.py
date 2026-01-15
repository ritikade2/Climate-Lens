# Ingest data from NASA GISTEMP
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# repo imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trust_bi.common.settings import Settings  
from trust_bi.ingest.gistemp import GISTEMPGlobalMonthly  
from trust_bi.warehouse.duckdb_io import DuckDBClient  


def build_dim_time(dates: pd.Series) -> pd.DataFrame:
    dt = pd.to_datetime(dates)
    out = pd.DataFrame(
        {
            "date": dt.dt.date,
            "year": dt.dt.year.astype(int),
            "month": dt.dt.month.astype(int),
            "day": dt.dt.day.astype(int),
            "month_start": pd.to_datetime(dict(year=dt.dt.year, month=dt.dt.month, day=1)).dt.date,
            "is_complete_period": True,
        }
    )
    return out.drop_duplicates(subset=["date"])


def main() -> None:
    s = Settings.load(".env")
    db = DuckDBClient(s.duckdb_path)

    ing = GISTEMPGlobalMonthly()

    gistemp_dir = s.raw_dir / "gistemp"
    gistemp_dir.mkdir(parents=True, exist_ok=True)

    # Stable cache file (so we don't re-pull every run)
    latest_path = gistemp_dir / "GLB_Ts_dSST_latest.csv"

    if not latest_path.exists():
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        stamped_path = gistemp_dir / f"GLB_Ts_dSST_{stamp}.csv"
        ing.download_raw(stamped_path)

        # overwrite/create the stable cache
        stamped_path.replace(latest_path)

        print(f"GISTEMP downloaded and cached as: {latest_path}")
    else:
        print(f"GISTEMP cache hit: {latest_path}")

    df = ing.load_recent_from_file(latest_path, start_year=2019, end_year=2025)

    fact = pd.DataFrame(
        {
            "kpi_id": "temp_anomaly_global_monthly",
            "geo_id": "GLOBAL",
            "date": df["date"],
            "source_id": "gistemp",
            "value": df["value"].astype(float),
            "value_se": None,
            "unit": "degC_anomaly",
            "n_obs": None,
            "coverage_pct": None,
            "data_quality_flag": "pass",
        }
    )

    db.upsert_df(fact, "fact_kpi_timeseries", pk_cols=["kpi_id", "geo_id", "date", "source_id"])
    db.upsert_df(build_dim_time(pd.Series(df["date"])), "dim_time", pk_cols=["date"], created_at_col=None)

    print(f"GISTEMP loaded to DuckDB: {len(fact)} rows (2019–2025)")


if __name__ == "__main__":
    main()
