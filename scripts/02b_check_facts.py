# Checking TABLEs created 
from __future__ import annotations

import sys
from pathlib import Path

# repo imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trust_bi.common.settings import Settings 
from trust_bi.warehouse.duckdb_io import DuckDBClient


def main() -> None:
    s = Settings.load(".env")
    db = DuckDBClient(s.duckdb_path)

    print("\n=== FACT KPI COUNTS (by KPI, geo_level) ===")
    print(
        db.query_df(
            """
            SELECT
                k.kpi_id,
                k.geo_level,
                COUNT(f.kpi_id) AS n_rows
            FROM dim_kpi k
            LEFT JOIN fact_kpi_timeseries f
              ON k.kpi_id = f.kpi_id
            GROUP BY 1, 2
            ORDER BY 2, 1
            """
        ).to_string(index=False)
    )

    print("\n=== LATEST DATE PER KPI ===")
    print(
        db.query_df(
            """
            SELECT
                kpi_id,
                MAX(date) AS latest_date
            FROM fact_kpi_timeseries
            GROUP BY 1
            ORDER BY 1
            """
        ).to_string(index=False)
    )

    print("\n=== SAMPLE ROWS (5 per KPI) ===")
    print(
        db.query_df(
            """
            SELECT *
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (PARTITION BY kpi_id ORDER BY date DESC) AS rn
                FROM fact_kpi_timeseries
            )
            WHERE rn <= 5
            ORDER BY kpi_id, date DESC
            """
        ).to_string(index=False)
    )

    print("\n=== KPIs WITH NO FACT DATA (⚠️) ===")
    print(
        db.query_df(
            """
            SELECT k.kpi_id, k.display_name, k.geo_level
            FROM dim_kpi k
            LEFT JOIN fact_kpi_timeseries f
              ON k.kpi_id = f.kpi_id
            WHERE f.kpi_id IS NULL
            ORDER BY k.geo_level, k.kpi_id
            """
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
