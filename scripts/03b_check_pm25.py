from __future__ import annotations
import sys
from pathlib import Path

# repo imports 
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trust_bi.common.settings import Settings
from trust_bi.warehouse.duckdb_io import DuckDBClient

def main():
    s = Settings.load(".env")
    db = DuckDBClient(s.duckdb_path)

    q1 = """
    SELECT
        kpi_id, 
        count(*) as rows,
        count(distinct geo_id) as countries,
        min(date) as min_date,
        max(date) as max_date,
    FROM fact_kpi_timeseries
    WHERE kpi_id = 'pm25_monthly_mean' AND source_id = 'cams'
    GROUP BY kpi_id;
    """
    print(db.query_df(q1).to_string(index = False))

    q2 = """
    SELECT
        geo_id, 
        Avg(value) as avg_pm25,
    FROM fact_kpi_timeseries
    WHERE kpi_id = 'pm25_monthly_mean' AND source_id = 'cams'
        and date BETWEEN '2023-01-01' and '2023-12-01'
    GROUP BY 1
    ORDER BY avg_pm25 DESC
    LIMIT 10;
    """
    print("\nTop 10 avg PM2.5 in 2023:")
    print(db.query_df(q2).to_string(index = False))

if __name__ == "__main__":
    main()