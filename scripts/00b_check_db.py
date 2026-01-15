from __future__ import annotations

import sys
from pathlib import Path

# repo imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trust_bi.common.settings import Settings  # noqa: E402
from trust_bi.warehouse.duckdb_io import DuckDBClient  # noqa: E402


def main() -> None:
    s = Settings.load(".env")
    db = DuckDBClient(s.duckdb_path)

    print("\n=== dim_source ===")
    print(db.query_df("SELECT * FROM dim_source ORDER BY source_id").to_string(index=False))

    print("\n=== dim_geo ===")
    print(db.query_df("SELECT * FROM dim_geo ORDER BY geo_level, geo_id").to_string(index=False))

    print("\n=== dim_kpi ===")
    print(db.query_df("SELECT * FROM dim_kpi ORDER BY geo_level, kpi_id").to_string(index=False))

    print("\n=== dim_kpi_alias ===")
    print(db.query_df("SELECT * FROM dim_kpi_alias ORDER BY alias").to_string(index=False))


if __name__ == "__main__":
    main()

