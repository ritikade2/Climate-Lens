from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trust_bi.common.settings import Settings
from trust_bi.warehouse.duckdb_io import DuckDBClient

s = Settings.load(".env")
db = DuckDBClient(s.duckdb_path)

# # what percent of country-month rows have population?
# q1 = """
# SELECT
#   SUM(CASE WHEN warn_months > 0 THEN 1 ELSE 0 END) AS excluded,
#   COUNT(*) AS total
# FROM (
#   SELECT
#     REPLACE(geo_id,'ISO3:','') AS iso3,
#     SUM(CASE WHEN data_quality_flag='warn' THEN 1 ELSE 0 END) AS warn_months
#   FROM fact_kpi_timeseries
#   WHERE kpi_id='precip_country_monthly_sum'
#     AND source_id='openmeteo'
#     AND date BETWEEN '2021-11-01' AND '2023-04-01'
#     AND geo_id LIKE 'ISO3:%'
#   GROUP BY 1
# );

# """

# print(db.query_df(q1))

# q2 = """
# WITH g AS (
#   SELECT date, value
#   FROM fact_kpi_timeseries
#   WHERE kpi_id='precip_country_monthly_sum_global_popw'
#     AND geo_id='GLOBAL'
#     AND source_id='openmeteo'
#     AND date BETWEEN '2021-11-01' AND '2023-04-01'
# ),
# pre AS (
#   SELECT AVG(value) pre_mean FROM g WHERE date BETWEEN '2021-11-01' AND '2022-10-01'
# ),
# post AS (
#   SELECT AVG(value) post_mean FROM g WHERE date BETWEEN '2022-11-01' AND '2023-04-01'
# )
# SELECT post_mean - pre_mean AS global_delta FROM pre, post;


# """

# print(db.query_df(q2))

q3 = """SELECT
  kpi_id, geo_id, date,
  confidence_grade, confidence_score, blocking_reason,
  coverage_score, missingness_score, uncertainty_score, source_stability_score
FROM fact_integrity_scores
WHERE kpi_id = 'pm25_monthly_mean_global_popw'
  AND geo_id = 'GLOBAL'
  AND date = '2021-01-01';
"""
print(db.query_df(q3))