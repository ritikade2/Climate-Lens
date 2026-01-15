from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trust_bi.common.settings import Settings
from trust_bi.warehouse.duckdb_io import DuckDBClient

s = Settings.load(".env")
db = DuckDBClient(s.duckdb_path)

q1 = """
SELECT data_quality_flag, COUNT(*)
FROM fact_kpi_timeseries
WHERE kpi_id='pm25_monthly_mean' AND source_id='cams'
GROUP BY 1
ORDER BY 2 DESC;
"""
print(db.query_df(q1))

q2 = """
SELECT confidence_grade, COUNT(*)
FROM fact_integrity_scores
WHERE kpi_id='pm25_monthly_mean'
GROUP BY 1
ORDER BY 1;
"""
print(db.query_df(q2))

q3 = """
SELECT geo_id, COUNT(*) AS warn_months
FROM fact_kpi_timeseries
WHERE kpi_id='pm25_monthly_mean' AND source_id='cams' AND data_quality_flag='warn'
GROUP BY 1
ORDER BY warn_months DESC
LIMIT 20;
"""
print(db.query_df(q3))

q4 = """
SELECT confidence_grade, count(*) 
from fact_integrity_scores
where kpi_id = 'pm25_monthly_mean_global_popw' and geo_id='GLOBAL'
group by 1 
order by 1;
"""
print(db.query_df(q4))

q5 = """
WITH wq AS (
  SELECT
    REPLACE(geo_id,'ISO3:','') AS iso3,
    SUM(CASE WHEN data_quality_flag='warn' THEN 1 ELSE 0 END) AS warn_months
  FROM fact_kpi_timeseries
  WHERE kpi_id='pm25_monthly_mean'
    AND source_id='cams'
    AND date BETWEEN '2020-01-01' AND '2021-06-01'
    AND geo_id LIKE 'ISO3:%'
  GROUP BY 1
)
SELECT
  warn_months,
  COUNT(*) AS countries
FROM wq
GROUP BY 1
ORDER BY 1;
"""
print(db.query_df(q5))

q6 = """
SELECT method, COUNT(*) 
FROM fact_contributions
WHERE event_id='3d7e7000-f8ff-4a8a-816a-f5f1ce7dfa12'
GROUP BY 1;
"""
print(db.query_df(q6))

q7 = """
SELECT contributor_geo_id, contribution_pct, contribution_value
FROM fact_contributions
WHERE event_id='3d7e7000-f8ff-4a8a-816a-f5f1ce7dfa12'
  AND method='pop_weighted_conf_weighted_delta'
ORDER BY contribution_pct DESC
LIMIT 15;
"""
print(db.query_df(q7))

q8 = """
SELECT
  c.contributor_geo_id,
  MAX(CASE WHEN c.method='pop_weighted_country_delta' THEN c.contribution_pct END) AS base_pct,
  MAX(CASE WHEN c.method='pop_weighted_conf_weighted_delta' THEN c.contribution_pct END) AS conf_pct
FROM fact_contributions c
WHERE c.event_id='3d7e7000-f8ff-4a8a-816a-f5f1ce7dfa12'
GROUP BY 1
ORDER BY conf_pct DESC
LIMIT 25;
"""
print(db.query_df(q8))

q9 = """
SELECT
  method, 
  sum(contribution_pct) as contri_pct_sum, 
  count(*) as n_rows
FROm fact_contributions
where event_id = '3d7e7000-f8ff-4a8a-816a-f5f1ce7dfa12'
group by 1
order by 1;
"""
print(db.query_df(q9))

q10 = """
SELECT DISTINCT geo_id
FROM fact_kpi_timeseries
WHERE kpi_id = 'pm25_monthly_mean_global_popw';
"""
print(db.query_df(q10))

# KPIS per country
q11 = """
SELECT kpi_id, geo_id, COUNT(*) AS n
FROM fact_kpi_timeseries
GROUP BY 1,2
ORDER BY n DESC;
"""
print(db.query_df(q11))