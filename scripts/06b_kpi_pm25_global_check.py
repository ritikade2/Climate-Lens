from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trust_bi.common.settings import Settings
from trust_bi.warehouse.duckdb_io import DuckDBClient

s = Settings.load(".env")
db = DuckDBClient(s.duckdb_path)

# what percent of country-month rows have population?
q1 = """
WITH k AS (
  SELECT
    REPLACE(geo_id,'ISO3:','') AS iso3,
    date,
    EXTRACT(year FROM date) AS year
  FROM fact_kpi_timeseries
  WHERE kpi_id='pm25_monthly_mean' AND source_id='cams'
),
p AS (
  SELECT country_iso3 AS iso3, year, population
  FROM dim_country_population_yearly
  WHERE source_id='worldbank'
)
SELECT
  COUNT(*) AS country_month_rows,
  SUM(CASE WHEN p.population IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_pop,
  ROUND(SUM(CASE WHEN p.population IS NOT NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS pct_rows_with_pop
FROM k
LEFT JOIN p
  ON k.iso3 = p.iso3 AND k.year = p.year;
"""
print(db.query_df(q1))

q2 = """
WITH k AS (
  SELECT DISTINCT REPLACE(geo_id,'ISO3:','') AS iso3
  FROM fact_kpi_timeseries
  WHERE kpi_id='pm25_monthly_mean' AND source_id='cams'
),
p AS (
  SELECT DISTINCT country_iso3 AS iso3
  FROM dim_country_population_yearly
  WHERE source_id='worldbank'
)
SELECT k.iso3 as kpi_ts_iso, p.iso3 as wb_iso3
FROM k
LEFT JOIN p ON k.iso3 = p.iso3
WHERE p.iso3 IS NULL
ORDER BY 1;

"""
print(db.query_df(q2))

q3 = """
SELECT year, COUNT(*) n
FROM dim_country_population_yearly
GROUP BY 1 
ORDER BY 1;
"""
print(db.query_df(q3))

q4 = """
WITH k2 AS (
  SELECT DISTINCT REPLACE(geo_id,'ISO3:','') AS iso3, EXTRACT(year FROM date) AS year
  FROM fact_kpi_timeseries
  WHERE kpi_id='pm25_monthly_mean' AND source_id='cams' AND geo_id LIKE 'ISO3:%'
),
p AS (
  SELECT DISTINCT country_iso3 AS iso3, year
  FROM dim_country_population_yearly
  WHERE source_id='worldbank'
)
SELECT k2.iso3, k2.year
FROM k2
LEFT JOIN p ON p.iso3=k2.iso3 AND p.year=k2.year
WHERE p.iso3 IS NULL
LIMIT 50;
"""
print(db.query_df(q4))

q5 = """
WITH joined AS (
  SELECT
    f.date,
    EXTRACT(year FROM f.date) AS year,
    REPLACE(f.geo_id,'ISO3:','') AS iso3,
    f.value,
    p.population
  FROM fact_kpi_timeseries f
  LEFT JOIN dim_country_population_yearly p
    ON p.country_iso3=REPLACE(f.geo_id,'ISO3:','')
   AND p.year=EXTRACT(year FROM f.date)
   AND p.source_id='worldbank'
  WHERE f.kpi_id='pm25_monthly_mean'
    AND f.source_id='cams'
    AND f.geo_id LIKE 'ISO3:%'
),
num AS (
  SELECT date, year, SUM(population) AS pop_included
  FROM joined
  WHERE population IS NOT NULL AND value IS NOT NULL
  GROUP BY 1,2
),
denom_kpi_universe AS (
  SELECT year, SUM(population) AS pop_kpi_universe
  FROM (SELECT DISTINCT iso3, year, population FROM joined WHERE population IS NOT NULL)
  GROUP BY 1
),
denom_wb AS (
  SELECT year, SUM(population) AS pop_wb_all
  FROM dim_country_population_yearly
  WHERE source_id='worldbank'
  GROUP BY 1
)
SELECT
  n.date,
  (n.pop_included/d1.pop_kpi_universe) AS coverage_vs_kpi_universe,
  (n.pop_included/d2.pop_wb_all)       AS coverage_vs_wb_all
FROM num n
JOIN denom_kpi_universe d1 USING(year)
JOIN denom_wb d2 USING(year)
ORDER BY n.date DESC
LIMIT 12;

"""
print(db.query_df(q5))

q6 = """
SELECT COUNT(DISTINCT country_iso3) AS dim_geo_countries
FROM dim_geo
WHERE geo_level='country' AND country_iso3 IS NOT NULL;
"""
print(db.query_df(q6))
