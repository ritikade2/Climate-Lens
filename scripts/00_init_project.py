from __future__ import annotations

import sys
from pathlib import Path

import duckdb

# repo imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trust_bi.common.settings import Settings


DDL = """
-- =====================
-- Dimensions
-- =====================
CREATE TABLE IF NOT EXISTS dim_time (
    date DATE PRIMARY KEY,
    year INTEGER,
    month INTEGER,
    day INTEGER,
    month_start DATE,
    is_complete_period BOOLEAN
);

CREATE TABLE IF NOT EXISTS dim_geo (
    geo_id VARCHAR PRIMARY KEY,
    geo_level VARCHAR, 
    country_iso3 VARCHAR,
    geo_name VARCHAR,
    lat DOUBLE,
    lon DOUBLE
);

CREATE TABLE IF NOT EXISTS dim_source (
    source_id VARCHAR PRIMARY KEY,
    source_name VARCHAR,
    source_version VARCHAR,
    provider VARCHAR,
    base_url VARCHAR
);

-- KPI registry (for deterministic alias resolution + LLM grounding)
CREATE TABLE IF NOT EXISTS dim_kpi (
    kpi_id VARCHAR PRIMARY KEY,
    display_name VARCHAR NOT NULL,
    description VARCHAR,
    unit VARCHAR,
    grain VARCHAR NOT NULL,      -- monthly, daily, annual
    geo_level VARCHAR NOT NULL   -- global | country
);

CREATE TABLE IF NOT EXISTS dim_kpi_alias (
    alias VARCHAR PRIMARY KEY,
    kpi_id VARCHAR NOT NULL REFERENCES dim_kpi(kpi_id)
);

-- =====================
-- Facts
-- =====================
CREATE TABLE IF NOT EXISTS fact_kpi_timeseries (
    kpi_id VARCHAR,
    geo_id VARCHAR,
    date DATE,
    source_id VARCHAR,
    value DOUBLE,
    value_se DOUBLE,
    unit VARCHAR,
    n_obs INTEGER,
    coverage_pct DOUBLE,
    data_quality_flag VARCHAR, -- pass|warn|fail
    created_at TIMESTAMP,
    PRIMARY KEY (kpi_id, geo_id, date, source_id) 
);

-- CAMS: detailed daily grid data
CREATE TABLE IF NOT EXISTS fact_grid_daily (
    grid_id VARCHAR,
    geo_id VARCHAR,
    date DATE,
    variable VARCHAR,
    value DOUBLE,
    unit VARCHAR,
    source_id VARCHAR,
    created_at TIMESTAMP,
    PRIMARY KEY (grid_id, geo_id, date, variable, source_id)
);

-- =====================
-- Trust & Integrity
-- =====================
CREATE TABLE IF NOT EXISTS fact_integrity_scores (
    kpi_id VARCHAR,
    geo_id VARCHAR,
    date DATE,
    coverage_score DOUBLE,
    missingness_score DOUBLE,
    uncertainty_score DOUBLE,
    source_stability_score DOUBLE,
    confidence_score DOUBLE,
    confidence_grade VARCHAR,
    blocking_reason VARCHAR,
    created_at TIMESTAMP,
    PRIMARY KEY (kpi_id, geo_id, date)
);

-- =====================
-- Drift registry
-- =====================
CREATE TABLE IF NOT EXISTS fact_drift_events (
    event_id VARCHAR PRIMARY KEY,
    kpi_id VARCHAR,
    geo_id VARCHAR,
    detected_at TIMESTAMP,
    cp_start DATE,
    cp_end DATE,
    effect_size DOUBLE,
    effect_size_pct DOUBLE,
    p_value DOUBLE,
    robustness_score DOUBLE,
    drift_type VARCHAR,
    artifact_evidence VARCHAR,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fact_contributions (
    event_id VARCHAR,
    contributor_geo_id VARCHAR,
    contribution_value DOUBLE,
    contribution_pct DOUBLE,
    method VARCHAR,
    created_at TIMESTAMP,
    PRIMARY KEY (event_id, contributor_geo_id, method)
);

-- =====================
-- Narrative audit
-- =====================
CREATE TABLE IF NOT EXISTS narrative_audit_log (
    request_id VARCHAR PRIMARY KEY,
    request_ts TIMESTAMP,
    user_query VARCHAR,
    resolved_kpis VARCHAR,
    allowed BOOLEAN,
    evidence_refs VARCHAR,
    refusal_reason VARCHAR,
    generated_text_hash VARCHAR
);

CREATE TABLE IF NOT EXISTS fact_narratives (
    request_id VARCHAR PRIMARY KEY,
    created_at TIMESTAMP,
    narrative_text VARCHAR
);
"""


SEED = """
-- =====================
-- GEO seed
-- =====================
INSERT INTO dim_geo (geo_id, geo_level, country_iso3, geo_name, lat, lon)
SELECT 'GLOBAL', 'global', NULL, 'Global', NULL, NULL
WHERE NOT EXISTS (SELECT 1 FROM dim_geo WHERE geo_id='GLOBAL');

-- =====================
-- Sources
-- =====================
INSERT INTO dim_source (source_id, source_name, source_version, provider, base_url)
SELECT 'gistemp', 'NASA GISTEMP', 'v4', 'NASA', 'https://data.giss.nasa.gov/gistemp/'
WHERE NOT EXISTS (SELECT 1 FROM dim_source WHERE source_id='gistemp');

INSERT INTO dim_source (source_id, source_name, source_version, provider, base_url)
SELECT 'cams', 'Copernicus CAMS', 'global_reanalysis', 'Copernicus/ECMWF', 'https://atmosphere.copernicus.eu/data'
WHERE NOT EXISTS (SELECT 1 FROM dim_source WHERE source_id='cams');

INSERT INTO dim_source (source_id, source_name, source_version, provider, base_url)
SELECT 'openmeteo_climate', 'Open-Meteo Climate API', 'v1', 'Open-Meteo', 'https://climate-api.open-meteo.com/'
WHERE NOT EXISTS (SELECT 1 FROM dim_source WHERE source_id='openmeteo_climate');

-- =====================
-- KPI seeds
-- =====================
-- Air quality (CAMS)
INSERT INTO dim_kpi (kpi_id, display_name, description, unit, grain, geo_level)
SELECT 'pm25_monthly_mean', 'PM2.5 monthly mean', 'Country-level monthly mean PM2.5 concentration', 'ug/m3', 'monthly', 'country'
WHERE NOT EXISTS (SELECT 1 FROM dim_kpi WHERE kpi_id='pm25_monthly_mean');

INSERT INTO dim_kpi (kpi_id, display_name, description, unit, grain, geo_level)
SELECT 'pm25_monthly_mean_global_popw', 'PM2.5 global pop-weighted monthly mean', 'Global population-weighted PM2.5', 'ug/m3', 'monthly', 'global'
WHERE NOT EXISTS (SELECT 1 FROM dim_kpi WHERE kpi_id='pm25_monthly_mean_global_popw');

-- Temperature (NASA GISTEMP)
INSERT INTO dim_kpi (kpi_id, display_name, description, unit, grain, geo_level)
SELECT 'temp_anomaly_global_monthly', 'Global temperature anomaly (monthly)', 'NASA GISTEMP global anomaly', 'degC_anomaly', 'monthly', 'global'
WHERE NOT EXISTS (SELECT 1 FROM dim_kpi WHERE kpi_id='temp_anomaly_global_monthly');

-- Open-Meteo climate (country-month)
INSERT INTO dim_kpi (kpi_id, display_name, description, unit, grain, geo_level)
SELECT 'temp_country_monthly_mean', 'Temperature (country, monthly)', 'Monthly mean temperature from Open-Meteo', 'degC', 'monthly', 'country'
WHERE NOT EXISTS (SELECT 1 FROM dim_kpi WHERE kpi_id='temp_country_monthly_mean');

INSERT INTO dim_kpi (kpi_id, display_name, description, unit, grain, geo_level)
SELECT 'precip_country_monthly_sum', 'Precipitation (country, monthly)', 'Monthly precipitation sum from Open-Meteo', 'mm', 'monthly', 'country'
WHERE NOT EXISTS (SELECT 1 FROM dim_kpi WHERE kpi_id='precip_country_monthly_sum');

INSERT INTO dim_kpi (kpi_id, display_name, description, unit, grain, geo_level)
SELECT 'rain_country_monthly_sum', 'Rain (country, monthly)', 'Monthly rain sum from Open-Meteo', 'mm', 'monthly', 'country'
WHERE NOT EXISTS (SELECT 1 FROM dim_kpi WHERE kpi_id='rain_country_monthly_sum');

INSERT INTO dim_kpi (kpi_id, display_name, description, unit, grain, geo_level)
SELECT 'snow_country_monthly_sum', 'Snowfall (country, monthly)', 'Monthly snowfall sum from Open-Meteo', 'cm', 'monthly', 'country'
WHERE NOT EXISTS (SELECT 1 FROM dim_kpi WHERE kpi_id='snow_country_monthly_sum');

-- =====================
-- KPI aliases (colloquial)
-- =====================
INSERT INTO dim_kpi_alias (alias, kpi_id)
SELECT 'aqi', 'pm25_monthly_mean'
WHERE NOT EXISTS (SELECT 1 FROM dim_kpi_alias WHERE alias='aqi');

INSERT INTO dim_kpi_alias (alias, kpi_id)
SELECT 'air quality', 'pm25_monthly_mean'
WHERE NOT EXISTS (SELECT 1 FROM dim_kpi_alias WHERE alias='air quality');

INSERT INTO dim_kpi_alias (alias, kpi_id)
SELECT 'pm2.5', 'pm25_monthly_mean'
WHERE NOT EXISTS (SELECT 1 FROM dim_kpi_alias WHERE alias='pm2.5');

INSERT INTO dim_kpi_alias (alias, kpi_id)
SELECT 'temperature', 'temp_country_monthly_mean'
WHERE NOT EXISTS (SELECT 1 FROM dim_kpi_alias WHERE alias='temperature');

INSERT INTO dim_kpi_alias (alias, kpi_id)
SELECT 'temp', 'temp_country_monthly_mean'
WHERE NOT EXISTS (SELECT 1 FROM dim_kpi_alias WHERE alias='temp');

INSERT INTO dim_kpi_alias (alias, kpi_id)
SELECT 'precip', 'precip_country_monthly_sum'
WHERE NOT EXISTS (SELECT 1 FROM dim_kpi_alias WHERE alias='precip');

INSERT INTO dim_kpi_alias (alias, kpi_id)
SELECT 'precipitation', 'precip_country_monthly_sum'
WHERE NOT EXISTS (SELECT 1 FROM dim_kpi_alias WHERE alias='precipitation');

INSERT INTO dim_kpi_alias (alias, kpi_id)
SELECT 'rain', 'rain_country_monthly_sum'
WHERE NOT EXISTS (SELECT 1 FROM dim_kpi_alias WHERE alias='rain');

INSERT INTO dim_kpi_alias (alias, kpi_id)
SELECT 'snow', 'snow_country_monthly_sum'
WHERE NOT EXISTS (SELECT 1 FROM dim_kpi_alias WHERE alias='snow');

INSERT INTO dim_kpi_alias (alias, kpi_id)
SELECT 'snowfall', 'snow_country_monthly_sum'
WHERE NOT EXISTS (SELECT 1 FROM dim_kpi_alias WHERE alias='snowfall');
"""


def ensure_dirs(settings: Settings) -> None:
    settings.duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    (settings.data_dir / "raw").mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "curated").mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "warehouse").mkdir(parents=True, exist_ok=True)

    # GISTEMP raw
    (settings.raw_dir / "gistemp").mkdir(parents=True, exist_ok=True)

    # CAMS raw
    (settings.raw_dir / "cams").mkdir(parents=True, exist_ok=True)
    (settings.raw_dir / "cams" / "unzipped").mkdir(parents=True, exist_ok=True)

    # Open-Meteo climate raw cache
    (settings.raw_dir / "openmeteo_climate").mkdir(parents=True, exist_ok=True)
    (settings.raw_dir / "openmeteo_climate" / "country_daily").mkdir(parents=True, exist_ok=True)
    (settings.raw_dir / "openmeteo_climate" / "country_monthly").mkdir(parents=True, exist_ok=True)


def init_db(settings: Settings) -> None:
    con = duckdb.connect(str(settings.duckdb_path))
    try:
        con.execute("PRAGMA threads=4;")
        con.execute(DDL)
        con.execute(SEED)
    finally:
        con.close()


def main() -> None:
    settings = Settings.load(".env")
    ensure_dirs(settings)
    init_db(settings)
    print(f"Initialized DuckDB at {settings.duckdb_path}")


if __name__ == "__main__":
    main()
