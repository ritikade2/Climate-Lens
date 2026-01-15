## Climate Lens

**Climate Lens** is a trust-aware, conversational analytics system for exploring global weather and climate KPIs.  
It combines precomputed climate metrics, statistical integrity checks, drift detection, and a conversational interface to help users interpret climate signals with transparency and reliability.

---

## Live Dashboard

**Live interactive dashboard:**  
https://ritikade2-climate-lens-scripts11-streamlit-dashboard-txjpi7.streamlit.app/

---

## What the System Does
Climate Lens enables users to:
- Explore long-term and recent trends in global climate and weather KPIs
- Visualize geographic trends through global maps
- Ask natural-language (NL) questions about climate metrics and receive grounded, evidence-backed answers
- Understand whether observed changes represent meaningful shifts or normal variability

The system is designed to surface **signals**, not just charts, and to clearly communicate **confidence and data quality** alongside insights.

---

## Data Sources

The platform integrates data from publicly available, scientific sources including:
- **Global temperature records** (e.g., NASA GISTEMP)
- **Atmospheric composition and air quality** datasets (e.g., CAMS PM2.5)
- **Weather and precipitation data** from global climate reanalysis and open meteorological APIs

All raw data ingestion and processing occurs upstream in an offline pipeline. The dashboard itself operates strictly on precomputed, curated outputs.

---

## Data Pipeline Overview

The data pipeline follows a deterministic, reproducible flow:
1. **Ingestion** - Raw climate and weather datasets are ingested from trusted public sources.
2. **KPI Construction** - Raw measurements are transformed into standardized KPIs at consistent spatial and temporal resolutions.
3. **Integrity Scoring** - Each KPI is evaluated for data quality using coverage, consistency, and volatility checks.
4. **Drift Detection** - Statistical methods are applied to identify meaningful shifts in KPI behavior over time.
5. **Materialization** - Final KPI time series, integrity scores, drift signals, and contribution summaries are stored in a DuckDB warehouse and consumed directly by the dashboard.

---

## Key KPIs

Example KPIs supported by Climate Lens include:
- Population-weighted **PM2.5 concentration**
- **Global and regional temperature anomalies**
- **Precipitation and weather aggregates** across countries and regions

KPIs are designed to be comparable across geographies and time periods.

---

## Drift Detection Logic

Drift represents a **statistically meaningful change** in the behavior of a KPI over time.  
Climate Lens detects drift by comparing recent distributions and trends against historical baselines, accounting for normal variability and seasonality.

Detected drift signals indicate **structural change**, not short-term noise.

---

## Integrity and Trust Signals

Each KPI is accompanied by an **integrity score**, which summarizes:
- Data completeness and coverage
- Stability and noise levels
- Sensitivity to window or boundary changes

These signals help users understand how reliable a given trend or comparison is before drawing conclusions.

---

## Conversational and Deterministic Reasoning

Climate Lens supports natural-language queries such as:
> “How has PM2.5 changed in US since 2018?”

The system responds using a two-layer approach:
- **LLM-first conversational reasoning** for explanation and synthesis
- **Deterministic fallback logic** when language models are unavailable or when data constraints limit interpretability

In all cases, responses are grounded in precomputed metrics and explicit evidence.


