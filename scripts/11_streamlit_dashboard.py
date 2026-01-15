"""
Trust-Aware Conversational BI (Streamlit) — Script 11

What this app does
- Reads precomputed pipeline tables from DuckDB:
  - fact_kpi_timeseries
  - fact_drift_events
  - fact_integrity_scores
  - fact_contributions

Key UX requirements implemented
- Single-page Dashboard UI.
- Explore filters are optional and do NOT constrain Q&A.
- Q&A is LLM-first (OpenAI via OPENAI_API_KEY loaded from .env). If missing/errors -> deterministic fallback.
- Trend chart + Map (Plotly choropleth if plotly installed, else fallback table + warning).
- 1-line definitions for KPIs + trust/drift/integrity.

Run
  streamlit run ./scripts/11_streamlit_app.py
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# DuckDB dependency (required for this app)
import duckdb


# -----------------------------
# Settings / DB access
# -----------------------------

@dataclass(frozen=True)
class Settings:
    duckdb_path: str

    @staticmethod
    def load(dotenv_path: str = ".env") -> "Settings":
        # Explicitly load .env for Streamlit runs
        load_dotenv(dotenv_path)
        # Also respect existing environment variables
        db_path = os.getenv("DUCKDB_PATH") or os.getenv("TACBI_DUCKDB_PATH") or os.getenv("DUCKDB_FILE")
        if not db_path:
            # common repo default
            candidate = Path("./data/pipeline.duckdb")
            db_path = str(candidate) if candidate.exists() else ""
        return Settings(duckdb_path=db_path)


class DuckDBClient:
    def __init__(self, path: str):
        self.path = path
        if not path:
            raise ValueError("DuckDB path is empty. Set DUCKDB_PATH in .env.")
        if not Path(path).exists():
            raise FileNotFoundError(f"DuckDB file not found: {path}")

    def query_df(self, sql: str, params: Optional[Tuple[Any, ...]] = None) -> pd.DataFrame:
        con = duckdb.connect(self.path, read_only=True)
        try:
            if params:
                return con.execute(sql, params).df()
            return con.execute(sql).df()
        finally:
            con.close()

    def table_columns(self, table_name: str) -> List[str]:
        df = self.query_df(f"DESCRIBE {table_name};")
        if df.empty or "column_name" not in df.columns:
            return []
        return df["column_name"].tolist()

    def table_exists(self, table_name: str) -> bool:
        df = self.query_df(
            "SELECT COUNT(*) AS n FROM information_schema.tables WHERE table_name = ?;",
            (table_name,),
        )
        return int(df["n"].iloc[0]) > 0


REQUIRED_TABLES = [
    "fact_kpi_timeseries",
    "fact_drift_events",
    "fact_integrity_scores",
    "fact_contributions",
]


# -----------------------------
# Definitions (1 line each)
# -----------------------------

TRUST_DEFINITIONS = {
    "trust": "Trust = how reliable the answer is given coverage, freshness, and stability signals.",
    "drift": "Drift = a statistically meaningful shift in a KPI’s level/trend vs its prior baseline.",
    "integrity_score": "Integrity score = a quality/reliability score for the KPI signal (higher is safer).",
    "contributions": "Contributions = top factors/segments associated with the detected drift (if available).",
}

# You can extend this; app will also show “(no definition yet)” for KPIs not listed.
KPI_DEFINITIONS: Dict[str, str] = {
    "pm25_monthly_mean": "PM2.5 monthly mean concentration (higher = worse air quality).",
    "precip_country_monthly_sum": "Monthly precipitation total for a country (sum over month).",
    "temp_country_monthly_mean": "Monthly mean temperature for a country.",
}


# -----------------------------
# Schema helpers
# -----------------------------

def _first_existing(cols: List[str], candidates: List[str]) -> Optional[str]:
    s = set(c.lower() for c in cols)
    for c in candidates:
        if c.lower() in s:
            # return the actual column casing from cols
            for real in cols:
                if real.lower() == c.lower():
                    return real
    return None


def _norm_geo(geo_id: str) -> str:
    # Accept: "ISO3:IND" -> "IND", "IND" -> "IND", "GLOBAL" -> "GLOBAL"
    if not geo_id:
        return geo_id
    if geo_id.upper() == "GLOBAL":
        return "GLOBAL"
    if ":" in geo_id:
        return geo_id.split(":", 1)[1].upper()
    return geo_id.upper()


def _discover_timeseries_cols(db: DuckDBClient) -> Dict[str, str]:
    cols = db.table_columns("fact_kpi_timeseries")
    kpi = _first_existing(cols, ["kpi_id", "kpi", "metric", "kpi_name"]) or "kpi_id"
    geo = _first_existing(cols, ["geo_id", "geo", "country", "iso3"]) or "geo_id"
    src = _first_existing(cols, ["source_id", "source"])  # optional
    date = _first_existing(cols, ["date", "ds", "dt", "timestamp", "day", "month"]) or "date"
    val = _first_existing(cols, ["value", "kpi_value", "val", "measurement"]) or "value"
    return {"kpi": kpi, "geo": geo, "src": src or "", "date": date, "val": val}


@st.cache_data(show_spinner=False)
def list_kpis(db_path: str) -> List[str]:
    db = DuckDBClient(db_path)
    cols = _discover_timeseries_cols(db)
    df = db.query_df(f"SELECT DISTINCT {cols['kpi']} AS kpi_id FROM fact_kpi_timeseries ORDER BY 1;")
    return df["kpi_id"].dropna().astype(str).tolist()


@st.cache_data(show_spinner=False)
def list_geos(db_path: str, kpi_id: Optional[str] = None) -> List[str]:
    db = DuckDBClient(db_path)
    cols = _discover_timeseries_cols(db)
    if kpi_id:
        df = db.query_df(
            f"SELECT DISTINCT {cols['geo']} AS geo_id FROM fact_kpi_timeseries WHERE {cols['kpi']} = ? ORDER BY 1;",
            (kpi_id,),
        )
    else:
        df = db.query_df(f"SELECT DISTINCT {cols['geo']} AS geo_id FROM fact_kpi_timeseries ORDER BY 1;")
    return df["geo_id"].dropna().astype(str).tolist()


@st.cache_data(show_spinner=False)
def list_years(db_path: str, kpi_id: Optional[str] = None) -> List[int]:
    db = DuckDBClient(db_path)
    cols = _discover_timeseries_cols(db)
    where = ""
    params: Tuple[Any, ...] = ()
    if kpi_id:
        where = f"WHERE {cols['kpi']} = ?"
        params = (kpi_id,)
    df = db.query_df(
        f"""
        SELECT DISTINCT EXTRACT(YEAR FROM CAST({cols['date']} AS DATE))::INT AS year
        FROM fact_kpi_timeseries
        {where}
        ORDER BY 1;
        """,
        params if params else None,
    )
    return [int(x) for x in df["year"].dropna().tolist()] if not df.empty else []


# -----------------------------
# Data fetch (precomputed only)
# -----------------------------

@st.cache_data(show_spinner=False)
def fetch_timeseries(
    db_path: str,
    kpi_id: str,
    geo_id: str,
    source_id: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
) -> pd.DataFrame:
    db = DuckDBClient(db_path)
    cols = _discover_timeseries_cols(db)
    where = [f"{cols['kpi']} = ?"]
    params: List[Any] = [kpi_id]

    # geo handling
    if geo_id and geo_id.upper() != "ALL":
        where.append(f"{cols['geo']} = ?")
        params.append(geo_id)

    # source optional
    if cols["src"] and source_id and source_id != "(auto)":
        where.append(f"{cols['src']} = ?")
        params.append(source_id)

    # date range optional
    if start_date:
        where.append(f"CAST({cols['date']} AS DATE) >= CAST(? AS DATE)")
        params.append(start_date)
    if end_date:
        where.append(f"CAST({cols['date']} AS DATE) <= CAST(? AS DATE)")
        params.append(end_date)

    sql = f"""
    SELECT
      CAST({cols['date']} AS DATE) AS date,
      CAST({cols['val']} AS DOUBLE) AS value,
      {cols['kpi']} AS kpi_id,
      {cols['geo']} AS geo_id
      {(',' + cols['src'] + ' AS source_id') if cols['src'] else ''}
    FROM fact_kpi_timeseries
    WHERE {' AND '.join(where)}
    ORDER BY 1;
    """
    df = db.query_df(sql, tuple(params))
    if df is None:
        df = pd.DataFrame()
    return df


@st.cache_data(show_spinner=False)
def fetch_latest_by_geo(
    db_path: str,
    kpi_id: str,
    source_id: Optional[str],
    year: Optional[int],
) -> pd.DataFrame:
    """
    Map dataset: one value per geo for a year (or latest available if year None).
    """
    db = DuckDBClient(db_path)
    cols = _discover_timeseries_cols(db)

    where = [f"{cols['kpi']} = ?"]
    params: List[Any] = [kpi_id]

    if cols["src"] and source_id and source_id != "(auto)":
        where.append(f"{cols['src']} = ?")
        params.append(source_id)

    if year is not None:
        where.append(f"EXTRACT(YEAR FROM CAST({cols['date']} AS DATE))::INT = ?")
        params.append(int(year))

        sql = f"""
        SELECT
          {cols['geo']} AS geo_id,
          AVG(CAST({cols['val']} AS DOUBLE)) AS value
        FROM fact_kpi_timeseries
        WHERE {' AND '.join(where)}
        GROUP BY 1
        ORDER BY 1;
        """
        df = db.query_df(sql, tuple(params))
        return df if df is not None else pd.DataFrame()

    # latest per geo (max date)
    sql = f"""
    WITH ranked AS (
      SELECT
        {cols['geo']} AS geo_id,
        CAST({cols['date']} AS DATE) AS date,
        CAST({cols['val']} AS DOUBLE) AS value,
        ROW_NUMBER() OVER (PARTITION BY {cols['geo']} ORDER BY CAST({cols['date']} AS DATE) DESC) AS rn
      FROM fact_kpi_timeseries
      WHERE {' AND '.join(where)}
    )
    SELECT geo_id, value
    FROM ranked
    WHERE rn = 1
    ORDER BY 1;
    """
    df = db.query_df(sql, tuple(params))
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False)
def fetch_drift_summary(db_path: str, kpi_id: str, geo_id: str) -> Dict[str, Any]:
    """
    Pull a precomputed drift row if available (best-effort, schema-flexible).
    """
    db = DuckDBClient(db_path)
    if not db.table_exists("fact_drift_events"):
        return {}
    cols = db.table_columns("fact_drift_events")
    kpi = _first_existing(cols, ["kpi_id", "kpi"]) or "kpi_id"
    geo = _first_existing(cols, ["geo_id", "geo"]) or "geo_id"
    # common drift fields
    drift_type = _first_existing(cols, ["drift_type", "type"]) or ""
    cp_start = _first_existing(cols, ["cp_start", "changepoint_start", "break_date", "cp_date"]) or ""
    effect = _first_existing(cols, ["effect_size", "magnitude", "delta"]) or ""
    pval = _first_existing(cols, ["p_value", "pval"]) or ""
    rob = _first_existing(cols, ["robustness_score", "robustness", "confidence"]) or ""

    select_cols = [kpi, geo]
    for c in [drift_type, cp_start, effect, pval, rob]:
        if c:
            select_cols.append(c)

    df = db.query_df(
        f"""
        SELECT {', '.join(select_cols)}
        FROM fact_drift_events
        WHERE {kpi} = ? AND {geo} = ?
        ORDER BY 1
        LIMIT 1;
        """,
        (kpi_id, geo_id),
    )
    if df is None or df.empty:
        return {}
    return df.iloc[0].to_dict()


@st.cache_data(show_spinner=False)
def fetch_integrity(db_path: str, kpi_id: str, geo_id: str) -> Dict[str, Any]:
    db = DuckDBClient(db_path)
    if not db.table_exists("fact_integrity_scores"):
        return {}
    cols = db.table_columns("fact_integrity_scores")
    kpi = _first_existing(cols, ["kpi_id", "kpi"]) or "kpi_id"
    geo = _first_existing(cols, ["geo_id", "geo"]) or "geo_id"
    score = _first_existing(cols, ["integrity_score", "score", "reliability_score"]) or ""
    coverage = _first_existing(cols, ["coverage", "coverage_pct"]) or ""
    freshness = _first_existing(cols, ["freshness_days", "freshness"]) or ""
    select_cols = [kpi, geo]
    for c in [score, coverage, freshness]:
        if c:
            select_cols.append(c)

    df = db.query_df(
        f"""
        SELECT {', '.join(select_cols)}
        FROM fact_integrity_scores
        WHERE {kpi} = ? AND {geo} = ?
        ORDER BY 1
        LIMIT 1;
        """,
        (kpi_id, geo_id),
    )
    if df is None or df.empty:
        return {}
    return df.iloc[0].to_dict()


# -----------------------------
# Question inference (LLM-first, but deterministic inference for routing)
# -----------------------------

def infer_from_question(question: str, known_kpis: List[str], known_geos: List[str], known_years: List[int]) -> Dict[str, Any]:
    q = (question or "").strip()
    q_low = q.lower()

    # KPI: direct match by id token
    kpi_guess = None
    for k in known_kpis:
        if k.lower() in q_low:
            kpi_guess = k
            break

    # KPI: fuzzy match via simple keywords
    if not kpi_guess:
        if "pm2.5" in q_low or "pm25" in q_low or "air quality" in q_low:
            for cand in ["pm25_monthly_mean", "pm25_monthly_mean_country", "pm25"]:
                for k in known_kpis:
                    if cand.lower() in k.lower():
                        kpi_guess = k
                        break
                if kpi_guess:
                    break
        elif "precip" in q_low or "rain" in q_low or "precipitation" in q_low:
            for k in known_kpis:
                if "precip" in k.lower():
                    kpi_guess = k
                    break
        elif "temp" in q_low or "temperature" in q_low or "heat" in q_low:
            for k in known_kpis:
                if "temp" in k.lower():
                    kpi_guess = k
                    break

    # GEO: accept ISO3 tokens or exact geo_id substrings
    geo_guess = None
    # explicit GLOBAL
    if re.search(r"\bglobal\b|\bworld\b|\bworldwide\b", q_low):
        geo_guess = "GLOBAL"

    if not geo_guess:
        # match geo_id in question (covers ISO3:IND or IND)
        for g in known_geos:
            if g.lower() in q_low:
                geo_guess = g
                break

    if not geo_guess:
        # match ISO3 codes (3 letters) present in question
        m = re.findall(r"\b[A-Z]{3}\b", q)
        if m:
            # choose first that exists as either "ISO3:XXX" or "XXX"
            for token in m:
                for g in known_geos:
                    if _norm_geo(g) == token:
                        geo_guess = g
                        break
                if geo_guess:
                    break

    # Year: first 4-digit year within known range
    year_guess: Optional[int] = None
    ym = re.search(r"\b(19\d{2}|20\d{2})\b", q)
    if ym:
        y = int(ym.group(1))
        if not known_years or y in known_years:
            year_guess = y

    return {"kpi_id": kpi_guess, "geo_id": geo_guess, "year": year_guess}


# -----------------------------
# Deterministic fallback answer (no LLM)
# -----------------------------

def _fmt_num(x: Any, digits: int = 2) -> str:
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "NA"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "NA"


def deterministic_answer(
    question: str,
    db_path: str,
    inferred: Dict[str, Any],
    default_kpi: str,
    default_geo: str,
    default_source: str,
) -> str:
    kpi_id = inferred.get("kpi_id") or default_kpi
    geo_id = inferred.get("geo_id") or default_geo
    year = inferred.get("year")

    # If user asks for global but we only have country series, keep GLOBAL only for drift/integrity pulls
    # and use a country-level series if available.
    ts_geo = geo_id if geo_id != "GLOBAL" else default_geo

    # Pull 12m window if available; else full series
    ts = fetch_timeseries(
        db_path=db_path,
        kpi_id=kpi_id,
        geo_id=ts_geo,
        source_id=None if default_source == "(auto)" else default_source,
        start_date=None,
        end_date=None,
    )

    if ts is None or ts.empty:
        return (
            f"I couldn’t find data for kpi={kpi_id}, geo={geo_id}. "
            f"Try asking with a known geo_id (e.g., ISO3:IND) or a KPI id from the sidebar."
        )

    ts = ts.sort_values("date")
    # year filter if asked
    if year is not None:
        ts_y = ts[ts["date"].dt.year == year].copy()
        if not ts_y.empty:
            ts = ts_y

    start_v = ts["value"].iloc[0]
    end_v = ts["value"].iloc[-1]
    abs_change = None if pd.isna(start_v) or pd.isna(end_v) else float(end_v - start_v)
    pct_change = None
    if abs_change is not None and pd.notna(start_v) and float(start_v) != 0.0:
        pct_change = float(abs_change) / float(start_v) * 100.0

    drift = fetch_drift_summary(db_path, kpi_id, geo_id) or {}
    integ = fetch_integrity(db_path, kpi_id, geo_id) or {}

    lines: List[str] = []
    lines.append(f"KPI: {kpi_id} | Geo: {geo_id}" + (f" | Year: {year}" if year else ""))
    lines.append("")
    lines.append("Trend (deterministic):")
    lines.append(f"- Start: {_fmt_num(start_v)}  End: {_fmt_num(end_v)}")
    lines.append(f"- Abs change: {_fmt_num(abs_change)}  Pct change: {_fmt_num(pct_change)}%")
    lines.append(f"- Points used: {len(ts)}  Window: {str(ts['date'].min())[:10]} to {str(ts['date'].max())[:10]}")
    lines.append("")

    if drift:
        # show a few fields if present
        keys = [k for k in ["drift_type", "cp_start", "effect_size", "p_value", "robustness_score"] if k in drift]
        if keys:
            lines.append("Drift (precomputed):")
            for k in keys:
                lines.append(f"- {k}: {drift.get(k)}")
            lines.append("")
    if integ:
        lines.append("Integrity (precomputed):")
        for k in ["integrity_score", "coverage", "freshness_days"]:
            if k in integ:
                lines.append(f"- {k}: {integ.get(k)}")
        lines.append("")

    lines.append("Trust note:")
    lines.append("- If coverage is low or freshness is old, treat changes as lower confidence.")
    return "\n".join(lines)


# -----------------------------
# OpenAI (LLM-first) with safe fallback
# -----------------------------

def llm_answer_openai(question: str, facts: Dict[str, Any]) -> str:
    """
    LLM-first narrative. If OPENAI_API_KEY missing or OpenAI errors -> caller must fallback.
    """
    # Explicitly load .env for Streamlit runs (again, safe)
    load_dotenv(".env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing")

    # Use OpenAI python SDK if available; otherwise error -> fallback
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("openai package not installed") from e

    client = OpenAI(api_key=api_key)

    system = (
        "You are a trust-aware analytics assistant. "
        "You MUST only use the provided FACTS JSON. "
        "If a value is missing, say it is missing. "
        "Keep the answer concise and structured: Answer, Evidence, Trust."
    )

    user = (
        "USER QUESTION:\n"
        f"{question}\n\n"
        "FACTS JSON (authoritative):\n"
        f"{json.dumps(facts, indent=2, default=str)[:12000]}"
    )

    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def build_facts_payload(
    question: str,
    db_path: str,
    inferred: Dict[str, Any],
    explore_kpi: str,
    explore_geo: str,
    explore_source: str,
) -> Dict[str, Any]:
    kpi_id = inferred.get("kpi_id") or explore_kpi
    geo_id = inferred.get("geo_id") or explore_geo
    year = inferred.get("year")

    ts_geo = geo_id if geo_id != "GLOBAL" else explore_geo
    ts = fetch_timeseries(
        db_path=db_path,
        kpi_id=kpi_id,
        geo_id=ts_geo,
        source_id=None if explore_source == "(auto)" else explore_source,
        start_date=None,
        end_date=None,
    )

    payload: Dict[str, Any] = {
        "question": question,
        "inferred": {"kpi_id": kpi_id, "geo_id": geo_id, "year": year},
        "definitions": {
            "kpi": KPI_DEFINITIONS.get(kpi_id, "(no definition yet)"),
            **TRUST_DEFINITIONS,
        },
    }

    if ts is None or ts.empty:
        payload["timeseries"] = {"available": False, "reason": "no rows returned"}
        return payload

    ts = ts.sort_values("date")
    if year is not None:
        ts_y = ts[ts["date"].dt.year == year].copy()
        if not ts_y.empty:
            ts = ts_y

    start_v = ts["value"].iloc[0]
    end_v = ts["value"].iloc[-1]
    abs_change = None if pd.isna(start_v) or pd.isna(end_v) else float(end_v - start_v)
    pct_change = None
    if abs_change is not None and pd.notna(start_v) and float(start_v) != 0.0:
        pct_change = float(abs_change) / float(start_v) * 100.0

    payload["timeseries"] = {
        "available": True,
        "n_points": int(len(ts)),
        "start_date": str(ts["date"].min())[:10],
        "end_date": str(ts["date"].max())[:10],
        "start_value": None if pd.isna(start_v) else float(start_v),
        "end_value": None if pd.isna(end_v) else float(end_v),
        "abs_change": abs_change,
        "pct_change": pct_change,
        "mean": None if ts["value"].isna().all() else float(ts["value"].mean()),
        "min": None if ts["value"].isna().all() else float(ts["value"].min()),
        "max": None if ts["value"].isna().all() else float(ts["value"].max()),
    }

    payload["drift_event"] = fetch_drift_summary(db_path, kpi_id, geo_id) or {"available": False}
    payload["integrity"] = fetch_integrity(db_path, kpi_id, geo_id) or {"available": False}

    return payload


# -----------------------------
# UI (single page)
# -----------------------------

def render_definitions(known_kpis: List[str], selected_kpi: str) -> None:
    st.subheader("Definitions")
    st.write(f"**KPI:** `{selected_kpi}` — {KPI_DEFINITIONS.get(selected_kpi, '(no definition yet)')}")
    st.write(f"**Drift:** {TRUST_DEFINITIONS['drift']}")
    st.write(f"**Integrity score:** {TRUST_DEFINITIONS['integrity_score']}")
    st.write(f"**Trust:** {TRUST_DEFINITIONS['trust']}")
    st.write(f"**Contributions:** {TRUST_DEFINITIONS['contributions']}")
    # Optional: show all KPI defs in a compact expander
    with st.expander("All KPI definitions (optional)", expanded=False):
        for k in sorted(set(known_kpis)):
            st.write(f"- `{k}`: {KPI_DEFINITIONS.get(k, '(no definition yet)')}")


def render_trend(db_path: str, kpi_id: str, geo_id: str, source_id: str, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    st.subheader("Trend")
    df = fetch_timeseries(db_path, kpi_id, geo_id, None if source_id == "(auto)" else source_id, start_date, end_date)
    if df is None or df.empty:
        st.warning("No timeseries rows returned for the selected filters.")
        return pd.DataFrame()
    df = df.sort_values("date")
    st.line_chart(df.set_index("date")["value"])
    return df


def render_map(db_path: str, kpi_id: str, source_id: str, year: Optional[int]) -> None:
    st.subheader("Map")
    geo_df = fetch_latest_by_geo(db_path, kpi_id, None if source_id == "(auto)" else source_id, year)
    if geo_df is None or geo_df.empty:
        st.warning("No geo-level rows returned for this KPI/year.")
        return

    geo_df = geo_df.copy()
    geo_df["iso3"] = geo_df["geo_id"].astype(str).map(_norm_geo)
    geo_df = geo_df[geo_df["iso3"].str.len() == 3]

    try:
        import plotly.express as px  # type: ignore

        title = f"{kpi_id} — {year}" if year else f"{kpi_id} — latest"
        fig = px.choropleth(
            geo_df,
            locations="iso3",
            color="value",
            hover_name="iso3",
            title=title,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.warning("Plotly is not installed (or failed to load). Showing a table instead.")
        st.dataframe(geo_df[["geo_id", "value"]].sort_values("geo_id"), use_container_width=True)


def render_qa(
    db_path: str,
    known_kpis: List[str],
    known_geos: List[str],
    known_years: List[int],
    explore_kpi: str,
    explore_geo: str,
    explore_source: str,
) -> None:
    st.subheader("Q&A (LLM-first, deterministic fallback)")

    q = st.text_input(
        "Ask anything about the KPIs (you can name KPI ids, ISO3 geos, and years).",
        placeholder="e.g., How has pm25_monthly_mean changed in ISO3:IND since 2015?",
    )

    if not q:
        return

    inferred = infer_from_question(q, known_kpis, known_geos, known_years)

    # Build facts payload once (used by both LLM and fallback)
    facts = build_facts_payload(q, db_path, inferred, explore_kpi, explore_geo, explore_source)

    # Try LLM first
    try:
        ans = llm_answer_openai(q, facts)
        st.markdown(ans)
        with st.expander("Debug: inferred context (optional)", expanded=False):
            st.json(facts["inferred"])
    except Exception as e:
        st.warning(f"LLM unavailable (using deterministic fallback). Reason: {type(e).__name__}")
        ans = deterministic_answer(q, db_path, inferred, explore_kpi, explore_geo, explore_source)
        st.text(ans)


def main() -> None:
    st.set_page_config(page_title="Climate Lens", layout="wide")
    st.title("Climate Lens")
    st.caption("<h4 style='margin-top:-10px; color:#666;'>Trust-aware conversational BI for weather and climate KPIs</h4>",unsafe_allow_html=True)

    s = Settings.load(".env")

    # Hard fail early if DB missing (better than weird empty UI)
    if not s.duckdb_path:
        st.error("DUCKDB_PATH not set and no default DuckDB found. Add DUCKDB_PATH to .env.")
        st.stop()

    try:
        db = DuckDBClient(s.duckdb_path)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Verify required tables exist (but degrade gracefully if optional ones missing)
    missing = [t for t in REQUIRED_TABLES if not db.table_exists(t)]
    if missing:
        st.error(f"Missing required DuckDB tables: {', '.join(missing)}")
        st.stop()

    # Optional explore controls (do not constrain Q&A)
    with st.sidebar:
        st.header("Explore (optional)")
        known_kpis = list_kpis(s.duckdb_path)
        selected_kpi = st.selectbox("KPI", options=known_kpis, index=0 if known_kpis else 0)

        # source is optional; if table has no source column, keep (auto)
        selected_source = "(auto)"

        known_geos = list_geos(s.duckdb_path, selected_kpi) if selected_kpi else list_geos(s.duckdb_path, None)
        selected_geo = st.selectbox("Geo", options=known_geos, index=0 if known_geos else 0)

        years = list_years(s.duckdb_path, selected_kpi) if selected_kpi else list_years(s.duckdb_path, None)
        # Map year selection (optional)
        year_sel = st.selectbox("Map year", options=["(latest)"] + [str(y) for y in years[-25:]], index=0)

        st.caption("Note: Q&A ignores these filters and infers from the question.")

    # Main page layout: definitions, trend + map, then Q&A
    render_definitions(known_kpis, selected_kpi)

    col1, col2 = st.columns([2, 1])
    with col1:
        # date range for trend (optional)
        st.caption("Trend date window (optional)")
        start_date = st.text_input("Start date (YYYY-MM-DD)", value="")
        end_date = st.text_input("End date (YYYY-MM-DD)", value="")
        start_date = start_date.strip() or None
        end_date = end_date.strip() or None
        _ = render_trend(s.duckdb_path, selected_kpi, selected_geo, selected_source, start_date, end_date)

    with col2:
        map_year = None if year_sel == "(latest)" else int(year_sel)
        render_map(s.duckdb_path, selected_kpi, selected_source, map_year)

    st.divider()
    render_qa(
        db_path=s.duckdb_path,
        known_kpis=known_kpis,
        known_geos=known_geos,
        known_years=years,
        explore_kpi=selected_kpi,
        explore_geo=selected_geo,
        explore_source=selected_source,
    )


if __name__ == "__main__":
    main()
