#!/usr/bin/env python3
"""
scripts/10_conversational_bi.py

Conversational BI driver for Trust-Aware KPI Drift + Attribution.

Edits (to accommodate precip + keep logic consistent across KPIs):
- Added KPI config resolver (pm25 / precip / temp) so this script can run for precip without hardcoding PM2.5.
- GEO trend mode now uses the correct per-geo KPI for the selected metric.
- Attribution % now uses ABS-denominator (SUM(ABS(contribution_value))) to behave correctly for negative deltas too.
- Prompts and printed headers no longer hardcode "GLOBAL PM2.5" (uses metric label).
- CLI now supports --metric (pm25|precip|temp). You can still pass --kpi explicitly if you want.

Assumptions:
- Drift events are in fact_drift_events.
- Integrity is in fact_integrity_scores.
- Contributions are in fact_contributions (methods from your 08/08b scripts).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI

# repo imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trust_bi.common.settings import Settings  # noqa: E402
from trust_bi.warehouse.duckdb_io import DuckDBClient  # noqa: E402


# -----------------------------
# KPI config
# -----------------------------

@dataclass(frozen=True)
class KPIConfig:
    label: str
    global_kpi_id: str
    country_kpi_id: str
    source_id: str
    global_geo_id: str = "GLOBAL"

    # Contribution methods (consistent with 08 / 08b)
    base_method: str = "pop_weighted_country_delta"
    conf_method: str = "pop_weighted_conf_weighted_delta"


KPI_CONFIGS: Dict[str, KPIConfig] = {
    "pm25": KPIConfig(
        label="PM2.5",
        global_kpi_id="pm25_monthly_mean_global_popw",
        country_kpi_id="pm25_monthly_mean",
        source_id="cams",
    ),
    "precip": KPIConfig(
        label="Precipitation",
        global_kpi_id="precip_country_monthly_sum_global_popw",
        country_kpi_id="precip_country_monthly_sum",
        source_id="openmeteo",
    ),
    "temp": KPIConfig(
        label="Temperature",
        global_kpi_id="temp_country_monthly_mean_global_popw",
        country_kpi_id="temp_country_monthly_mean",
        source_id="openmeteo",
    ),
}


def _resolve_config(metric: Optional[str], kpi_id: Optional[str]) -> KPIConfig:
    """
    Resolve KPI config from either:
    - --metric (pm25|precip|temp) OR
    - --kpi (explicit global kpi id)

    If --kpi matches one of known global KPI IDs, we pick that config.
    Otherwise, we fall back to pm25 to avoid breaking runs (but you should pass --metric).
    """
    if metric:
        m = metric.strip().lower()
        if m not in KPI_CONFIGS:
            raise ValueError(f"Unknown --metric '{metric}'. Choose from: {sorted(KPI_CONFIGS.keys())}")
        return KPI_CONFIGS[m]

    if kpi_id:
        kid = kpi_id.strip()
        for cfg in KPI_CONFIGS.values():
            if cfg.global_kpi_id == kid:
                return cfg

    # default
    return KPI_CONFIGS["pm25"]


# -----------------------------
# Params
# -----------------------------

@dataclass(frozen=True)
class Params:
    # global KPI
    kpi_id: str
    geo_id: str

    # per-geo KPI (ISO3:*), used when the user asks for specific countries
    country_kpi_id: str

    # LLM + behavior
    model: str = "gpt-4.1-mini"
    temperature: float = 0.2

    # optional trend request
    geo_ids: Optional[List[str]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # attribution
    top_n: int = 10


# -----------------------------
# DB helpers
# -----------------------------

def _latest_global_event(db: DuckDBClient, kpi_id: str, geo_id: str) -> Optional[Dict[str, Any]]:
    df = db.query_df(
        f"""
        SELECT
          event_id,
          kpi_id,
          geo_id,
          cp_start,
          drift_type,
          effect_size,
          effect_size_pct,
          robustness_score,
          p_value,
          detected_at
        FROM fact_drift_events
        WHERE kpi_id = '{kpi_id}' AND geo_id = '{geo_id}'
        ORDER BY detected_at DESC
        LIMIT 1;
        """
    )
    if df.empty:
        return None
    r = df.iloc[0].to_dict()
    r["cp_start"] = pd.to_datetime(r["cp_start"])
    r["detected_at"] = pd.to_datetime(r["detected_at"])
    return r


def _global_integrity_at_cp(db: DuckDBClient, kpi_id: str, geo_id: str, cp_date: pd.Timestamp) -> Optional[Dict[str, Any]]:
    df = db.query_df(
        f"""
        SELECT
          kpi_id,
          geo_id,
          date,
          confidence_grade,
          confidence_score,
          blocking_reason,
          coverage_score,
          missingness_score,
          uncertainty_score,
          source_stability_score
        FROM fact_integrity_scores
        WHERE kpi_id = '{kpi_id}'
          AND geo_id = '{geo_id}'
          AND date = CAST('{cp_date.date().isoformat()}' AS DATE)
        LIMIT 1;
        """
    )
    if df.empty:
        return None
    out = df.iloc[0].to_dict()
    out["date"] = pd.to_datetime(out["date"])
    return out


def _top_contributors(
    db: DuckDBClient,
    event_id: str,
    method: str,
    top_n: int = 10,
) -> Dict[str, Any]:
    """
    Returns:
      - top: list of {contributor_geo_id, contribution_value, contribution_pct}
      - remainder: aggregated remainder bucket
      - denom_abs: SUM(ABS(contribution_value)) for stable pct math
    """
    # ABS denom so negative deltas behave sanely too (precip drift can be negative)
    denom_df = db.query_df(
        f"""
        SELECT SUM(ABS(contribution_value)) AS denom_abs
        FROM fact_contributions
        WHERE event_id = '{event_id}' AND method = '{method}';
        """
    )
    denom_abs = float(denom_df.loc[0, "denom_abs"]) if not denom_df.empty and pd.notna(denom_df.loc[0, "denom_abs"]) else 0.0

    if denom_abs <= 0:
        return {"top": [], "remainder": None, "denom_abs": denom_abs}

    top = db.query_df(
        f"""
        SELECT
          contributor_geo_id,
          contribution_value,
          (ABS(contribution_value) / {denom_abs}) AS contribution_pct
        FROM fact_contributions
        WHERE event_id = '{event_id}' AND method = '{method}'
        ORDER BY contribution_pct DESC
        LIMIT {int(top_n)};
        """
    )

    if top.empty:
        return {"top": [], "remainder": None, "denom_abs": denom_abs}

    top_list = top.to_dict(orient="records")
    top_keys = ",".join([f"'{x}'" for x in top["contributor_geo_id"].astype(str).tolist()])

    remainder_df = db.query_df(
        f"""
        SELECT
          'OTHER' AS contributor_geo_id,
          SUM(contribution_value) AS contribution_value,
          (SUM(ABS(contribution_value)) / {denom_abs}) AS contribution_pct
        FROM fact_contributions
        WHERE event_id = '{event_id}'
          AND method = '{method}'
          AND contributor_geo_id NOT IN ({top_keys});
        """
    )

    remainder = None
    if not remainder_df.empty and pd.notna(remainder_df.loc[0, "contribution_value"]):
        remainder = remainder_df.iloc[0].to_dict()

    return {"top": top_list, "remainder": remainder, "denom_abs": denom_abs}


# -----------------------------
# Trend data discovery + fetching
# -----------------------------

def _discover_timeseries_table(db: DuckDBClient) -> Optional[str]:
    """
    Best-effort discovery of a time series table.

    Find table with: kpi_id, geo_id, date, and (value OR kpi_value).
    If multiple match, picks the first alphabetically.
    """
    candidates = db.query_df(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
        ORDER BY table_name;
        """
    )
    if candidates.empty:
        return None

    for t in candidates["table_name"].tolist():
        cols = db.query_df(
            f"""
            SELECT LOWER(column_name) AS col
            FROM information_schema.columns
            WHERE table_schema = 'main' AND table_name = '{t}'
            """
        )
        if cols.empty:
            continue
        colset = set(cols["col"].tolist())
        if {"kpi_id", "geo_id", "date"}.issubset(colset) and ("value" in colset or "kpi_value" in colset):
            return t
    return None


def _get_timeseries(db: DuckDBClient, table: str, kpi_id: str, geo_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    val_col = "value"
    cols = db.query_df(
        f"""
        SELECT LOWER(column_name) AS col
        FROM information_schema.columns
        WHERE table_schema = 'main' AND table_name = '{table}'
        """
    )
    if not cols.empty and "kpi_value" in set(cols["col"].tolist()):
        val_col = "kpi_value"

    df = db.query_df(
        f"""
        SELECT
          CAST(date AS DATE) AS date,
          {val_col} AS value
        FROM {table}
        WHERE kpi_id = '{kpi_id}'
          AND geo_id = '{geo_id}'
          AND CAST(date AS DATE) >= CAST('{start_date}' AS DATE)
          AND CAST(date AS DATE) <= CAST('{end_date}' AS DATE)
        ORDER BY CAST(date AS DATE);
        """
    )
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    return df


def _summarize_series(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {
            "n_points": 0,
            "start_date": None,
            "end_date": None,
            "start_value": None,
            "end_value": None,
            "abs_change": None,
            "pct_change": None,
            "mean": None,
            "min": None,
            "max": None,
        }

    start = df.iloc[0]
    end = df.iloc[-1]
    start_v = float(start["value"]) if pd.notna(start["value"]) else None
    end_v = float(end["value"]) if pd.notna(end["value"]) else None

    abs_change = None
    pct_change = None
    if start_v is not None and end_v is not None:
        abs_change = end_v - start_v
        if start_v != 0:
            pct_change = abs_change / start_v

    return {
        "n_points": int(len(df)),
        "start_date": start["date"].date().isoformat(),
        "end_date": end["date"].date().isoformat(),
        "start_value": start_v,
        "end_value": end_v,
        "abs_change": abs_change,
        "pct_change": pct_change,
        "mean": float(df["value"].mean()) if len(df) else None,
        "min": float(df["value"].min()) if len(df) else None,
        "max": float(df["value"].max()) if len(df) else None,
    }


def _sample_points(df: pd.DataFrame, max_points: int = 24) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    use = df if len(df) <= max_points else df.tail(max_points)
    return [{"date": r["date"].date().isoformat(), "value": float(r["value"])} for _, r in use.iterrows()]


def _latest_geo_event(db: DuckDBClient, kpi_id: str, geo_id: str) -> Optional[Dict[str, Any]]:
    df = db.query_df(
        f"""
        SELECT
          event_id,
          kpi_id,
          geo_id,
          cp_start,
          drift_type,
          effect_size,
          effect_size_pct,
          robustness_score,
          p_value,
          detected_at
        FROM fact_drift_events
        WHERE kpi_id = '{kpi_id}' AND geo_id = '{geo_id}'
        ORDER BY detected_at DESC
        LIMIT 1;
        """
    )
    if df.empty:
        return None
    r = df.iloc[0].to_dict()
    r["cp_start"] = pd.to_datetime(r["cp_start"])
    r["detected_at"] = pd.to_datetime(r["detected_at"])
    return r


# -----------------------------
# Payload building
# -----------------------------

def _build_evidence_payload(db: DuckDBClient, p: Params, cfg: KPIConfig) -> Dict[str, Any]:
    """
    Build evidence payload.

    Modes:
    - GLOBAL mode (default): latest GLOBAL drift + attribution (reads 08/08b outputs).
    - GEO trend mode: if p.geo_ids is provided -> time series summary for those geos (+ optional latest drift per geo).
    """
    # GEO trend mode
    if p.geo_ids:
        effective_kpi = p.country_kpi_id  # key fix: precip uses precip_country_monthly_sum, not the global KPI
        ts_table = _discover_timeseries_table(db)

        # Determine date range
        if ts_table is not None and (p.start_date is None or p.end_date is None):
            maxd = db.query_df(f"SELECT MAX(CAST(date AS DATE)) AS max_date FROM {ts_table};")
            max_date = None
            if not maxd.empty and pd.notna(maxd.loc[0, "max_date"]):
                max_date = pd.to_datetime(maxd.loc[0, "max_date"]).date()
            if max_date is not None:
                end_date = p.end_date or max_date.isoformat()
                # last 24 months default (monthly KPIs)
                start_date = p.start_date or (pd.Timestamp(end_date) - pd.DateOffset(months=24)).date().isoformat()
            else:
                # fallback
                end_date = p.end_date or datetime.utcnow().date().isoformat()
                start_date = p.start_date or (pd.Timestamp(end_date) - pd.DateOffset(months=24)).date().isoformat()
        else:
            end_date = p.end_date or datetime.utcnow().date().isoformat()
            start_date = p.start_date or (pd.Timestamp(end_date) - pd.DateOffset(months=24)).date().isoformat()

        geo_payload = []
        for g in p.geo_ids:
            g_norm = g.strip()
            if not g_norm:
                continue

            # If user gives "IND" or "USA", normalize to ISO3:*
            if re.fullmatch(r"[A-Z]{3}", g_norm):
                g_norm = f"ISO3:{g_norm}"

            series = None
            summary = None
            samples = None
            if ts_table:
                series = _get_timeseries(db, ts_table, effective_kpi, g_norm, start_date, end_date)
                summary = _summarize_series(series)
                samples = _sample_points(series)
            else:
                summary = _summarize_series(pd.DataFrame())
                samples = []

            latest_event = _latest_geo_event(db, effective_kpi, g_norm)

            geo_payload.append(
                {
                    "geo_id": g_norm,
                    "kpi_id": effective_kpi,
                    "start_date": start_date,
                    "end_date": end_date,
                    "series_summary": summary,
                    "sample_points": samples,
                    "latest_drift_event": latest_event,
                }
            )

        return {
            "mode": "geo_trends",
            "metric_label": cfg.label,
            "kpi_id": effective_kpi,
            "geos": geo_payload,
        }

    # GLOBAL mode
    ev = _latest_global_event(db, p.kpi_id, p.geo_id)
    if ev is None:
        return {
            "mode": "global",
            "metric_label": cfg.label,
            "error": f"No drift event found for kpi_id={p.kpi_id}, geo_id={p.geo_id}",
        }

    cp = ev["cp_start"]
    integrity = _global_integrity_at_cp(db, p.kpi_id, p.geo_id, cp)

    base = _top_contributors(db, ev["event_id"], method=cfg.base_method, top_n=p.top_n)
    conf = _top_contributors(db, ev["event_id"], method=cfg.conf_method, top_n=p.top_n)

    return {
        "mode": "global",
        "metric_label": cfg.label,
        "kpi_id": ev["kpi_id"],
        "geo_id": ev["geo_id"],
        "source_id": cfg.source_id,
        "event": {
            "event_id": ev["event_id"],
            "cp_start": ev["cp_start"].isoformat(),
            "drift_type": ev.get("drift_type"),
            "effect_size": ev.get("effect_size"),
            "effect_size_pct": ev.get("effect_size_pct"),
            "robustness_score": ev.get("robustness_score"),
            "p_value": ev.get("p_value"),
            "detected_at": ev.get("detected_at").isoformat() if ev.get("detected_at") is not None else None,
        },
        "integrity_at_cp": integrity,
        "attribution": {
            "base_method": cfg.base_method,
            "conf_method": cfg.conf_method,
            "top_n": p.top_n,
            "base": base,
            "confidence_weighted": conf,
        },
    }


# -----------------------------
# Prompting / LLM
# -----------------------------

def _build_prompt(payload: Dict[str, Any]) -> str:
    # NOTE: keep prompt generic (no PM2.5 hardcoding)
    if payload.get("mode") == "geo_trends":
        return f"""
You are a careful analytics assistant.

User is asking for trend analysis across one or more geographies for a KPI.
Use the evidence payload to summarize trends and mention any detected drift events per geo.

Requirements:
- Be concise.
- Always include concrete dates.
- If the sample is small or missing, say so.
- Do NOT claim causality. Observational only.

Evidence payload (JSON):
{json.dumps(payload, indent=2, default=str)}
""".strip()

    metric_label = payload.get("metric_label", "KPI")
    return f"""
You are a careful analytics assistant producing a trust-aware drift summary for a GLOBAL KPI.

Your job:
1) Summarize the detected shift (what changed, when, and by how much).
2) Report evidence quality: robustness_score, drift_type, effect_size_pct, and integrity grade at the change point.
3) Provide attribution highlights from two decompositions:
   - base attribution (population-weighted contributors)
   - confidence-weighted attribution (after reliability filtering)
4) Do NOT claim causality. Observational only.
5) If integrity indicates blocking_reason or poor grade, refuse interpretation and explain.

Keep it readable and structured.

Evidence payload (JSON):
{json.dumps(payload, indent=2, default=str)}

Now write the trust-aware summary for: {metric_label} (GLOBAL).
""".strip()


def _call_llm(client: OpenAI, model: str, temperature: float, prompt: str) -> str:
    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
    )
    # normalize output
    if hasattr(resp, "output_text"):
        return resp.output_text
    # fallback
    return str(resp)


# -----------------------------
# CLI / main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Trust-aware Conversational BI (GLOBAL drift + GEO trends)")

    # New: metric shortcut
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="Metric shortcut: pm25 | precip | temp (recommended). If provided, overrides default KPI config.",
    )

    # Backward compatible: explicit kpi + geo
    parser.add_argument(
        "--kpi",
        type=str,
        default=None,
        help="Explicit KPI ID (typically a GLOBAL *_global_popw KPI). If it matches a known config, we infer other fields.",
    )
    parser.add_argument(
        "--geo",
        type=str,
        default="GLOBAL",
        help="Geo ID for drift mode (default GLOBAL).",
    )

    # Trend mode
    parser.add_argument(
        "--geo-ids",
        type=str,
        default=None,
        help="Comma-separated geo_ids for trend mode (e.g., ISO3:IND,ISO3:USA or IND,USA). If set, runs geo trend mode.",
    )
    parser.add_argument("--start-date", type=str, default=None, help="Trend mode start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, default=None, help="Trend mode end date (YYYY-MM-DD).")

    # Output controls
    parser.add_argument("--top-n", type=int, default=10, help="Top N contributors to show (global mode).")

    # LLM
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="OpenAI model")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")

    args = parser.parse_args()

    # Resolve KPI config (precip support lives here)
    cfg = _resolve_config(args.metric, args.kpi)
    kpi_id = args.kpi.strip() if args.kpi else cfg.global_kpi_id

    # Parse geo_ids
    geo_ids = None
    if args.geo_ids:
        geo_ids = [x.strip() for x in args.geo_ids.split(",") if x.strip()]

    p = Params(
        kpi_id=kpi_id,
        geo_id=args.geo,
        country_kpi_id=cfg.country_kpi_id,
        model=args.model,
        temperature=args.temperature,
        geo_ids=geo_ids,
        start_date=args.start_date,
        end_date=args.end_date,
        top_n=args.top_n,
    )

    # Load settings + DB
    s = Settings.load(".env")
    db = DuckDBClient(s.duckdb_path)

    # OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment.")
    client = OpenAI(api_key=api_key)

    payload = _build_evidence_payload(db, p, cfg)
    prompt = _build_prompt(payload)

    # Optional debug prints (keep commented unless you want noisy logs)
    # print("DEBUG payload:", json.dumps(payload, indent=2, default=str)[:3000])
    # print("DEBUG prompt head:", prompt[:1500])

    out = _call_llm(client, p.model, p.temperature, prompt)
    print(out)


if __name__ == "__main__":
    main()
