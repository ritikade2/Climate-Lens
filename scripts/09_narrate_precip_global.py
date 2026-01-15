from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

import pandas as pd

# repo imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trust_bi.common.settings import Settings  # noqa: E402
from trust_bi.warehouse.duckdb_io import DuckDBClient  # noqa: E402
from trust_bi.narrative.audit import NarrativeAuditRow, write_audit  # noqa: E402

# -----------------------------
# Config / constants
# -----------------------------
GLOBAL_KPI = "precip_country_monthly_sum_global_popw"
COUNTRY_KPI = "precip_country_monthly_sum"
SOURCE_ID = "openmeteo"
GLOBAL_GEO = "GLOBAL"

METHOD_BASE = "pop_weighted_country_delta"
METHOD_CONF = "pop_weighted_conf_weighted_delta"


@dataclass(frozen=True)
class Params:
    min_global_grade: str = "C"   # allow narrative for C+; refuse for D
    min_robustness: float = 0.80  # refuse if drift is weak
    top_n: int = 10
    store_full_text: bool = False  # keep False for memory safety


def grade_ok(grade: Optional[str], min_grade: str) -> bool:
    order = {"A": 4, "B": 3, "C": 2, "D": 1}
    if grade is None:
        return False
    return order.get(str(grade), 0) >= order.get(str(min_grade), 0)


def _normalize_grade(g: Optional[str]) -> Optional[str]:
    """
    Compatibility: some integrity scripts store confidence_grade as high/medium/low/none.
    PM2.5 global uses A/B/C/D; precip may still have high/medium/low in older rows.
    """
    if g is None:
        return None
    s = str(g).strip()
    m = {"high": "A", "medium": "B", "low": "C", "none": "D"}
    return m.get(s.lower(), s)


def main() -> None:
    s = Settings.load(".env")
    db = DuckDBClient(s.duckdb_path)
    p = Params()

    request_id = str(uuid4())
    user_query = "Summarize the latest detected GLOBAL precipitation drift and top contributors (trust-aware)."

    # 1) Pick latest GLOBAL drift event
    ev = db.query_df(
        f"""
        SELECT
          event_id, kpi_id, geo_id, detected_at, cp_start, cp_end,
          effect_size, effect_size_pct, p_value, robustness_score,
          drift_type, artifact_evidence
        FROM fact_drift_events
        WHERE kpi_id='{GLOBAL_KPI}' AND geo_id='{GLOBAL_GEO}'
        ORDER BY detected_at DESC
        LIMIT 1;
        """
    )

    if ev.empty:
        refusal = "no_global_drift_event"
        response_text = "Refusal: No GLOBAL drift event exists yet for the requested KPI."
        _audit(s, request_id, user_query, allowed=False, refusal=refusal, response_text=response_text)
        print(response_text)
        return

    e = ev.loc[0].to_dict()
    event_id = e["event_id"]
    cp = str(e["cp_start"])
    robustness = float(e["robustness_score"]) if e["robustness_score"] is not None else 0.0

    # 2) Trust gate: GLOBAL integrity at cp_start
    g = db.query_df(
        f"""
        SELECT confidence_grade, confidence_score, blocking_reason
        FROM fact_integrity_scores
        WHERE kpi_id='{GLOBAL_KPI}' AND geo_id='{GLOBAL_GEO}' AND date='{cp}'
        LIMIT 1;
        """
    )
    if g.empty:
        refusal = "missing_global_integrity"
        response_text = f"Refusal: Missing GLOBAL integrity score at change point {cp}."
        _audit(
            s,
            request_id,
            user_query,
            allowed=False,
            refusal=refusal,
            response_text=response_text,
            event_id=event_id,
            cp=cp,
        )
        print(response_text)
        return

    g_grade_raw = str(g.loc[0, "confidence_grade"])
    g_grade = _normalize_grade(g_grade_raw)
    g_block = g.loc[0, "blocking_reason"]
    g_score = float(g.loc[0, "confidence_score"])

    if not grade_ok(g_grade, p.min_global_grade) or (pd.notna(g_block) and str(g_block).strip() != ""):
        refusal = f"global_gate_failed(grade={g_grade},blocking_reason={g_block})"
        response_text = (
            "Refusal: KPI is not interpretable under trust rules.\n"
            f"- KPI: {GLOBAL_KPI} (GLOBAL)\n"
            f"- Change point: {cp}\n"
            f"- Global confidence grade: {g_grade} (score={g_score:.3f})\n"
            f"- Blocking reason: {g_block}\n"
        )
        _audit(
            s,
            request_id,
            user_query,
            allowed=False,
            refusal=refusal,
            response_text=response_text,
            event_id=event_id,
            cp=cp,
        )
        print(response_text)
        return

    if robustness < p.min_robustness:
        refusal = f"low_robustness({robustness:.3f})"
        response_text = (
            "Refusal: detected change is not robust enough to narrate under current thresholds.\n"
            f"- Change point: {cp}\n"
            f"- Robustness: {robustness:.3f} (min required {p.min_robustness:.2f})\n"
        )
        _audit(
            s,
            request_id,
            user_query,
            allowed=False,
            refusal=refusal,
            response_text=response_text,
            event_id=event_id,
            cp=cp,
        )
        print(response_text)
        return

    # 3) Pull contributions (base + confidence-weighted)
    base = db.query_df(
        f"""
        SELECT contributor_geo_id, contribution_pct, contribution_value
        FROM fact_contributions
        WHERE event_id='{event_id}' AND method='{METHOD_BASE}'
        ORDER BY contribution_pct DESC
        LIMIT {p.top_n};
        """
    )

    conf = db.query_df(
        f"""
        SELECT contributor_geo_id, contribution_pct, contribution_value
        FROM fact_contributions
        WHERE event_id='{event_id}' AND method='{METHOD_CONF}'
        ORDER BY contribution_pct DESC
        LIMIT {p.top_n};
        """
    )

    # 4) Build evidence-bounded narrative (no causality, no policy)
    response_text = render_narrative(
        global_kpi=GLOBAL_KPI,
        source_id=SOURCE_ID,
        cp=cp,
        drift_type=str(e.get("drift_type") or "unknown"),
        robustness=robustness,
        effect_size=float(e.get("effect_size") or 0.0),
        effect_pct=float(e.get("effect_size_pct") or 0.0),
        p_value=e.get("p_value"),
        confidence_grade=str(g_grade),
        confidence_score=g_score,
        base_contrib=base,
        conf_contrib=conf,
    )

    # 5) Audit row (hash only, optional text storage)
    _audit(
        s,
        request_id,
        user_query,
        allowed=True,
        refusal=None,
        response_text=response_text,
        event_id=event_id,
        cp=cp,
        store_full_text=p.store_full_text,
    )

    print(response_text)


def render_narrative(
    global_kpi: str,
    source_id: str,
    cp: str,
    drift_type: str,
    robustness: float,
    effect_size: float,
    effect_pct: float,
    p_value: Optional[float],
    confidence_grade: str,
    confidence_score: float,
    base_contrib: pd.DataFrame,
    conf_contrib: pd.DataFrame,
) -> str:
    def fmt_top(df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return "No contribution rows available."
        lines = []
        for _, r in df.iterrows():
            geo = r["contributor_geo_id"]
            pct = float(r["contribution_pct"]) * 100.0
            lines.append(f"- {geo}: {pct:.1f}%")
        return "\n".join(lines)

    pv = "NA" if (p_value is None or pd.isna(p_value)) else f"{float(p_value):.4g}"
    cp_s = str(pd.to_datetime(cp).date())

    txt = f"""GLOBAL precipitation drift summary (trust-aware)

            Evidence
            - KPI: {global_kpi} (geo=GLOBAL), source={source_id}
            - Detected change point: {cp}
            - Drift type: {drift_type}
            - Effect size (model units): {effect_size:.3f} | effect %: {effect_pct:.3f}
            - Robustness score: {robustness:.3f}
            - p-value: {pv}
            - Global measurement confidence: grade {confidence_grade} (score={confidence_score:.3f})

            Attribution (population-weighted country deltas)
            Top contributors (base):
            {fmt_top(base_contrib)}

            Top contributors (confidence-weighted, strict trust rule applied):
            {fmt_top(conf_contrib)}

            Trust boundaries / caveats
            - This is an observational change detection + attribution decomposition. No causal claims are made.
            - Attribution is computed from pre/post mean deltas per country multiplied by population (not climate drivers, not policy, not source apportionment).
            - Confidence-weighted attribution down-weights or zeros countries with WARN months in the pre/post window (measurement reliability filter).
            """
    return txt


def _audit(
    s: Settings,
    request_id: str,
    user_query: str,
    allowed: bool,
    refusal: Optional[str],
    response_text: str,
    event_id: Optional[str] = None,
    cp: Optional[str] = None,
    store_full_text: bool = False,
) -> None:
    resolved = f"{GLOBAL_KPI},{COUNTRY_KPI}"
    refs = f"event_id={event_id};cp={cp};source={SOURCE_ID}"

    row = NarrativeAuditRow(
        request_id=request_id,
        user_query=user_query,
        resolved_kpis=resolved,
        allowed=allowed,
        evidence_refs=refs,
        refusal_reason=refusal,
        response_text=response_text,
    )
    write_audit(
        duckdb_path=str(s.duckdb_path),
        row=row,
        store_full_text=store_full_text,
        keep_last_n_text_rows=1000,
    )


if __name__ == "__main__":
    main()
