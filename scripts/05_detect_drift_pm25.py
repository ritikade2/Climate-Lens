"""
pulls PM2.5 + integrity
computes change-points via rolling mean shift 
labels events as true_drift vs artifact_suspected
refuses/blocks drift if recent integrity is weak
writes to fact_drift_events
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd

# repo imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trust_bi.common.settings import Settings  # noqa: E402
from trust_bi.warehouse.duckdb_io import DuckDBClient  # noqa: E402

# -----------------------------
# Configs/constants
# -----------------------------
KPI_ID = "pm25_monthly_mean"
SOURCE_ID = "cams"


@dataclass(frozen=True)
class DriftParams:
    # windows in months
    pre_window: int = 12
    post_window: int = 6

    # minimum separation between detected events (months)
    min_gap: int = 6

    # statistical thresholds
    min_abs_change: float = 5.0          # µg/m3
    min_pct_change: float = 0.10         # 10%
    min_z: float = 2.5                   # standardized shift

    # integrity gates (trust-first)
    recent_months_gate: int = 6
    max_warn_rate_recent: float = 0.34   # if >34% recent months are warn => block/flag
    min_grade_recent: str = "B"          # require A/B in last N months (else block/flag)


def _grade_ok(g: str, min_grade: str) -> bool:
    order = {"A": 4, "B": 3, "C": 2, "D": 1}
    return order.get(g, 0) >= order.get(min_grade, 0)


def _robust_std(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if x.size < 3:
        return float(np.nan)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    rs = 1.4826 * mad
    if rs == 0:
        rs = float(np.std(x, ddof=1)) if x.size >= 2 else float(np.nan)
    return float(rs)


def _integrity_gate(g: pd.DataFrame, p: DriftParams) -> Tuple[bool, str]:
    """
    Returns (allowed, message). If not allowed => classify event as artifact_suspected.
    """
    tail = g.tail(p.recent_months_gate)
    if tail.empty:
        return False, "insufficient_history_for_integrity_gate"

    warn_rate = (tail["data_quality_flag"].fillna("warn") == "warn").mean()

    if warn_rate > p.max_warn_rate_recent:
        return False, f"too_many_warn_months_recent({warn_rate:.2f})"

    grades = tail["confidence_grade"].fillna("D").tolist()
    if not all(_grade_ok(x, p.min_grade_recent) for x in grades):
        return False, f"low_confidence_grade_recent(min={p.min_grade_recent})"

    return True, "ok"


def _detect_best_cp(values: pd.Series, dates: pd.Series, p: DriftParams) -> Optional[dict]:
    n = len(values)
    if n < (p.pre_window + p.post_window + 6):
        return None

    best = None
    arr = values.to_numpy(dtype=float)

    for i in range(p.pre_window, n - p.post_window):
        pre = arr[i - p.pre_window: i]
        post = arr[i: i + p.post_window]

        if np.isnan(pre).mean() > 0.25 or np.isnan(post).mean() > 0.25:
            continue

        pre_mean = float(np.nanmean(pre))
        post_mean = float(np.nanmean(post))
        delta = post_mean - pre_mean

        s = _robust_std(pre)
        if not np.isfinite(s) or s == 0:
            continue

        z = delta / s
        pct = (delta / pre_mean) if pre_mean != 0 else np.nan

        if abs(delta) < p.min_abs_change:
            continue
        if np.isfinite(pct) and abs(pct) < p.min_pct_change:
            continue
        if abs(z) < p.min_z:
            continue

        score = abs(z) * (abs(delta) / max(1.0, abs(pre_mean)))

        if (best is None) or (score > best["score"]):
            best = {
                "cp_idx": i,
                "cp_date": pd.to_datetime(dates.iloc[i]).date(),
                "pre_mean": pre_mean,
                "post_mean": post_mean,
                "effect_size": float(delta),
                "effect_size_pct": float(pct) if np.isfinite(pct) else None,
                "z": float(z),
                "score": float(score),
                "pre_std_robust": float(s),
            }

    return best


def main() -> None:
    s = Settings.load(".env")
    db = DuckDBClient(s.duckdb_path)
    params = DriftParams()

    q = f"""
    SELECT
      f.geo_id,
      f.date,
      f.value,
      f.data_quality_flag,
      i.confidence_grade,
      i.confidence_score
    FROM fact_kpi_timeseries f
    LEFT JOIN fact_integrity_scores i
      ON f.kpi_id = i.kpi_id AND f.geo_id = i.geo_id AND f.date = i.date
    WHERE f.kpi_id = '{KPI_ID}' AND f.source_id = '{SOURCE_ID}'
    ORDER BY f.geo_id, f.date;
    """
    df = db.query_df(q)
    if df.empty:
        raise RuntimeError("No PM2.5 rows found. Run scripts 03 + 04 first.")

    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    events = []

    for geo_id, g in df.groupby("geo_id", sort=False):
        g = g.sort_values("date").reset_index(drop=True)

        allowed, reason = _integrity_gate(g, params)

        best = _detect_best_cp(g["value"], g["date"], params)
        if best is None:
            continue

        # NOTE: min_gap is for multi-event detection; since only keeping 1 best CP per geo,
        # min_gap is inherently satisfied. Keeping param for future multi-event version.

        drift_type = "true_drift" if allowed else "artifact_suspected"
        artifact_evidence = None if allowed else reason

        cp_date = pd.to_datetime(best["cp_date"])
        window_mask = (
            (g["date"] >= (cp_date - pd.DateOffset(months=12)))
            & (g["date"] <= (cp_date + pd.DateOffset(months=6)))
        )
        gg = g.loc[window_mask].copy()
        warn_rate = (gg["data_quality_flag"].fillna("warn") == "warn").mean() if not gg.empty else 1.0
        robustness = float(np.clip(1.0 - warn_rate, 0.0, 1.0))

        created_at = datetime.utcnow()
        events.append(
            {
                "event_id": str(uuid4()),
                "kpi_id": KPI_ID,
                "geo_id": geo_id,
                "detected_at": created_at,
                "cp_start": best["cp_date"],
                "cp_end": best["cp_date"],
                "effect_size": float(best["effect_size"]),
                "effect_size_pct": best["effect_size_pct"],
                "p_value": None,
                "robustness_score": robustness,
                "drift_type": drift_type,
                "artifact_evidence": artifact_evidence,
                "created_at": created_at,
            }
        )

    if not events:
        print("No drift events detected under current thresholds.")
        return

    out = pd.DataFrame(events)

    # must include created_at (table requires it).
    db.upsert_df(out, "fact_drift_events", pk_cols=["event_id"])

    print(f"Detected {len(out):,} drift events for {out['geo_id'].nunique()} geos.")
    print(
        out[
            ["geo_id", "cp_start", "effect_size", "effect_size_pct", "robustness_score", "drift_type"]
        ].head(15).to_string(index=False)
    )


if __name__ == "__main__":
    main()
