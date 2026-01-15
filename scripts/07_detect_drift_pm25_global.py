"""
Detect drift in GLOBAL pop-weighted monthly PM2.5 (pm25_monthly_mean_global_popw).

- Reads: fact_kpi_timeseries (GLOBAL, monthly)
- Detects: best step-change OR best trend-change (slope break)
- Writes: fact_drift_events (one row, if detected)

Notes:
- Uses data_quality_flag to compute a simple robustness_score from recent WARN rate.
- p_value intentionally None.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

# repo imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trust_bi.common.settings import Settings 
from trust_bi.warehouse.duckdb_io import DuckDBClient 

## -----------------------------
# Config / constants
# -----------------------------
KPI_ID = "pm25_monthly_mean_global_popw"
SOURCE_ID = "cams"
GEO_ID = "GLOBAL"

@dataclass
class DriftParams:
    # Step-change detection (months)
    pre_window: int = 18
    post_window: int = 6
    min_abs_change: float = 1.0 # global shifts can be small
    min_pct_change: float = 0.02 # 2%
    min_z: float = 1.8
    # Trend-change detection (slop break)
    trend_window: int = 24 # months before/after candidate
    min_slope_change_per_year: float = 1.0 # ug/m3 per year change in slope



def robust_std(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    s = 1.4826 * mad
    if s == 0:
        s = float(np.std(x, ddof=1)) if x.size >= 2 else float("nan")
    return float(s)



def detect_best_step(values: pd.Series, dates: pd.Series, p: DriftParams) -> Optional[Dict[str, Any]]:
    n = len(values)
    if n < (p.pre_window + p.post_window + 6):
        return None

    arr = values.to_numpy(dtype=float)
    best = None

    for i in range(p.pre_window, n - p.post_window):
        pre = arr[i - p.pre_window : i]
        post = arr[i : i + p.post_window]

        if np.isnan(pre).mean() > 0.10 or np.isnan(post).mean() > 0.10:
            continue

        pre_mean = float(np.nanmean(pre))
        post_mean = float(np.nanmean(post))
        delta = post_mean - pre_mean

        s_pre = robust_std(pre)
        if not np.isfinite(s_pre) or s_pre == 0:
            continue

        z = delta / s_pre
        pct = (delta / pre_mean) if pre_mean != 0 else np.nan

        if abs(delta) < p.min_abs_change:
            continue
        if np.isfinite(pct) and abs(pct) < p.min_pct_change:
            continue
        if abs(z) < p.min_z:
            continue

        score = abs(z) * abs(delta)

        if best is None or score > best["score"]:
            best = {
                "type": "step_change",
                "cp_idx": i,
                "cp_date": pd.to_datetime(dates.iloc[i]).date(),
                "pre_mean": pre_mean,
                "post_mean": post_mean,
                "effect_size": float(delta),
                "effect_size_pct": float(pct) if np.isfinite(pct) else None,
                "stat": float(z),
                "score": float(score),
            }

    return best



def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 6:
        return float("nan")

    x0 = x - x.mean()
    denom = float(np.sum(x0 * x0))
    if denom == 0:
        return float("nan")

    b = float(np.sum(x0 * (y - y.mean())) / denom)  # slope per x unit
    return b


def detect_best_trend(values: pd.Series, dates: pd.Series, p: DriftParams) -> Optional[Dict[str, Any]]:
    n = len(values)
    w = p.trend_window
    if n < (2 * w + 6):
        return None

    y = values.to_numpy(dtype=float)
    x = np.arange(n, dtype=float)  # months index

    best = None

    for i in range(w, n - w):
        y1 = y[i - w : i]
        y2 = y[i : i + w]
        x1 = x[i - w : i]
        x2 = x[i : i + w]

        if np.isnan(y1).mean() > 0.10 or np.isnan(y2).mean() > 0.10:
            continue

        b1 = _ols_slope(x1, y1)  # per month
        b2 = _ols_slope(x2, y2)  # per month
        if not np.isfinite(b1) or not np.isfinite(b2):
            continue

        slope_change_per_month = b2 - b1
        slope_change_per_year = slope_change_per_month * 12.0

        if abs(slope_change_per_year) < p.min_slope_change_per_year:
            continue

        score = abs(slope_change_per_year)

        if best is None or score > best["score"]:
            best = {
                "type": "trend_change",
                "cp_idx": i,
                "cp_date": pd.to_datetime(dates.iloc[i]).date(),
                "effect_size": float(slope_change_per_year),  # slope change per year
                "effect_size_pct": None,
                "stat": float(slope_change_per_year),
                "score": float(score),
                "pre_slope_per_year": float(b1 * 12.0),
                "post_slope_per_year": float(b2 * 12.0),
            }

    return best



def main() -> None:
    s = Settings.load(".env")
    db = DuckDBClient(s.duckdb_path)
    p = DriftParams()

    q = f"""
    SELECT
      date,
      value,
      data_quality_flag
    FROM fact_kpi_timeseries
    WHERE kpi_id = '{KPI_ID}'
      AND source_id = '{SOURCE_ID}'
      AND geo_id = '{GEO_ID}'
    ORDER BY date;
    """
    df = db.query_df(q)
    if df.empty:
        raise RuntimeError("No global PM2.5 KPI rows found. Run scripts/06_build_kpi_pm25_global.py first.")

    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    best_step = detect_best_step(df["value"], df["date"], p)
    best_trend = detect_best_trend(df["value"], df["date"], p)

    if best_step and best_trend:
        best = best_step if best_step["score"] >= best_trend["score"] else best_trend
    else:
        best = best_step or best_trend

    if best is None:
        print("No GLOBAL drift event detected (step or trend) under current thresholds.")
        return

    # Robustness from last 6 months WARN rate
    tail = df.tail(6).copy()
    warn_rate = (tail["data_quality_flag"].fillna("warn") == "warn").mean()
    robustness = float(np.clip(1.0 - warn_rate, 0.0, 1.0))

    drift_type = "true_drift" if warn_rate <= 0.34 else "artifact_suspected"
    artifact_evidence = f"type={best['type']}"
    
    if drift_type != "true_drift":
        artifact_evidence = f"warn_rate_recent({warn_rate:.2f});" + artifact_evidence

    out = pd.DataFrame(
        [
            {
                "event_id": str(uuid4()),
                "kpi_id": KPI_ID,
                "geo_id": GEO_ID,
                "detected_at": datetime.utcnow(),
                "cp_start": best["cp_date"],
                "cp_end": best["cp_date"],
                "effect_size": float(best["effect_size"]),
                "effect_size_pct": best.get("effect_size_pct"),
                "p_value": None,
                "robustness_score": robustness,
                "drift_type": drift_type,
                "artifact_evidence": artifact_evidence,
                "created_at": datetime.utcnow(),
            }
        ]
    )

    db.upsert_df(out.drop(columns=["created_at"]), "fact_drift_events", pk_cols=["event_id"])

    
    print("GLOBAL drift event written to fact_drift_events:")
    cols = ["geo_id", "cp_start", "effect_size", "effect_size_pct", "robustness_score", "drift_type"]
    print(out[cols].to_string(index=False))

    if best["type"] == "trend_change":
        print(f"Pre slope (ug/m3 per year):  {best['pre_slope_per_year']:.3f}")
        print(f"Post slope (ug/m3 per year): {best['post_slope_per_year']:.3f}")


if __name__ == "__main__":
    main()