# scripts/07b_detect_drift_global_popw_kpis.py
from __future__ import annotations

import sys
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

# repo imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trust_bi.common.settings import Settings  # noqa: E402
from trust_bi.warehouse.duckdb_io import DuckDBClient  # noqa: E402


# -----------------------------
# Config
# -----------------------------
GEO_ID = "GLOBAL"

# (kpi_id, source_id) -> source_id used only to filter fact_kpi_timeseries
TARGETS: List[Tuple[str, str]] = [
    ("pm25_monthly_mean_global_popw", "cams"),
    ("temp_country_monthly_mean_global_popw", "openmeteo"),
    ("precip_country_monthly_sum_global_popw", "openmeteo"),
]

PRE_WINDOW_MONTHS = 12
POST_WINDOW_MONTHS = 12
ROBUST_RECENT_MONTHS = 18

MIN_ABS_PCT_FOR_TRUE_DRIFT = 0.10
MIN_ROBUSTNESS_FOR_TRUE_DRIFT = 0.80


# -----------------------------
# Helpers
# -----------------------------
def month_floor(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)

def month_add(ts: pd.Timestamp, n: int) -> pd.Timestamp:
    # safe month add via period
    return (ts.to_period("M") + n).to_timestamp()

def make_event_id(kpi_id: str, geo_id: str, cp_start: pd.Timestamp) -> str:
    key = f"{kpi_id}|{geo_id}|{cp_start.date().isoformat()}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()  # stable, short enough


def fetch_series(db: DuckDBClient, *, kpi_id: str, source_id: str, geo_id: str) -> pd.DataFrame:
    df = db.query_df(
        f"""
        SELECT
          date::DATE AS date,
          value::DOUBLE AS value,
          data_quality_flag
        FROM fact_kpi_timeseries
        WHERE kpi_id = '{kpi_id}'
          AND source_id = '{source_id}'
          AND geo_id = '{geo_id}'
        ORDER BY 1;
        """
    )
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["data_quality_flag"] = df["data_quality_flag"].astype(str)
    df = df.dropna(subset=["date"]).copy()
    return df


@dataclass
class DriftResult:
    cp_start: pd.Timestamp
    cp_end: pd.Timestamp
    effect_size: float
    effect_size_pct: Optional[float]
    robustness_score: float
    drift_type: str
    evidence: str


def compute_robustness(monthly: pd.DataFrame) -> float:
    if monthly.empty:
        return 0.0
    tail = monthly.tail(ROBUST_RECENT_MONTHS)
    if tail.empty:
        return 0.0
    pass_rate = (tail["data_quality_flag"].str.lower() == "pass").mean()
    return float(pass_rate)


def scan_changepoint(df: pd.DataFrame) -> Optional[DriftResult]:
    if df.empty:
        return None

    d = df.copy().sort_values("date")
    d["date"] = d["date"].map(month_floor)

    # ensure monthly unique rows
    d = d.groupby("date", as_index=False).agg(
        value=("value", "mean"),
        data_quality_flag=("data_quality_flag", "max"),
    )

    n = len(d)
    min_needed = PRE_WINDOW_MONTHS + POST_WINDOW_MONTHS + 1
    if n < min_needed:
        return None

    dates = d["date"].tolist()
    vals = d["value"].to_numpy(dtype=float)

    best = None
    best_abs = -1.0

    for i in range(PRE_WINDOW_MONTHS, n - POST_WINDOW_MONTHS):
        pre = vals[i - PRE_WINDOW_MONTHS : i]
        post = vals[i : i + POST_WINDOW_MONTHS]

        if np.all(np.isnan(pre)) or np.all(np.isnan(post)):
            continue

        pre_mean = float(np.nanmean(pre))
        post_mean = float(np.nanmean(post))
        eff = post_mean - pre_mean

        abs_eff = abs(eff)
        if abs_eff > best_abs:
            best_abs = abs_eff
            pct = None
            if pre_mean != 0 and not np.isnan(pre_mean):
                pct = float(eff / abs(pre_mean))
            best = (dates[i], eff, pct, pre_mean, post_mean)

    if best is None:
        return None

    cp_start, eff, pct, pre_mean, post_mean = best
    cp_start = pd.to_datetime(cp_start)
    cp_end = month_add(cp_start, POST_WINDOW_MONTHS)  # end marker (start of month after post window)

    robustness = compute_robustness(d)

    is_true = (
        (pct is not None)
        and (abs(pct) >= MIN_ABS_PCT_FOR_TRUE_DRIFT)
        and (robustness >= MIN_ROBUSTNESS_FOR_TRUE_DRIFT)
    )
    drift_type = "true_drift" if is_true else "warn_drift"

    evidence = (
        f"scan(pre={PRE_WINDOW_MONTHS}m, post={POST_WINDOW_MONTHS}m); "
        f"pre_mean={pre_mean:.6f}, post_mean={post_mean:.6f}; "
        f"robust_recent={ROBUST_RECENT_MONTHS}m"
    )

    return DriftResult(
        cp_start=cp_start,
        cp_end=cp_end,
        effect_size=float(eff),
        effect_size_pct=float(pct) if pct is not None else None,
        robustness_score=float(robustness),
        drift_type=drift_type,
        evidence=evidence,
    )


def ensure_drift_table_exists(duckdb_path: Path) -> None:
    # In your repo this already exists, but keeping it defensive (schema must match)
    con = duckdb.connect(str(duckdb_path))
    try:
        con.execute(
            """
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
            """
        )
    finally:
        con.close()


def upsert_drift_event(db: DuckDBClient, *, kpi_id: str, geo_id: str, r: DriftResult) -> pd.DataFrame:
    event_id = make_event_id(kpi_id, geo_id, r.cp_start)

    out = pd.DataFrame(
        [
            {
                "event_id": event_id,
                "kpi_id": kpi_id,
                "geo_id": geo_id,
                "detected_at": datetime.utcnow(),
                "cp_start": r.cp_start.date(),
                "cp_end": r.cp_end.date(),
                "effect_size": r.effect_size,
                "effect_size_pct": r.effect_size_pct,
                "p_value": None,  # v1: None
                "robustness_score": r.robustness_score,
                "drift_type": r.drift_type,
                "artifact_evidence": r.evidence,
                "created_at": datetime.utcnow(),
            }
        ]
    )

    # Upsert via event_id (PK)
    db.upsert_df(out, "fact_drift_events", pk_cols=["event_id"])
    return out


def main() -> None:
    s = Settings.load(".env")
    db = DuckDBClient(s.duckdb_path)

    print(f"DuckDB path: {s.duckdb_path}")
    ensure_drift_table_exists(s.duckdb_path)

    for (kpi_id, source_id) in TARGETS:
        print(f"\n=== Detecting drift: {kpi_id} ({source_id}) @ {GEO_ID} ===")
        df = fetch_series(db, kpi_id=kpi_id, source_id=source_id, geo_id=GEO_ID)
        if df.empty:
            print("No data found. Skipping.")
            continue

        res = scan_changepoint(df)
        if res is None:
            print("Not enough data / no valid changepoint found. Skipping.")
            continue

        written = upsert_drift_event(db, kpi_id=kpi_id, geo_id=GEO_ID, r=res)
        row = written.iloc[0].to_dict()

        print("Drift event written to fact_drift_events:")
        print(
            pd.DataFrame(
                [
                    {
                        "event_id": row["event_id"],
                        "kpi_id": row["kpi_id"],
                        "geo_id": row["geo_id"],
                        "cp_start": row["cp_start"],
                        "cp_end": row["cp_end"],
                        "effect_size": row["effect_size"],
                        "effect_size_pct": row["effect_size_pct"],
                        "robustness_score": row["robustness_score"],
                        "drift_type": row["drift_type"],
                    }
                ]
            ).to_string(index=False)
        )


if __name__ == "__main__":
    main()
