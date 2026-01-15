from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

import pandas as pd
import requests

GISTEMP_GLOBAL_CSV = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts%2BdSST.csv"

MONTH_COLS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
MONTH_NUM = {m: i + 1 for i, m in enumerate(MONTH_COLS)}


@dataclass(frozen = True)
class GISTEMPGlobalMonthly:
    url: str = GISTEMP_GLOBAL_CSV

    def download_raw(self, out_path: Path, timeout: int = 60) -> Path:
        """
        Saves raw CSV snapshot to disk (reproducible).
        Tries requests first; falls back to curl if TLS fails.
        """
        out_path.parent.mkdir(parents = True, exist_ok = True)

        try:
            r = requests.get(self.url, timeout = timeout)
            r.raise_for_status()
            out_path.write_text(r.text, encoding = "utf-8")
            return out_path
        except Exception:
            # curl fallback (macOS TLS)
            result = subprocess.run(
                ["curl", "-L", "--fail", "--silent", "--show-error", self.url],
                check=True,
                capture_output=True,
                text=True,
            )
            out_path.write_text(result.stdout, encoding = "utf-8")
            return out_path

    @staticmethod
    def parse_csv_file(csv_path: Path) -> pd.DataFrame:
        text = csv_path.read_text(encoding = "utf-8")
        lines = text.splitlines()
        skiprows = 1 if lines and "Year" not in lines[0] else 0
        df = pd.read_csv(pd.io.common.StringIO(text), skiprows=skiprows)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    @staticmethod
    def to_long_monthly(df_wide: pd.DataFrame) -> pd.DataFrame:
        if "Year" not in df_wide.columns:
            raise ValueError("Expected 'Year' column in GISTEMP CSV.")

        cols = ["Year"] + [c for c in MONTH_COLS if c in df_wide.columns]
        df = df_wide[cols].copy()

        long_df = df.melt(id_vars=["Year"], value_vars = MONTH_COLS, var_name = "month_name", value_name = "value")
        long_df["value"] = pd.to_numeric(long_df["value"], errors = "coerce")
        long_df["month"] = long_df["month_name"].map(MONTH_NUM).astype("Int64")
        long_df["year"] = pd.to_numeric(long_df["Year"], errors = "coerce").astype("Int64")

        long_df["date"] = pd.to_datetime(
            dict(year = long_df["year"].astype(int), month = long_df["month"].astype(int), day = 1),
            errors="coerce",
        ).dt.date

        return long_df[["date", "year", "month", "value"]].dropna(subset=["date", "value"]).sort_values("date")

    def load_recent_from_file(self, csv_path: Path, start_year: int = 2019, end_year: int = 2025) -> pd.DataFrame:
        wide = self.parse_csv_file(csv_path)
        long_df = self.to_long_monthly(wide)
        return long_df[(long_df["year"] >= start_year) & (long_df["year"] <= end_year)].copy()
