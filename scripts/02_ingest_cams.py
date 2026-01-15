# Ingest Air quality data from CAMS
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trust_bi.common.settings import Settings  # noqa: E402
from trust_bi.ingest.cams import CAMSMonthlyDownloader  # noqa: E402


def main() -> None:
    s = Settings.load(".env")

    years = list(range(2019, 2025)) # 2019-2024 inclusive. 2025 there may be lag.
    months = list(range(1, 13)) # all months  

    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_file = s.raw_dir / "cams" / f"cams_pm25_monthly_{years[0]}_{years[-1]}_{stamp}.nc.zip"

    dl = CAMSMonthlyDownloader.from_sources_yaml(REPO_ROOT)
    path = dl.retrieve_monthly(out_file, years= years, months= months)

    print(f"CAMS raw saved: {path}")
    print("Note: this is CAMS reanalysis months. Adding second CAMS product for 2025 data separately.")


if __name__ == "__main__":
    main()
