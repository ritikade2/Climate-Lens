from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import cdsapi
import yaml


def _load_sources_yaml(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"Missing config file: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class CAMSMonthlyDownloader:
    dataset: str
    request_template: Dict[str, Any]

    @staticmethod
    def from_sources_yaml(repo_root: Path) -> "CAMSMonthlyDownloader":
        cfg = _load_sources_yaml(repo_root / "configs" / "sources.yaml")
        cams = cfg.get("cams", {})
        dataset = cams.get("dataset")
        template = cams.get("request_template")

        if not dataset or not template:
            raise RuntimeError(
                "configs/sources.yaml must contain cams.dataset and cams.request_template."
            )

        return CAMSMonthlyDownloader(dataset=dataset, request_template=template)

    def retrieve_monthly(
        self,
        out_path: Path,
        years: Iterable[int],
        months: Iterable[int],
    ) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        req = dict(self.request_template)
        req["year"] = [str(y) for y in years]
        req["month"] = [f"{m:02d}" for m in months]

        client = cdsapi.Client()
        client.retrieve(self.dataset, req, str(out_path))
        return out_path
