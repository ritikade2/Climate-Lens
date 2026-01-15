from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _load_dotenv_if_present(dotenv_path: str | Path = ".env") -> None:
    """
    Minimal .env loader.
    - Supports KEY=VALUE
    - Strips surrounding quotes "..."
    - Ignores comments and blank lines
    - Does NOT override already-set environment variables
    """
    p = Path(dotenv_path)
    if not p.exists():
        return

    for raw_line in p.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()

        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]

        if key and key not in os.environ:
            os.environ[key] = val


def _require(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(
            f"Missing required environment variable: {name}\n"
            f"Create a .env file (not committed) or export it in your shell."
        )
    return v


@dataclass(frozen=True)
class Settings:
    # keys
    openai_api_key: Optional[str]   # optional for now; ingestion doesn't need it
    duckdb_path: Path

    # project paths
    project_root: Path
    data_dir: Path
    raw_dir: Path
    curated_dir: Path

    @staticmethod
    def load(dotenv_path: str | Path = ".env") -> "Settings":
        _load_dotenv_if_present(dotenv_path)

        # OpenAI key not required for ingestion
        openai_api_key = os.getenv("OPENAI_API_KEY")

        duckdb_path_str = os.getenv("DUCKDB_PATH", "data/warehouse/duckdb/trust_bi.duckdb")
        project_root = Path(__file__).resolve().parents[3]  # src/trust_bi/common/settings.py -> repo root
        duckdb_path = (project_root / duckdb_path_str).resolve()

        data_dir = project_root / "data"
        raw_dir = data_dir / "raw"
        curated_dir = data_dir / "curated"

        return Settings(
            openai_api_key=openai_api_key,
            duckdb_path=duckdb_path,
            project_root=project_root,
            data_dir=data_dir,
            raw_dir=raw_dir,
            curated_dir=curated_dir,
        )
