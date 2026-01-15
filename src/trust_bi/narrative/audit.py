from __future__ import annotations
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import duckdb


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

@dataclass(frozen = True)
class NarrativeAuditRow:
    request_id: str
    user_query: str
    resolved_kpis: str
    allowed: bool
    evidence_refs: str
    refusal_reason: Optional[str] = None
    response_text: Optional[str] = None # only hashed, not stored.
    

def write_audit(
        duckdb_path: str,
        row: NarrativeAuditRow,
        store_full_text: bool = False,
        keep_last_n_text_rows: int = 1000) -> None:
    """
    Always writes to narrative_audit_log (small). 
    Optionally writes full narrative text to fact_narratives (capped).
    """
    now = datetime.utcnow()
    text_hash = sha256_text(row.response_text or "")

    con = duckdb.connect(duckdb_path)
    try:
        # 1) Audit row
        con.execute(
            """
            INSERT INTO narrative_audit_log
              (request_id, request_ts, user_query, resolved_kpis, allowed, evidence_refs, refusal_reason, generated_text_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(request_id) DO UPDATE SET
              request_ts = excluded.request_ts,
              user_query = excluded.user_query,
              resolved_kpis = excluded.resolved_kpis,
              allowed = excluded.allowed,
              evidence_refs = excluded.evidence_refs,
              refusal_reason = excluded.refusal_reason,
              generated_text_hash = excluded.generated_text_hash
            """,
            [
                row.request_id,
                now,
                row.user_query,
                row.resolved_kpis,
                row.allowed,
                row.evidence_refs,
                row.refusal_reason,
                text_hash,
            ],
        )

        if store_full_text and row.response_text:
            con.execute(
                """
                INSERT INTO fact_narratives (request_id, created_at, narrative_text)
                VALUES (?, ?, ?)
                ON CONFLICT(request_id) DO UPDATE SET
                  created_at = excluded.created_at,
                  narrative_text = excluded.narrative_text
                """,
                [row.request_id, now, row.response_text],
            )

            # data retention cap: keep newest N
            con.execute(
                """
                DELETE FROM fact_narratives
                WHERE request_id IN (
                  SELECT request_id FROM fact_narratives
                  ORDER BY created_at DESC
                  OFFSET ?
                )
                """,
                [keep_last_n_text_rows],
            )
    finally:
        con.close()