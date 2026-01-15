from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
import duckdb
import pandas as pd

@dataclass(frozen = True)
class DuckDBClient:
    path:Path

    def connect(self) -> duckdb.DuckDBPyConnection:
        self.path.parent.mkdir(parents = True, exist_ok = True)
        con = duckdb.connect(str(self.path))
        con.execute("PRAGMA threads=4;")
        return con
    
    def query_df(self, sql: str, params: Optional[Iterable[Any]] = None) -> pd.DataFrame:
        con = self.connect()
        try:
            if params is None:
                return con.execute(sql).df()
            return con.execute(sql, params).df()
        finally:
            con.close()
    
    def execute(self, sql: str, params: Optional[Iterable[Any]] = None) -> None:
        con = self.connect()
        try:
            if params is None:
                con.execute(sql)
            else:
                con.execute(sql, params)
        finally:
            con.close()
    
    def upsert_df( self, df: pd.DataFrame, table: str, pk_cols: list[str], created_at_col: str | None = "created_at",) -> None:
        """
        Write staging -> delete matching PKs -> insert.
        If created_at_col is None, no timestamp column is added.
        """
        if df.empty:
            return

        if created_at_col is not None and created_at_col not in df.columns:
            df = df.copy()
            df[created_at_col] = pd.Timestamp.utcnow()

        staging = f"stg_{table}"
        con = self.connect()
        try:
            con.register("incoming_df", df)
            con.execute(f"CREATE OR REPLACE TEMP TABLE {staging} AS SELECT * FROM incoming_df;")

            join_cond = " AND ".join([f"t.{c} = s.{c}" for c in pk_cols])
            con.execute(
                f"""
                DELETE FROM {table} AS t
                USING {staging} AS s
                WHERE {join_cond};
                """
            )
            con.execute(f"INSERT INTO {table} SELECT * FROM {staging};")
        finally:
            con.close()
