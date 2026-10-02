"""Persistence and query layer built on SQLAlchemy (SQLite for local dev, PostgreSQL for production)."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, Text, create_engine, inspect, select, text
from sqlalchemy.engine import Engine

from app.config import Settings, get_settings
from app.core.profiling import detect_relationships, semantic_type
from .sql_guard import validate_sql

META = MetaData()
DATASETS = Table(
    "if_datasets", META,
    Column("name", String(128), primary_key=True),
    Column("rows", Integer), Column("columns", Integer),
    Column("cleaning_log", Text), Column("source_file", String(256)),
    Column("created_at", DateTime),
)
GLOSSARY = Table(
    "if_glossary", META,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("title", String(256)), Column("body", Text), Column("created_at", DateTime),
)


class DataStore:
    def __init__(self, settings: Optional[Settings] = None, engine: Optional[Engine] = None):
        self.settings = settings or get_settings()
        url = self.settings.database_url
        if engine is None:
            if url.startswith("sqlite:///") and ":memory:" not in url:
                Path(url.replace("sqlite:///", "", 1)).parent.mkdir(parents=True, exist_ok=True)
            engine = create_engine(url, future=True)
        self.engine = engine
        META.create_all(self.engine)
        self._rel_cache: Optional[List[Dict[str, Any]]] = None

    # ------------------------------------------------------------------ datasets
    def register_dataset(self, name: str, df: pd.DataFrame, cleaning_log: Optional[List[Dict]] = None,
                         source_file: str = "") -> None:
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", name):
            raise ValueError(f"Invalid table name: {name}")
        if name.startswith("if_"):
            raise ValueError("Table names starting with 'if_' are reserved")
        df.to_sql(name, self.engine, if_exists="replace", index=False, chunksize=5000)
        with self.engine.begin() as conn:
            conn.execute(DATASETS.delete().where(DATASETS.c.name == name))
            conn.execute(DATASETS.insert().values(
                name=name, rows=len(df), columns=df.shape[1], cleaning_log=json.dumps(cleaning_log or [], default=str),
                source_file=source_file, created_at=datetime.now(timezone.utc).replace(tzinfo=None)))
        self._rel_cache = None

    def list_datasets(self) -> List[Dict[str, Any]]:
        with self.engine.connect() as conn:
            rows = conn.execute(select(DATASETS).order_by(DATASETS.c.created_at)).mappings().all()
        return [{**dict(r), "cleaning_log": json.loads(r["cleaning_log"] or "[]"),
                 "created_at": r["created_at"].isoformat() if r["created_at"] else None} for r in rows]

    def dataset_names(self) -> List[str]:
        return [d["name"] for d in self.list_datasets()]

    def get_dataframe(self, name: str) -> pd.DataFrame:
        if name not in self.dataset_names():
            raise KeyError(f"Dataset '{name}' not found")
        df = pd.read_sql_table(name, self.engine)
        return df

    def get_cleaning_log(self, name: str) -> List[Dict]:
        for d in self.list_datasets():
            if d["name"] == name:
                return d["cleaning_log"]
        raise KeyError(name)

    def drop_dataset(self, name: str) -> None:
        if name not in self.dataset_names():
            raise KeyError(name)
        with self.engine.begin() as conn:
            conn.execute(text(f'DROP TABLE IF EXISTS "{name}"'))
            conn.execute(DATASETS.delete().where(DATASETS.c.name == name))
        self._rel_cache = None

    # ------------------------------------------------------------------ relationships & schema
    def relationships(self, refresh: bool = False) -> List[Dict[str, Any]]:
        if self._rel_cache is None or refresh:
            frames = {n: self.get_dataframe(n) for n in self.dataset_names()}
            self._rel_cache = detect_relationships(frames) if len(frames) > 1 else []
        return self._rel_cache

    def schema(self, sample_rows: int = 3) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        insp = inspect(self.engine)
        for n in self.dataset_names():
            cols = []
            sample = pd.read_sql_query(f'SELECT * FROM "{n}" LIMIT {sample_rows}', self.engine)
            for c in insp.get_columns(n):
                cols.append({"name": c["name"], "type": str(c["type"]),
                             "sample": [str(v) for v in sample[c["name"]].tolist()] if c["name"] in sample else []})
            out[n] = cols
        return out

    def schema_text(self) -> str:
        lines = [f"SQL dialect: {self.engine.dialect.name}"]
        for t, cols in self.schema().items():
            lines.append(f"TABLE {t}(" + ", ".join(f"{c['name']} {c['type']}" for c in cols) + ")")
        for r in self.relationships():
            lines.append(f"JOIN {r['left_table']}.{r['left_column']} = {r['right_table']}.{r['right_column']}  ({r['cardinality']})")
        return "\n".join(lines)

    def joined_frame(self, fact: Optional[str] = None, tables: Optional[List[str]] = None) -> pd.DataFrame:
        """Denormalise `fact` with every table reachable through many-to-one / one-to-one relationships."""
        names = tables or self.dataset_names()
        if not names:
            raise KeyError("No datasets loaded")
        counts = {d["name"]: d["rows"] for d in self.list_datasets() if d["name"] in names}
        fact = fact or max(counts, key=counts.get)
        df = self.get_dataframe(fact)
        joined = {fact}
        rels = [r for r in self.relationships() if r["left_table"] in names and r["right_table"] in names]
        progress = True
        while progress:
            progress = False
            for r in rels:
                for a, ac, b, bc, card in ((r["left_table"], r["left_column"], r["right_table"], r["right_column"], r["cardinality"]),
                                           (r["right_table"], r["right_column"], r["left_table"], r["left_column"],
                                            {"one-to-many": "many-to-one", "many-to-one": "one-to-many"}.get(r["cardinality"], "one-to-one"))):
                    # a is already joined; b must be the unique ("one") side
                    if a in joined and b not in joined and card in ("many-to-one", "one-to-one") and ac in df.columns:
                        other = self.get_dataframe(b)
                        other = other.rename(columns={c: f"{b}_{c}" for c in other.columns if c in df.columns and c != bc})
                        df = df.merge(other, left_on=ac, right_on=bc, how="left", suffixes=("", f"_{b}"))
                        if bc != ac and bc in df.columns:
                            df = df.drop(columns=[bc])
                        joined.add(b)
                        progress = True
        return df

    # ------------------------------------------------------------------ querying
    def run_sql(self, sql: str, limit: Optional[int] = None) -> pd.DataFrame:
        limit = min(limit or self.settings.max_result_rows, self.settings.max_result_rows)
        safe = validate_sql(sql, self.dataset_names())
        wrapped = f"SELECT * FROM ({safe}) AS _q LIMIT {int(limit)}"
        with self.engine.connect() as conn:
            if self.engine.dialect.name == "postgresql":
                conn.execute(text("SET TRANSACTION READ ONLY"))
                conn.execute(text("SET LOCAL statement_timeout = 15000"))
            return pd.read_sql_query(text(wrapped), conn)

    # ------------------------------------------------------------------ glossary (business definitions)
    def add_glossary(self, title: str, body: str) -> int:
        with self.engine.begin() as conn:
            r = conn.execute(GLOSSARY.insert().values(title=title, body=body, created_at=datetime.now(timezone.utc).replace(tzinfo=None)))
            return int(r.inserted_primary_key[0])

    def list_glossary(self) -> List[Dict[str, Any]]:
        with self.engine.connect() as conn:
            return [dict(r) for r in conn.execute(select(GLOSSARY.c.id, GLOSSARY.c.title, GLOSSARY.c.body)).mappings().all()]
