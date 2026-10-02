"""Ingestion of CSV / Excel files into pandas DataFrames with sanitised names."""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Dict, Union

import pandas as pd

SUPPORTED = {".csv", ".tsv", ".txt", ".xlsx", ".xlsm", ".xls"}


def snake(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip())
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", s)
    s = re.sub(r"_+", "_", s).strip("_").lower()
    if not s:
        s = "col"
    if s[0].isdigit():
        s = f"c_{s}"
    return s


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    seen: Dict[str, int] = {}
    cols = []
    for c in df.columns:
        base = snake(c)
        n = seen.get(base, 0)
        seen[base] = n + 1
        cols.append(base if n == 0 else f"{base}_{n}")
    out = df.copy()
    out.columns = cols
    return out


def table_name(filename: str, sheet: str | None = None) -> str:
    stem = Path(filename).stem
    name = snake(stem)
    if sheet:
        name = f"{name}_{snake(sheet)}"
    return name


def load_file(filename: str, content: Union[bytes, str, Path]) -> Dict[str, pd.DataFrame]:
    """Return {table_name: DataFrame}. Excel workbooks yield one table per non-empty sheet."""
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED)}")
    if isinstance(content, (str, Path)):
        content = Path(content).read_bytes()
    buf = io.BytesIO(content)
    tables: Dict[str, pd.DataFrame] = {}
    if ext in {".csv", ".txt", ".tsv"}:
        sep = "\t" if ext == ".tsv" else None
        try:
            df = pd.read_csv(buf, sep=sep, engine="python", encoding="utf-8-sig")
        except UnicodeDecodeError:
            buf.seek(0)
            df = pd.read_csv(buf, sep=sep, engine="python", encoding="latin-1")
        tables[table_name(filename)] = df
    else:
        sheets = pd.read_excel(buf, sheet_name=None)
        for sheet, df in sheets.items():
            if df.empty:
                continue
            tables[table_name(filename, sheet if len(sheets) > 1 else None)] = df
    if not tables:
        raise ValueError("No data found in file.")
    return {k: sanitize_columns(v.dropna(how="all").dropna(axis=1, how="all")) for k, v in tables.items()}
