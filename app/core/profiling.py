"""Dataset profiling: semantic types, quality score, distributions and cross-dataset relationships."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pandas.api import types as pdt


def semantic_type(s: pd.Series) -> str:
    n = len(s)
    nun = s.nunique(dropna=True)
    if pdt.is_bool_dtype(s):
        return "boolean"
    if pdt.is_datetime64_any_dtype(s):
        return "datetime"
    if pdt.is_numeric_dtype(s):
        if nun <= 2:
            return "boolean"
        name = str(s.name).lower()
        if (name.endswith("_id") or name == "id") and nun > 0.5 * max(n, 1):
            return "identifier"
        if pdt.is_integer_dtype(s) and nun == n and n > 20:
            return "identifier"
        return "numeric"
    if nun == 0:
        return "empty"
    if nun <= max(20, 0.05 * n) :
        return "categorical"
    if nun > 0.9 * n:
        return "identifier" if s.astype(str).str.len().mean() < 40 else "text"
    return "text" if s.astype(str).str.len().mean() > 40 else "categorical"


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
        return None if np.isnan(f) or np.isinf(f) else round(f, 6)
    except (TypeError, ValueError):
        return None


def profile_column(s: pd.Series) -> Dict[str, Any]:
    kind = semantic_type(s)
    n = len(s)
    p: Dict[str, Any] = {
        "name": s.name, "dtype": str(s.dtype), "semantic_type": kind,
        "missing": int(s.isna().sum()), "missing_pct": round(float(s.isna().mean() * 100), 2) if n else 0.0,
        "unique": int(s.nunique(dropna=True)),
    }
    if kind == "numeric":
        d = s.dropna().astype(float)
        if len(d):
            q1, q3 = d.quantile(0.25), d.quantile(0.75)
            p.update(mean=_num(d.mean()), std=_num(d.std()), min=_num(d.min()), q1=_num(q1),
                     median=_num(d.median()), q3=_num(q3), max=_num(d.max()),
                     skew=_num(d.skew()) if len(d) > 2 else None,
                     zeros=int((d == 0).sum()), negatives=int((d < 0).sum()))
            iqr = q3 - q1
            p["iqr_outliers"] = int(((d < q1 - 1.5 * iqr) | (d > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
    elif kind == "datetime":
        d = s.dropna()
        if len(d):
            p.update(min=str(d.min()), max=str(d.max()), span_days=int((d.max() - d.min()).days))
    elif kind in {"categorical", "boolean", "identifier", "text"}:
        vc = s.value_counts(dropna=True).head(5)
        p["top_values"] = [{"value": str(k), "count": int(v), "pct": round(v / max(n, 1) * 100, 2)} for k, v in vc.items()]
    return p


def profile_dataframe(df: pd.DataFrame, name: str = "dataset") -> Dict[str, Any]:
    cols = [profile_column(df[c]) for c in df.columns]
    n, m = df.shape
    dup = int(df.duplicated().sum())
    missing_cells = int(df.isna().sum().sum())
    completeness = 1 - missing_cells / max(n * m, 1)
    uniqueness = 1 - dup / max(n, 1)
    outlier_rate = np.mean([c.get("iqr_outliers", 0) / max(n, 1) for c in cols if c["semantic_type"] == "numeric"] or [0])
    quality = round(100 * (0.6 * completeness + 0.25 * uniqueness + 0.15 * (1 - min(outlier_rate * 5, 1))), 1)
    return {
        "name": name, "rows": n, "columns": m, "duplicate_rows": dup, "missing_cells": missing_cells,
        "memory_kb": round(float(df.memory_usage(deep=True).sum()) / 1024, 1),
        "quality_score": quality, "column_profiles": cols,
        "numeric_columns": [c["name"] for c in cols if c["semantic_type"] == "numeric"],
        "categorical_columns": [c["name"] for c in cols if c["semantic_type"] in {"categorical", "boolean"}],
        "datetime_columns": [c["name"] for c in cols if c["semantic_type"] == "datetime"],
        "identifier_columns": [c["name"] for c in cols if c["semantic_type"] == "identifier"],
    }


def detect_relationships(frames: Dict[str, pd.DataFrame], min_overlap: float = 0.6) -> List[Dict[str, Any]]:
    """Infer join keys across datasets: same column name (or *_id) with strong value overlap and a unique side."""
    rels: List[Dict[str, Any]] = []
    for (a, da), (b, db) in combinations(frames.items(), 2):
        for ca in da.columns:
            for cb in db.columns:
                if ca != cb and not (ca.endswith("_id") and cb.endswith("_id") and ca.split("_")[0] == cb.split("_")[0]):
                    continue
                sa, sb = da[ca].dropna(), db[cb].dropna()
                if sa.empty or sb.empty or pdt.is_float_dtype(sa) or pdt.is_datetime64_any_dtype(sa):
                    continue
                if sa.nunique() < 2 or sb.nunique() < 2:
                    continue
                va, vb = set(sa.unique()), set(sb.unique())
                overlap = len(va & vb) / max(min(len(va), len(vb)), 1)
                if overlap < min_overlap:
                    continue
                a_unique, b_unique = sa.is_unique, sb.is_unique
                if a_unique and b_unique:
                    card = "one-to-one"
                elif a_unique:
                    card = "one-to-many"
                elif b_unique:
                    card = "many-to-one"
                else:
                    continue  # many-to-many joins are unsafe for aggregation
                rels.append({"left_table": a, "left_column": ca, "right_table": b, "right_column": cb,
                             "cardinality": card, "overlap": round(overlap, 3)})
    return rels
