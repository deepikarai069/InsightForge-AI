"""Automatic KPI detection and computation."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from .trends import aggregate_series, latest_period_change

_KPI_HINTS = [
    ("revenue", "sum"), ("sales", "sum"), ("amount", "sum"), ("profit", "sum"), ("cost", "sum"),
    ("quantity", "sum"), ("units", "sum"), ("spend", "sum"), ("price", "mean"), ("rating", "mean"),
    ("score", "mean"), ("discount", "mean"), ("margin", "mean"), ("duration", "mean"),
]


def _rate_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_bool_dtype(df[c]) or
            (pd.api.types.is_numeric_dtype(df[c]) and df[c].dropna().isin([0, 1]).all() and df[c].nunique() == 2)]


def compute_kpis(df: pd.DataFrame, date_col: Optional[str] = None, max_kpis: int = 8) -> List[Dict[str, Any]]:
    kpis: List[Dict[str, Any]] = [{"name": "Rows", "value": int(len(df)), "format": "int"}]
    used = set()
    for hint, agg in _KPI_HINTS:
        for c in df.columns:
            if c in used or hint not in c.lower() or not pd.api.types.is_numeric_dtype(df[c]):
                continue
            if c.lower().endswith("_id") or df[c].nunique() <= 2:
                continue
            used.add(c)
            k = {"name": f"{'Total' if agg == 'sum' else 'Avg'} {c.replace('_', ' ')}", "column": c, "agg": agg,
                 "value": float(getattr(df[c], agg)()), "format": "number"}
            if date_col and date_col in df.columns:
                ch = latest_period_change(aggregate_series(df, date_col, c, agg))
                if ch:
                    k["change_pct"] = ch["pct_change"]
                    k["change_period"] = f"{ch['previous_period']} → {ch['current_period']}"
            kpis.append(k)
            break
        if len(kpis) >= max_kpis:
            break
    for c in _rate_columns(df)[:2]:
        kpis.append({"name": f"{c.replace('_', ' ').title()} rate", "column": c, "value": float(df[c].mean() * 100), "format": "pct"})
    return kpis[:max_kpis + 2]
