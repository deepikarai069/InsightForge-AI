"""Automatic, fully-logged data cleaning. Every change is recorded so results stay auditable."""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

NULL_TOKENS = {"", "na", "n/a", "nan", "null", "none", "nil", "-", "--", "?", "missing"}
_NUM_CLEAN = re.compile(r"[,$€£₹%\s]")


@dataclass
class CleaningLog:
    actions: List[Dict] = field(default_factory=list)

    def add(self, action: str, column: str | None = None, count: int = 0, detail: str = "") -> None:
        self.actions.append({"action": action, "column": column, "count": int(count), "detail": detail})

    def to_list(self) -> List[Dict]:
        return list(self.actions)


def _looks_numeric(s: pd.Series, threshold: float = 0.9) -> bool:
    sample = s.dropna().astype(str).head(500)
    if sample.empty:
        return False
    cleaned = sample.str.replace(_NUM_CLEAN, "", regex=True)
    ok = pd.to_numeric(cleaned, errors="coerce").notna().mean()
    return ok >= threshold


def _looks_datetime(s: pd.Series, threshold: float = 0.9) -> bool:
    sample = s.dropna().astype(str).head(300)
    if sample.empty or sample.str.fullmatch(r"-?\d+(\.\d+)?").mean() > 0.5:
        return False
    if not sample.str.contains(r"[-/:]|[A-Za-z]{3}", regex=True).mean() > 0.8:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() >= threshold


def clean_dataframe(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    impute: bool = True,
    max_impute_pct: float = 30.0,
) -> Tuple[pd.DataFrame, CleaningLog]:
    log = CleaningLog()
    out = df.copy()

    # 1. Normalise string columns
    for col in out.select_dtypes(include=["object", "string"]).columns:
        s = out[col].astype("string")
        stripped = s.str.strip()
        changed = int((s != stripped).fillna(False).sum())
        if changed:
            log.add("strip_whitespace", col, changed)
        nulls = stripped.str.lower().isin(NULL_TOKENS)
        if nulls.any():
            log.add("standardise_null_tokens", col, int(nulls.sum()), "converted placeholder strings to NULL")
        stripped = stripped.mask(nulls)
        out[col] = stripped.astype(object).where(stripped.notna(), np.nan)

    # 2. Type inference
    for col in out.columns:
        if out[col].dtype != object:
            continue
        if _looks_numeric(out[col]):
            conv = pd.to_numeric(out[col].astype(str).str.replace(_NUM_CLEAN, "", regex=True), errors="coerce")
            conv[out[col].isna()] = np.nan
            log.add("convert_to_numeric", col, int(conv.notna().sum()), "parsed numeric-looking strings (currency/commas/%)")
            out[col] = conv
        elif _looks_datetime(out[col]):
            conv = pd.to_datetime(out[col], errors="coerce", format="mixed")
            log.add("convert_to_datetime", col, int(conv.notna().sum()))
            out[col] = conv
        else:
            # Harmonise categorical casing when it collapses variants (e.g. "west", "West", "WEST")
            s = out[col].dropna().astype(str)
            if 0 < s.nunique() <= max(50, len(s) * 0.05):
                lower_map = s.groupby(s.str.lower()).agg(lambda x: x.value_counts().idxmax())
                mapped = out[col].map(lambda v: lower_map.get(str(v).lower(), v) if pd.notna(v) else v)
                changed = int((mapped.fillna("") != out[col].fillna("")).sum())
                if changed:
                    log.add("harmonise_case", col, changed, "merged case-variants of the same category")
                out[col] = mapped

    # 3. Duplicates
    if drop_duplicates:
        n = int(out.duplicated().sum())
        if n:
            out = out.drop_duplicates().reset_index(drop=True)
            log.add("drop_duplicate_rows", None, n, "exact duplicate rows removed")

    # 4. Missing values
    if impute:
        for col in out.columns:
            miss = int(out[col].isna().sum())
            if miss == 0:
                continue
            pct = miss / len(out) * 100
            if pct > max_impute_pct:
                log.add("skip_imputation", col, miss, f"{pct:.1f}% missing exceeds {max_impute_pct}% — left as NULL")
                continue
            if pd.api.types.is_numeric_dtype(out[col]) and not pd.api.types.is_bool_dtype(out[col]):
                med = out[col].median()
                out[col] = out[col].fillna(med)
                log.add("impute_median", col, miss, f"filled with median={med:.4g}")
            elif pd.api.types.is_datetime64_any_dtype(out[col]):
                log.add("skip_imputation", col, miss, "datetime left as NULL")
            else:
                out[col] = out[col].fillna("Unknown")
                log.add("impute_unknown", col, miss, "filled with 'Unknown'")

    return out, log
