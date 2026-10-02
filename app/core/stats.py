"""Statistical analysis with p-values and effect sizes (SciPy)."""
from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats


def correlations(df: pd.DataFrame, numeric: List[str], min_abs: float = 0.5, max_pairs: int = 10) -> List[Dict[str, Any]]:
    out = []
    for a, b in combinations(numeric, 2):
        d = df[[a, b]].dropna()
        if len(d) < 10 or d[a].nunique() < 3 or d[b].nunique() < 3:
            continue
        r, p = stats.pearsonr(d[a], d[b])
        rho, _ = stats.spearmanr(d[a], d[b])
        if abs(r) >= min_abs and p < 0.05:
            out.append({"x": a, "y": b, "pearson_r": round(float(r), 4), "spearman_rho": round(float(rho), 4),
                        "p_value": float(p), "n": int(len(d)),
                        "strength": "very strong" if abs(r) > 0.8 else "strong" if abs(r) > 0.65 else "moderate"})
    return sorted(out, key=lambda x: -abs(x["pearson_r"]))[:max_pairs]


def group_differences(df: pd.DataFrame, numeric: List[str], categorical: List[str], max_findings: int = 10) -> List[Dict[str, Any]]:
    """Does a numeric metric differ significantly across the levels of a categorical column?"""
    out = []
    for cat in categorical:
        levels = df[cat].dropna().unique()
        if not 2 <= len(levels) <= 30:
            continue
        for num in numeric:
            groups = [g[num].dropna().values for _, g in df.groupby(cat, observed=True) if g[num].notna().sum() >= 5]
            if len(groups) < 2:
                continue
            try:
                stat, p = stats.kruskal(*groups) if len(groups) > 2 else stats.mannwhitneyu(*groups)
            except ValueError:
                continue
            if p >= 0.01:
                continue
            allv = np.concatenate(groups)
            ss_tot = ((allv - allv.mean()) ** 2).sum()
            ss_between = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
            eta2 = float(ss_between / ss_tot) if ss_tot > 0 else 0.0
            if eta2 < 0.01:
                continue
            means = df.groupby(cat, observed=True)[num].mean().sort_values(ascending=False)
            out.append({"metric": num, "dimension": cat, "p_value": float(p), "effect_size_eta2": round(eta2, 4),
                        "highest": {"group": str(means.index[0]), "mean": round(float(means.iloc[0]), 4)},
                        "lowest": {"group": str(means.index[-1]), "mean": round(float(means.iloc[-1]), 4)},
                        "test": "kruskal" if len(groups) > 2 else "mann-whitney"})
    return sorted(out, key=lambda x: -x["effect_size_eta2"])[:max_findings]


def categorical_associations(df: pd.DataFrame, categorical: List[str], max_findings: int = 5) -> List[Dict[str, Any]]:
    out = []
    cats = [c for c in categorical if 2 <= df[c].nunique() <= 20]
    for a, b in combinations(cats, 2):
        ct = pd.crosstab(df[a], df[b])
        if ct.shape[0] < 2 or ct.shape[1] < 2 or ct.values.min() < 0:
            continue
        chi2, p, _, _ = stats.chi2_contingency(ct)
        n = ct.values.sum()
        v = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1)))) if n else 0.0
        if p < 0.01 and v >= 0.15:
            out.append({"a": a, "b": b, "cramers_v": round(v, 4), "p_value": float(p)})
    return sorted(out, key=lambda x: -x["cramers_v"])[:max_findings]


def normality(s: pd.Series) -> Dict[str, Any]:
    d = s.dropna().astype(float)
    if len(d) < 8:
        return {"tested": False}
    sample = d.sample(min(len(d), 5000), random_state=0)
    stat, p = stats.shapiro(sample)
    return {"tested": True, "shapiro_w": round(float(stat), 4), "p_value": float(p), "is_normal": bool(p > 0.05)}


def compare_two_groups(a: pd.Series, b: pd.Series) -> Dict[str, Any]:
    a, b = a.dropna().astype(float), b.dropna().astype(float)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    pooled = np.sqrt((a.var() + b.var()) / 2)
    d = float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "t": float(t), "p_value": float(p), "cohens_d": round(d, 4)}
