"""Anomaly detection: robust statistics, IQR, Isolation Forest (multivariate) and time-series residuals."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def robust_z(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0:
        std = s.std()
        return (s - s.mean()) / std if std else pd.Series(0.0, index=s.index)
    return 0.6745 * (s - med) / mad


def univariate_anomalies(df: pd.DataFrame, columns: List[str], z_thresh: float = 4.0, max_rows: int = 20) -> Dict[str, Any]:
    findings = []
    for c in columns:
        s = df[c].dropna().astype(float)
        if len(s) < 30 or s.nunique() < 5:
            continue
        if s.min() > 0 and s.skew() > 1.5:   # heavy right tail → judge outliers on log scale
            z = robust_z(np.log1p(s))
        else:
            z = robust_z(s)
        idx = z.index[z.abs() > z_thresh]
        if len(idx) == 0 or len(idx) / len(s) > 0.05:   # >5% flagged = distribution shape, not anomalies
            continue
        rows = df.loc[idx].assign(_robust_z=z.loc[idx].round(2)).sort_values("_robust_z", key=abs, ascending=False)
        findings.append({"column": c, "method": "robust_z (MAD)", "threshold": z_thresh, "count": int(len(idx)),
                         "pct_of_rows": round(len(idx) / len(df) * 100, 3),
                         "examples": rows.head(5).to_dict(orient="records"),
                         "index": [int(i) for i in idx[:max_rows]]})
    return {"findings": sorted(findings, key=lambda f: -f["count"])}


def multivariate_anomalies(df: pd.DataFrame, columns: List[str], contamination: float = 0.01, seed: int = 42) -> pd.DataFrame:
    """Return rows flagged by an Isolation Forest, with anomaly scores (higher = more anomalous)."""
    X = df[columns].dropna()
    if len(X) < 50 or len(columns) < 2:
        return pd.DataFrame()
    Xs = (X - X.mean()) / X.std().replace(0, 1)
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)
    pred = model.fit_predict(Xs)
    scores = -model.score_samples(Xs)
    out = df.loc[X.index[pred == -1]].copy()
    out["_anomaly_score"] = scores[pred == -1].round(4)
    return out.sort_values("_anomaly_score", ascending=False)


def time_series_anomalies(series: pd.Series, window: int = 7, z_thresh: float = 3.5) -> pd.DataFrame:
    """Flag points that deviate from a rolling median beyond z_thresh robust standard deviations."""
    s = series.astype(float).dropna()
    if len(s) < window * 3:
        return pd.DataFrame()
    base = s.rolling(window, center=True, min_periods=window // 2).median()
    resid = s - base
    z = robust_z(resid)
    flag = z.abs() > z_thresh
    out = pd.DataFrame({"value": s[flag], "expected": base[flag], "robust_z": z[flag].round(2)})
    out["deviation_pct"] = ((out["value"] - out["expected"]) / out["expected"].abs().replace(0, np.nan) * 100).round(1)
    out["direction"] = np.where(out["value"] > out["expected"], "spike", "drop")
    return out.sort_values("robust_z", key=abs, ascending=False)
