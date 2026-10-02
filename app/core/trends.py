"""Time-series trend detection and period-over-period root-cause (driver) analysis."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


def choose_grain(dates: pd.Series) -> str:
    """Return a pandas frequency alias using period-START labels (compatible with pandas 2 and 3)."""
    span = (dates.max() - dates.min()).days
    if span >= 120:
        return "MS"
    if span >= 30:
        return "W-MON"
    return "D"


def period_end(start: pd.Timestamp, grain: str) -> pd.Timestamp:
    """Last day (inclusive) of the bucket that begins at `start`."""
    if grain == "D":
        return start
    off = pd.tseries.frequencies.to_offset(grain)
    return start + off - pd.Timedelta(days=1)


def aggregate_series(df: pd.DataFrame, date_col: str, metric: str, agg: str = "sum",
                     grain: Optional[str] = None, drop_incomplete: bool = True) -> pd.Series:
    d = df[[date_col, metric]].dropna()
    grain = grain or choose_grain(d[date_col])
    kw = {"closed": "left", "label": "left"} if grain.startswith("W") else {}
    s = d.set_index(date_col)[metric].resample(grain, **kw).agg(agg)
    s.name = metric
    if drop_incomplete and len(s) > 3 and grain != "D":
        if d[date_col].max().normalize() < period_end(s.index[-1], grain):
            s = s.iloc[:-1]
    return s


def detect_trend(series: pd.Series) -> Dict[str, Any]:
    s = series.copy()
    if len(s) < 4:
        return {"direction": "insufficient_data", "n_periods": len(s)}
    x = np.arange(len(s))
    res = stats.linregress(x, s.values.astype(float))
    mean = float(s.mean()) or 1e-9
    rel_slope = float(res.slope / abs(mean))
    significant = res.pvalue < 0.05 and abs(rel_slope) > 0.005
    direction = "flat"
    if significant:
        direction = "upward" if res.slope > 0 else "downward"
    first, last = float(s.iloc[0]), float(s.iloc[-1])
    return {
        "direction": direction, "slope_per_period": float(res.slope), "relative_slope_pct": round(rel_slope * 100, 3),
        "r_squared": round(float(res.rvalue ** 2), 4), "p_value": float(res.pvalue), "n_periods": len(s),
        "first_value": first, "last_value": last,
        "pct_change_first_last": round((last - first) / abs(first) * 100, 2) if first else None,
        "start": str(s.index[0].date()), "end": str(s.index[-1].date()),
    }


def latest_period_change(series: pd.Series) -> Optional[Dict[str, Any]]:
    s = series.dropna()
    if len(s) < 3:
        return None
    if len(s) < 2 or s.iloc[-2] == 0:
        return None
    cur, prev = float(s.iloc[-1]), float(s.iloc[-2])
    return {"current_period": str(s.index[-1].date()), "previous_period": str(s.index[-2].date()),
            "current": cur, "previous": prev, "pct_change": round((cur - prev) / abs(prev) * 100, 2)}


def seasonality(series: pd.Series, period: int = 12) -> Optional[Dict[str, Any]]:
    if len(series) < period * 2:
        return None
    s = series.astype(float)
    detr = s - s.rolling(period, center=True, min_periods=period // 2).mean()
    seasonal = detr.groupby(np.arange(len(s)) % period).mean()
    strength = float(seasonal.std() / (s.std() or 1))
    return {"period": period, "strength": round(strength, 3), "peak_position": int(seasonal.idxmax()), "trough_position": int(seasonal.idxmin())}


def driver_analysis(
    df: pd.DataFrame, date_col: str, metric: str, dimensions: List[str],
    current: Optional[tuple] = None, previous: Optional[tuple] = None, agg: str = "sum", top_n: int = 5,
) -> Dict[str, Any]:
    """Decompose the change in `metric` between two periods into contributions by each dimension member.

    If periods are omitted, compares the last full grain bucket with the one before it.
    """
    d = df.dropna(subset=[date_col]).copy()
    if current is None or previous is None:
        grain = choose_grain(d[date_col])
        s = aggregate_series(d, date_col, metric, agg, grain)
        if len(s) < 2:
            return {"error": "Not enough periods to compare"}
        current = (s.index[-1], period_end(s.index[-1], grain))
        previous = (s.index[-2], period_end(s.index[-2], grain))
    cur = d[(d[date_col] >= pd.Timestamp(current[0])) & (d[date_col] <= pd.Timestamp(current[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))]
    prev = d[(d[date_col] >= pd.Timestamp(previous[0])) & (d[date_col] <= pd.Timestamp(previous[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))]
    total_cur, total_prev = float(getattr(cur[metric], agg)()), float(getattr(prev[metric], agg)())
    delta = total_cur - total_prev
    result: Dict[str, Any] = {
        "metric": metric, "agg": agg,
        "current_period": [str(pd.Timestamp(current[0]).date()), str(pd.Timestamp(current[1]).date())],
        "previous_period": [str(pd.Timestamp(previous[0]).date()), str(pd.Timestamp(previous[1]).date())],
        "current_total": total_cur, "previous_total": total_prev, "delta": delta,
        "pct_change": round(delta / abs(total_prev) * 100, 2) if total_prev else None, "drivers": [],
    }
    drivers: List[Dict[str, Any]] = []
    for dim in dimensions:
        if dim not in d.columns or d[dim].nunique() > 50:
            continue
        a = cur.groupby(dim, observed=True)[metric].agg(agg)
        b = prev.groupby(dim, observed=True)[metric].agg(agg)
        comp = pd.concat([a.rename("current"), b.rename("previous")], axis=1).fillna(0)
        comp["delta"] = comp["current"] - comp["previous"]
        comp["share_of_change_pct"] = comp["delta"] / delta * 100 if delta else 0.0
        comp["pct_change"] = np.where(comp["previous"] != 0, comp["delta"] / comp["previous"].abs() * 100, np.nan)
        # rank by contribution in the direction of the overall change
        comp = comp.sort_values("delta", ascending=(delta < 0))
        for member, row in comp.head(top_n).iterrows():
            if row["delta"] == 0 or np.sign(row["delta"]) != np.sign(delta):
                continue
            drivers.append({"dimension": dim, "member": str(member), "current": float(row["current"]),
                            "previous": float(row["previous"]), "delta": float(row["delta"]),
                            "share_of_change_pct": round(float(row["share_of_change_pct"]), 1),
                            "member_pct_change": None if pd.isna(row["pct_change"]) else round(float(row["pct_change"]), 1)})
    result["drivers"] = sorted(drivers, key=lambda x: -abs(x["delta"]))
    # --- hotspot: the 2-D cell (dimension pair) that best explains the change --------------
    hotspot = None
    if delta and len(dimensions) >= 1:
        from itertools import combinations
        dims_ok = [x for x in dimensions if x in d.columns and d[x].nunique() <= 50]
        best_score = 0.0
        for combo in list(combinations(dims_ok, 2)) or []:
            a = cur.groupby(list(combo), observed=True)[metric].agg(agg)
            b = prev.groupby(list(combo), observed=True)[metric].agg(agg)
            comp = pd.concat([a.rename("current"), b.rename("previous")], axis=1).fillna(0)
            comp["delta"] = comp["current"] - comp["previous"]
            comp = comp[np.sign(comp["delta"]) == np.sign(delta)]
            if comp.empty or total_prev == 0:
                continue
            comp["share"] = comp["delta"] / delta
            comp["base_share"] = comp["previous"] / total_prev
            comp = comp[comp["base_share"] > 0.005]
            comp["lift"] = comp["share"] / comp["base_share"]
            comp["score"] = comp["delta"].abs() * comp["lift"].clip(upper=20)
            if comp.empty:
                continue
            row = comp.sort_values("score", ascending=False).iloc[0]
            if row["score"] > best_score and row["share"] >= 0.3:
                best_score = float(row["score"])
                key = comp.sort_values("score", ascending=False).index[0]
                key = key if isinstance(key, tuple) else (key,)
                hotspot = {"cell": {c: str(v) for c, v in zip(combo, key)}, "current": float(row["current"]),
                           "previous": float(row["previous"]), "delta": float(row["delta"]),
                           "share_of_change_pct": round(float(row["share"] * 100), 1),
                           "base_share_pct": round(float(row["base_share"] * 100), 1), "lift": round(float(row["lift"]), 1)}
        result["hotspot"] = hotspot
    return result
