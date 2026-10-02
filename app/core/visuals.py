"""Plotly figure factory. All functions return plotly Figures (JSON-serialisable)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

TEMPLATE = "plotly_white"


def histogram(df: pd.DataFrame, col: str) -> go.Figure:
    return px.histogram(df, x=col, nbins=40, marginal="box", title=f"Distribution of {col}", template=TEMPLATE)


def correlation_heatmap(df: pd.DataFrame, numeric: List[str]) -> Optional[go.Figure]:
    if len(numeric) < 2:
        return None
    corr = df[numeric].corr()
    return px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                     title="Correlation matrix", template=TEMPLATE, aspect="auto")


def time_series(series: pd.Series, title: str, anomalies: Optional[pd.DataFrame] = None) -> go.Figure:
    fig = go.Figure(go.Scatter(x=series.index, y=series.values, mode="lines+markers", name=series.name or "value"))
    if anomalies is not None and len(anomalies):
        fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies["value"], mode="markers", name="anomaly",
                                 marker=dict(color="red", size=11, symbol="x")))
    fig.update_layout(title=title, template=TEMPLATE, xaxis_title=None, yaxis_title=series.name)
    return fig


def bar_by_category(df: pd.DataFrame, cat: str, metric: Optional[str] = None, agg: str = "sum", top: int = 15) -> go.Figure:
    if metric:
        g = df.groupby(cat, observed=True)[metric].agg(agg).sort_values(ascending=False).head(top).reset_index()
        return px.bar(g, x=cat, y=metric, title=f"{agg.title()} of {metric} by {cat}", template=TEMPLATE)
    g = df[cat].value_counts().head(top).reset_index()
    g.columns = [cat, "count"]
    return px.bar(g, x=cat, y="count", title=f"Top {cat} values", template=TEMPLATE)


def waterfall(drivers: List[Dict[str, Any]], start: float, end: float, title: str = "What drove the change") -> go.Figure:
    labels = ["Previous"] + [f"{d['dimension']}={d['member']}" for d in drivers] + ["Other", "Current"]
    explained = sum(d["delta"] for d in drivers)
    vals = [start] + [d["delta"] for d in drivers] + [(end - start) - explained, end]
    measure = ["absolute"] + ["relative"] * (len(drivers) + 1) + ["total"]
    fig = go.Figure(go.Waterfall(x=labels, y=vals, measure=measure))
    fig.update_layout(title=title, template=TEMPLATE, showlegend=False)
    return fig


def auto_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Pick a sensible chart for an arbitrary result table (used by the chat agent)."""
    if df is None or df.empty or len(df.columns) < 1:
        return None
    cols = list(df.columns)
    dt = [c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c])]
    if not dt:  # SQL results often return dates as strings
        for c in cols:
            if df[c].dtype == object and df[c].astype(str).str.match(r"^\d{4}-\d{2}(-\d{2})?").mean() > 0.9:
                df = df.assign(**{c: pd.to_datetime(df[c], errors="coerce")})
                dt = [c]
                break
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and c not in dt]
    cat = [c for c in cols if c not in dt and c not in num]
    if dt and num:
        fig = px.line(df.sort_values(dt[0]), x=dt[0], y=num[:3], markers=True, template=TEMPLATE)
        return fig
    if cat and num and len(df) <= 60:
        return px.bar(df, x=cat[0], y=num[0], color=cat[1] if len(cat) > 1 else None, template=TEMPLATE)
    if len(num) >= 2:
        return px.scatter(df, x=num[0], y=num[1], template=TEMPLATE)
    if len(num) == 1 and len(df) > 5:
        return px.histogram(df, x=num[0], template=TEMPLATE)
    return None
