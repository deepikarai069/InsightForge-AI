import numpy as np
import pandas as pd
import pytest

from app.core import anomaly, stats, trends
from app.core.cleaning import clean_dataframe
from app.core.profiling import detect_relationships, profile_dataframe
from app.data.loader import load_file, snake


def test_snake_case_and_loader():
    assert snake("Order Date (UTC)") == "order_date_utc"
    t = load_file("My Sales.csv", b"Order ID,Total $\n1,5\n2,6\n")
    assert list(t) == ["my_sales"] and list(t["my_sales"].columns) == ["order_id", "total"]


def test_loader_rejects_unknown_type():
    with pytest.raises(ValueError):
        load_file("x.exe", b"")


def test_cleaning_types_dupes_missing():
    n = 40
    df = pd.DataFrame({"d": [f"2024-01-{(i % 28) + 1:02d}" for i in range(n)],
                       "amt": [f"${1000 + i:,.2f}" for i in range(n)], "c": ["West", "west", "West ", "East"] * 10})
    df.loc[3, "amt"] = "n/a"
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)   # 5 exact duplicates
    out, log = clean_dataframe(df)
    acts = {a["action"] for a in log.actions}
    assert {"convert_to_numeric", "convert_to_datetime", "drop_duplicate_rows", "impute_median"} <= acts
    assert pd.api.types.is_datetime64_any_dtype(out["d"]) and out["amt"].dtype.kind == "f"
    assert set(out["c"].unique()) <= {"West", "East"}


def test_profile_and_relationships():
    a = pd.DataFrame({"id": range(100), "x": np.random.default_rng(0).normal(size=100), "k": ["a", "b"] * 50})
    p = profile_dataframe(a, "a")
    assert p["rows"] == 100 and "x" in p["numeric_columns"] and "k" in p["categorical_columns"]
    b = pd.DataFrame({"cust_id": [1, 1, 2, 3, 4, 5, 2, 3], "v": range(8)})
    c = pd.DataFrame({"cust_id": [1, 2, 3, 4, 5], "name": list("abcde")})
    rels = detect_relationships({"b": b, "c": c})
    assert rels and rels[0]["cardinality"] == "many-to-one"


def test_trend_and_driver_analysis():
    d = pd.date_range("2024-01-01", "2024-06-30")
    df = pd.DataFrame({"date": np.tile(d, 2), "region": ["A"] * len(d) + ["B"] * len(d), "rev": 10.0})
    df.loc[(df.region == "B") & (df.date >= "2024-06-01"), "rev"] = 2.0
    s = trends.aggregate_series(df, "date", "rev")
    assert len(s) == 6
    drv = trends.driver_analysis(df, "date", "rev", ["region"])
    assert drv["delta"] < 0 and drv["drivers"][0]["member"] == "B"
    up = pd.Series(np.arange(1, 13) * 10.0, index=pd.date_range("2024-01-01", periods=12, freq="MS"))
    assert trends.detect_trend(up)["direction"] == "upward"


def test_anomaly_detection():
    rng = np.random.default_rng(1)
    s = pd.Series(rng.normal(100, 5, 120), index=pd.date_range("2024-01-01", periods=120))
    s.iloc[60] = 400
    ta = anomaly.time_series_anomalies(s)
    assert s.index[60] in ta.index and ta.loc[s.index[60], "direction"] == "spike"
    df = pd.DataFrame({"v": np.r_[rng.normal(10, 1, 200), [500]]})
    assert anomaly.univariate_anomalies(df, ["v"])["findings"][0]["count"] >= 1


def test_stats_correlation_and_groups():
    rng = np.random.default_rng(2)
    x = rng.normal(size=300)
    df = pd.DataFrame({"x": x, "y": 2 * x + rng.normal(scale=0.3, size=300), "g": rng.choice(["a", "b"], 300)})
    df.loc[df.g == "b", "x"] += 3
    assert stats.correlations(df, ["x", "y"])
    assert stats.group_differences(df, ["x"], ["g"])
