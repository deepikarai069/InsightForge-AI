import pandas as pd
import pytest

from app.ai.nl2sql import classify_intent
from app.ai.sandbox import SandboxError, run_python
from app.ai.verify import verify_numbers
from app.data.sql_guard import SQLValidationError, validate_sql


@pytest.mark.parametrize("sql", [
    "DROP TABLE orders", "SELECT * FROM orders; DELETE FROM orders", "UPDATE orders SET revenue=0",
    "SELECT * FROM sqlite_master", "SELECT * FROM if_datasets", "WITH x AS (SELECT 1) INSERT INTO orders SELECT * FROM x",
    "SELECT load_extension('x')", "PRAGMA table_info(orders)", "SELECT * INTO t FROM orders"])
def test_sql_guard_blocks(sql):
    with pytest.raises(SQLValidationError):
        validate_sql(sql, ["orders"])


def test_sql_guard_allows_select_and_cte():
    assert validate_sql("-- c\nSELECT a FROM orders o JOIN products p ON 1=1;", ["orders", "products"])
    assert validate_sql("WITH t AS (SELECT * FROM orders) SELECT * FROM t WHERE x = 'update'", ["orders"])


def test_sandbox():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 2, 3]})
    assert run_python("result = df.groupby('a').b.sum()", df).to_dict() == {1: 3, 2: 3}
    for bad in ["import os", "result = open('/etc/passwd')", "result = df.__class__", "df.to_csv('x')", "while True: pass", "x = 1"]:
        with pytest.raises(SandboxError):
            run_python(bad, df)


def test_verify_numbers():
    ok = verify_numbers("Revenue was 1.86M, up 27.9%", {"v": 1_860_000, "share": 27.9})
    assert ok["verified"]
    bad = verify_numbers("Revenue was 9.99M", {"v": 1_860_000})
    assert not bad["verified"] and bad["unverified"]


def test_intent_classification():
    assert classify_intent("why did revenue drop in December?") == "investigate"
    assert classify_intent("any outliers in orders?") == "anomalies"
    assert classify_intent("what is the sales trend") == "trend"
    assert classify_intent("what is AOV?") == "define"
    assert classify_intent("what is total revenue by region?") == "query"
