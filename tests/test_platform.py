import pytest
from fastapi.testclient import TestClient


def test_ingest_cleans_and_relates(svc):
    names = svc.store.dataset_names()
    assert set(names) == {"orders", "customers", "products"}
    log = svc.store.get_cleaning_log("orders")
    acts = {a["action"] for a in log}
    assert {"convert_to_numeric", "convert_to_datetime", "drop_duplicate_rows"} <= acts
    assert len(svc.store.relationships()) == 2
    assert "customers_segment" in svc.store.joined_frame("orders").columns or "segment" in svc.store.joined_frame("orders").columns


def test_nl_to_sql_answers_match_ground_truth(svc):
    a = svc.ask("total revenue by region")
    truth = svc.run_sql("SELECT region, SUM(revenue) r FROM orders GROUP BY 1 ORDER BY 2 DESC")
    assert a.rows[0]["region"] == truth.region.iloc[0]
    assert a.rows[0]["total_revenue"] == pytest.approx(truth.r.iloc[0])
    assert a.verification["verified"] and a.sql and a.chart_json


def test_join_query(svc):
    a = svc.ask("top 5 products by revenue")
    assert len(a.rows) == 5 and "JOIN products" in a.sql
    a = svc.ask("total revenue by segment")
    assert {"Consumer", "Enterprise"} <= {r["segment"] for r in a.rows}


def test_root_cause_finds_planted_issue(svc):
    a = svc.ask("why did revenue drop in December?")
    assert a.intent == "investigate"
    cell = a.evidence["hotspot"]["cell"]
    assert cell == {"region": "West", "channel": "Online"} or cell == {"channel": "Online", "region": "West"}


def test_anomaly_finds_planted_day(svc):
    a = svc.ask("any anomalies in revenue?")
    assert "2024-08-14" in a.answer or "2024-11-29" in a.answer


def test_rag_definition(svc):
    a = svc.ask("how do we calculate gross margin?")
    assert a.intent == "define" and "cost of goods sold" in a.answer.lower()


def test_unanswerable_is_graceful(svc):
    a = svc.ask("tell me a joke about squirrels")
    assert a.answer and not a.rows


def test_auto_analysis_and_reports(svc):
    res = svc.auto_analysis("orders")
    cats = {i["category"] for i in res["insights"]}
    assert {"driver", "anomaly", "data_quality"} <= cats
    html = svc.report("orders")
    assert "<html" in html and "Root cause" in html and "West" in html
    assert "## Key findings" in svc.report("orders", fmt="md")


def test_api(svc):
    from app.api import main
    main.app.dependency_overrides[main.get_service] = lambda: svc
    c = TestClient(main.app)
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/chat", json={"question": "average order value by channel"})
    assert r.status_code == 200 and r.json()["rows"]
    assert c.post("/sql", json={"sql": "DROP TABLE orders"}).status_code == 400
    assert c.post("/sql", json={"sql": "SELECT COUNT(*) n FROM orders"}).json()["rows"][0]["n"] > 1000
    up = c.post("/datasets/upload", files=[("files", ("tiny.csv", b"a,b\n1,2\n3,4\n", "text/csv"))])
    assert up.status_code == 200 and up.json()["datasets"][0]["name"] == "tiny"
    assert c.get("/datasets/nope/profile").status_code == 404
    assert "<html" in c.get("/reports").text
    assert c.get("/glossary/search", params={"q": "gross margin"}).json()["results"]
    main.app.dependency_overrides.clear()
