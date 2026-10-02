"""FastAPI backend.  Run:  uvicorn app.api.main:app --reload"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from app.ai.schemas import Answer
from app.data.sql_guard import SQLValidationError
from app.services import InsightForgeService

app = FastAPI(title="InsightForge AI", version="1.0.0",
              description="Autonomous data analytics & business intelligence copilot")


@lru_cache
def get_service() -> InsightForgeService:
    return InsightForgeService()


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    session_id: str = "default"


class SQLRequest(BaseModel):
    sql: str
    limit: int = Field(200, ge=1, le=5000)


class GlossaryRequest(BaseModel):
    title: str
    text: str


class AnalysisRequest(BaseModel):
    fact: Optional[str] = None
    tables: Optional[List[str]] = None


@app.get("/health")
def health(svc: InsightForgeService = Depends(get_service)) -> Dict[str, Any]:
    return {"status": "ok", "llm": svc.llm.name if svc.llm else "offline-heuristic", "vector_backend": svc.kb.backend,
            "database": svc.store.engine.dialect.name, "datasets": len(svc.store.dataset_names())}


@app.post("/datasets/upload")
async def upload(files: List[UploadFile] = File(...), clean: bool = True, svc: InsightForgeService = Depends(get_service)):
    out = []
    for f in files:
        data = await f.read()
        if len(data) > svc.settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"{f.filename} exceeds {svc.settings.max_upload_mb} MB")
        try:
            out += svc.ingest_file(f.filename or "upload.csv", data, clean)
        except ValueError as e:
            raise HTTPException(400, str(e))
    return {"datasets": out, "relationships": svc.store.relationships(refresh=True)}


@app.get("/datasets")
def datasets(svc: InsightForgeService = Depends(get_service)):
    return {"datasets": svc.store.list_datasets(), "relationships": svc.store.relationships()}


@app.get("/datasets/{name}/profile")
def profile(name: str, svc: InsightForgeService = Depends(get_service)):
    try:
        return svc.profile(name)
    except KeyError:
        raise HTTPException(404, f"Dataset '{name}' not found")


@app.delete("/datasets/{name}")
def delete(name: str, svc: InsightForgeService = Depends(get_service)):
    try:
        svc.store.drop_dataset(name)
    except KeyError:
        raise HTTPException(404, f"Dataset '{name}' not found")
    svc.kb.replace_schema_docs(svc.store.schema())
    return {"deleted": name}


@app.post("/analysis/auto")
def auto(req: AnalysisRequest, svc: InsightForgeService = Depends(get_service)):
    if not svc.store.dataset_names():
        raise HTTPException(400, "Upload a dataset first")
    return svc.auto_analysis(req.fact, req.tables)


@app.post("/chat", response_model=Answer)
def chat(req: ChatRequest, svc: InsightForgeService = Depends(get_service)):
    return svc.ask(req.question, req.session_id)


@app.post("/sql")
def sql(req: SQLRequest, svc: InsightForgeService = Depends(get_service)):
    try:
        df = svc.run_sql(req.sql, req.limit)
    except SQLValidationError as e:
        raise HTTPException(400, f"Rejected: {e}")
    except Exception as e:
        raise HTTPException(400, f"SQL error: {str(e)[:300]}")
    return {"columns": list(df.columns), "rows": df.to_dict(orient="records"), "row_count": len(df)}


@app.post("/glossary")
def add_glossary(req: GlossaryRequest, svc: InsightForgeService = Depends(get_service)):
    return {"chunks_indexed": svc.add_glossary(req.title, req.text)}


@app.get("/glossary/search")
def search_glossary(q: str = Query(min_length=2), k: int = 4, svc: InsightForgeService = Depends(get_service)):
    return {"results": svc.kb.search(q, k, kind="glossary")}


@app.get("/reports", response_class=HTMLResponse)
def report(fact: Optional[str] = None, fmt: str = "html", svc: InsightForgeService = Depends(get_service)):
    if not svc.store.dataset_names():
        raise HTTPException(400, "Upload a dataset first")
    body = svc.report(fact, fmt=fmt)
    return PlainTextResponse(body, media_type="text/markdown") if fmt == "md" else HTMLResponse(body)
