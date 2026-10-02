"""Pydantic schemas used for structured LLM outputs and API responses."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Intent(BaseModel):
    intent: Literal["query", "investigate", "anomalies", "trend", "define"] = Field(
        description="query=numbers/aggregation; investigate=why did a metric change; anomalies=unusual values; "
                    "trend=direction over time; define=business term / metric definition")
    metric: Optional[str] = Field(None, description="Main metric column if identifiable")


class QueryPlan(BaseModel):
    sql: str = Field(description="A single read-only SELECT statement using only the listed tables/columns")
    python: Optional[str] = Field(None, description="Optional pandas code for post-processing; must assign `result`")
    explanation: str = Field(description="One-sentence description of what the query computes")


class Narrative(BaseModel):
    executive_summary: str
    root_causes: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class Answer(BaseModel):
    question: str
    answer: str
    intent: str
    engine: str
    sql: Optional[str] = None
    columns: List[str] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    chart_json: Optional[str] = None
    evidence: Dict[str, Any] = Field(default_factory=dict)
    context: List[Dict[str, Any]] = Field(default_factory=list)
    verification: Dict[str, Any] = Field(default_factory=dict)
    steps: List[str] = Field(default_factory=list)
    error: Optional[str] = None
