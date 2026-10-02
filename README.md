<div align="center">

# 🔮 InsightForge



*An AI-powered BI & analytics copilot that turns plain English into guarded SQL, automated insights, charts and reports.*

**Created by Deepika Rai**

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
[![CI](https://github.com/<your-username>/insightforge/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/insightforge/actions/workflows/ci.yml)

[Overview](#-overview) · [Features](#-key-features) · [Architecture](#-system-architecture) · [Quickstart](#-quickstart--installation) · [Project Structure](#-repository-layout) · [Configuration](#-configuration--environment-variables) · [Testing](#-testing--ci) · [Contributing](#-roadmap--contributing)

</div>

---

## 📖 Overview

Business data lives in databases and spreadsheets, but the people who need answers from it rarely write SQL. Giving a language model direct database access is risky: it can invent numbers, write a wrong query that *looks* right, or run something destructive.

**InsightForge** bridges that gap. Upload CSV/Excel files (or point it at a database), and ask questions in plain English. Every question goes through an agent that retrieves business definitions, plans a query, passes it through strict SQL guardrails, executes it read-only, and then **verifies that every number in the final answer matches computed evidence**.

It also works without asking anything: InsightForge profiles and cleans your data, detects trends and anomalies, finds the segment behind a metric change (for example *"Online orders in the West region"*), and produces ready-to-share reports. No API key? A deterministic offline engine handles common questions, and you can plug in Anthropic or OpenAI for free-form ones.

---

## ✨ Key Features

### 🧠 Agentic Text-to-SQL & Guardrails
- **LangGraph agent** that classifies intent, retrieves context, plans, executes, repairs and summarises (`app/ai/agent.py`).
- **NL → SQL** with structured query plans, schema awareness and chat history (`nl2sql.py`).
- **SQL guard** allows a single `SELECT` only, with a table allow-list, forbidden keywords and a row cap (`app/data/sql_guard.py`).
- **Self-correction:** on a SQL error the agent receives the error and retries (up to 2×).
- **RAG over a business glossary** (ChromaDB) so "revenue" or "active customer" follow *your* definitions (`rag.py`).

### 📊 Deep Analytics & Anomaly Detection
- **Auto-profiling:** semantic types, data quality score, column statistics (`profiling.py`).
- **Fully logged cleaning:** null tokens, currency/date parsing, case harmonisation, de-duplication, imputation (`cleaning.py`).
- **Statistics:** Pearson/Spearman correlations, Kruskal/Mann-Whitney tests with effect sizes, χ²/Cramér's V (`stats.py`).
- **KPIs & trends:** auto-detected KPIs with period-over-period change, regression trend and seasonality (`kpi.py`, `trends.py`).
- **Root-cause analysis:** decomposes a change by dimension and finds the hotspot segment.
- **Anomalies:** robust z-score/MAD, rolling-median time-series residuals, Isolation Forest (`anomaly.py`).

### 🖥️ Interactive UI & Automated Reports
- **Streamlit app** with a guided workflow: upload → validate → clean → explore → dashboard → anomalies → chat → SQL → insights → report → export.
- **Plotly charts:** time series with anomaly markers, waterfall of drivers, category bars, histograms, correlation heatmap.
- **Reports:** one-click HTML (interactive) or Markdown with executive summary, KPIs, findings, root causes and recommendations (`app/reports/generator.py`).
- **REST API** with Swagger docs for integration (`app/api/main.py`).

### 🛡️ Safety & Trust
- **Number verification:** every figure in the narrative is matched against computed evidence; unmatched figures are flagged (`verify.py`).
- **Python sandbox** for LLM-written code: AST validation, no imports/IO/dunder access, stripped builtins, line budget (`sandbox.py`).
- **Read-only execution** (read-only transaction on PostgreSQL) with result row limits.
- **No silent data changes:** every cleaning action is shown in the UI and the report.

---

## 🏗️ System Architecture

```mermaid
flowchart LR
    U[User Query / Dashboard] --> A[Agent<br/>LangGraph]
    A --> R[RAG<br/>glossary + schema<br/>ChromaDB]
    R --> N[NL2SQL<br/>query plan]
    N --> G{SQL Guard<br/>SELECT-only, allow-list}
    G -- rejected --> N
    G -- ok --> S[(Analytics Store<br/>SQLite / PostgreSQL)]
    S --> V[Verification<br/>numbers vs evidence]
    V -- error / mismatch --> N
    V --> O[Insights · Charts · Reports]
    S --> C[Core Analytics<br/>profile · clean · stats · trends · anomalies]
    C --> O
```

The agent graph in detail:

```
classify → retrieve (RAG) → ┬ plan → execute ⇄ repair ┐
                            ├ investigate (drivers + hotspot) ├→ summarize → verify → END
                            └ anomalies · trend · define ┘
```

Both the Streamlit UI and the FastAPI backend call the same `InsightForgeService` façade (`app/services.py`). See [docs/architecture.md](docs/architecture.md) for the full safety model.

---

## 🚀 Quickstart & Installation

### Prerequisites
- Python **3.10+** (CI runs on 3.12)
- Docker and Docker Compose (optional, for the full stack with PostgreSQL)
- An Anthropic or OpenAI API key *(optional; InsightForge works offline without one)*

### Step 1 · Clone & set up the environment
```bash
git clone https://github.com/<your-username>/insightforge.git
cd insightforge
cp .env.example .env
```

### Step 2 · Run with Docker Compose (API + UI + PostgreSQL)
```bash
docker compose up --build        # or: make docker
```

### Step 3 · Or install locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install                      # pip install -r requirements.txt

make api                          # FastAPI on :8000
make ui                           # Streamlit on :8501 (in another terminal)
```

### Step 4 · Seed the sample data
```bash
python -m scripts.generate_sample_data    # or: make sample
```
Sample files (`orders.csv`, `customers.csv`, `products.csv`, `business_glossary.md`) are also bundled in `sample_data/`. In the UI, click **Load demo e-commerce data**.

### Access
| Service | URL |
|---|---|
| Streamlit app | http://localhost:8501 |
| FastAPI Swagger docs | http://localhost:8000/docs |

### Enable an LLM (optional)
```bash
pip install langchain-anthropic        # or: langchain-openai
# .env
LLM_PROVIDER=anthropic                 # or openai
LLM_MODEL=claude-sonnet-4-5
ANTHROPIC_API_KEY=your-key
```

### Try these questions
- `total revenue by region` · `top 5 products by revenue`
- `why did revenue drop in December?` → finds the Online / West hotspot
- `any anomalies in revenue?`
- `how do we calculate gross margin?` → answered from the glossary via RAG

---

## 📁 Repository Layout

```
insightforge/
├── app/
│   ├── ai/            # agent · nl2sql · rag · llm · sandbox · verify · schemas
│   ├── api/           # FastAPI app (main.py)
│   ├── core/          # cleaning · profiling · stats · trends · anomaly · kpi · insights · visuals
│   ├── data/          # loader · store (SQLAlchemy) · sql_guard
│   ├── reports/       # HTML / Markdown report generator
│   ├── ui/            # Streamlit app
│   ├── config.py      # settings from environment variables
│   └── services.py    # shared service façade used by API and UI
├── sample_data/       # orders.csv · customers.csv · products.csv · business_glossary.md
├── scripts/           # generate_sample_data.py
├── tests/             # test_core.py · test_ai.py · test_platform.py
├── docs/              # architecture.md
├── .github/workflows/ci.yml
├── Dockerfile · docker-compose.yml · Makefile
└── requirements.txt · .env.example
```

### REST API
| Method & path | Purpose |
|---|---|
| `POST /datasets/upload` | Multi-file upload (`?clean=true`) with profiles and inferred relationships |
| `GET /datasets` · `GET /datasets/{name}/profile` | List and profile datasets |
| `POST /analysis/auto` | Full autonomous analysis: KPIs, insights, drivers, anomalies |
| `POST /chat` | Question in, answer + SQL + rows + chart + evidence + verification out |
| `POST /sql` | Guarded read-only SQL |
| `POST /glossary` · `GET /glossary/search?q=` | Business definitions (RAG) |
| `GET /reports?fmt=html\|md` | Generated report |
| `GET /health` | Health check |

---

## ⚙️ Configuration & Environment Variables

Copy `.env.example` to `.env`. All values are optional unless noted.

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./data/insightforge.db` | SQLAlchemy URL. Use `postgresql+psycopg2://user:pass@host:5432/db` for PostgreSQL |
| `LLM_PROVIDER` | `none` | `none` (offline engine), `anthropic` or `openai` |
| `LLM_MODEL` | `claude-sonnet-4-5` | Chat model name for the chosen provider |
| `ANTHROPIC_API_KEY` | – | Required when `LLM_PROVIDER=anthropic` |
| `OPENAI_API_KEY` | – | Required when `LLM_PROVIDER=openai` or `EMBEDDING_PROVIDER=openai` |
| `CHROMA_PATH` | `./data/chroma` | ChromaDB vector store location |
| `EMBEDDING_PROVIDER` | `hashing` | `hashing` (offline, lexical) or `openai` (semantic) |
| `MAX_RESULT_ROWS` | `1000` | Row cap for query results |
| `MAX_UPLOAD_MB` | `100` | Maximum upload size |

---

## 🧪 Testing & CI

```bash
make test        # or: pytest -q
```

The suite covers core analytics, AI safety (SQL guard, sandbox, verification) and end-to-end ground-truth checks on the sample data. GitHub Actions (`.github/workflows/ci.yml`) installs dependencies and runs `pytest` on every push and pull request.

---

## 🗺️ Roadmap & Contributing

### Roadmap
- [ ] Native PDF report export
- [ ] Scheduled and emailed reports
- [ ] More connectors (MySQL, Google Sheets, cloud warehouses)
- [ ] Authentication, workspaces and persistent project history
- [ ] Stronger sandbox isolation (container-level)
- [ ] Faster analytics on very large tables by pushing aggregation into SQL

### Contributing
1. **Open an issue** first for bugs or larger features (include steps to reproduce and your Python version).
2. **Fork** the repo and create a branch: `git checkout -b feature/your-change`.
3. Install dependencies, make your change, and **add or update tests**.
4. Run `make test` and make sure everything passes.
5. Open a **pull request** describing what changed and why. Keep PRs focused and small.

Please keep changes to the SQL guard and sandbox conservative: safety checks should only ever get stricter, and any relaxation needs tests and a clear justification.

---



<div align="center">

Built by **Deepika Rai** to make business data answerable, and every answer checkable. ⭐ Star the repo if it helps you.

</div>