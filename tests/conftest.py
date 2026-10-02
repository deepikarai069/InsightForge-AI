import sys, warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from app.config import Settings  # noqa: E402
from app.services import InsightForgeService  # noqa: E402

SAMPLE = ROOT / "sample_data"


@pytest.fixture(scope="session")
def sample_dir():
    if not (SAMPLE / "orders.csv").exists():
        from scripts.generate_sample_data import main
        main()
    return SAMPLE


@pytest.fixture()
def svc(tmp_path, sample_dir):
    s = InsightForgeService(Settings(database_url=f"sqlite:///{tmp_path}/t.db", chroma_path=":memory:"), use_env_llm=False)
    for f in ["orders.csv", "customers.csv", "products.csv"]:
        s.ingest_file(f, (sample_dir / f).read_bytes())
    s.add_glossary("Business glossary", (sample_dir / "business_glossary.md").read_text())
    return s
