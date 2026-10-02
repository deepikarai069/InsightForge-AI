.PHONY: install sample test api ui docker
install: ; pip install -r requirements.txt
sample:  ; python scripts/generate_sample_data.py
test:    ; pytest -q
api:     ; uvicorn app.api.main:app --reload --port 8000
ui:      ; streamlit run app/ui/streamlit_app.py
docker:  ; docker compose up --build
