# Advanced RAG Application

Session-based, document-grounded chat API (FastAPI) with hybrid retrieval (BM25 + vectors),
cross-encoder reranking, persisted conversations and background answer evaluation.

## Run

```bash
pip install -r requirements.txt
cp .env.example .env            # set GOOGLE_API_KEY
uvicorn app.main:create_app --factory --reload
# UI: http://127.0.0.1:8000/ui     API docs: http://127.0.0.1:8000/docs
```

## Test

```bash
pip install -r requirements-dev.txt
pytest          # uses fakes for Gemini, Docling, the reranker and DeepEval - no network needed
ruff check . && ruff format --check .
```

## Layout

```
app/
  main.py           create_app() + lifespan (startup/shutdown); no import-time side effects
  config.py         Settings - the ONLY place environment variables are read
  container.py      composition root: builds and wires every long-lived object once
  exceptions.py     typed domain errors, each carrying its HTTP status
  domain.py         plain dataclasses/enums shared across layers
  api/              HTTP only: routes/, schemas.py, dependencies.py (Depends), error_handlers.py
  services/         business logic: SessionService, DocumentService, ChatService
  rag/              pipeline.py (stateless), llm.py (retries/timeouts), index.py, retrieval.py, prompts.py
  ingestion/        storage, validators, loader (Docling), ingestor
  evaluation/       metrics.py (DeepEval), service.py (background, bounded concurrency)
  persistence/      SQLite: sessions, documents (+chunks), turns, evaluations
tests/              fakes.py + API, pipeline and robustness tests
```

Dependencies point one way: `api -> services -> rag / ingestion / persistence`. Services depend on
small `Protocol`s (`ChatModel`, `Retriever`, `DocumentLoader`, `AnswerEvaluator`), so tests swap in fakes
through `create_app(settings, ContainerOverrides(...))`.

