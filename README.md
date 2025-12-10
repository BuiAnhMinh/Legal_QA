# Vietnamese Legal QA System (Scaffold)

Initialized folder structure and Dockerized Postgres (with `pgvector`) ready for a Vietnamese legal QA project. No app code yet—just the foundation.

## Quick start
- Copy environment template: `cp .env.example .env` and adjust credentials/ports.
- Start the database: `docker compose up -d`
- Verify health: `docker compose ps` or `docker compose logs -f db`

## Project layout
- `backend/` — placeholder for API/embeddings code; `src/`, `tests/` ready to fill.
- `data/` — `raw/` for source legal texts, `processed/` for cleaned/structured data.
- `docs/` — planning, architecture notes.
- `docker/` — Postgres init hooks (`docker-entrypoint-initdb.d`).
- `scripts/` — helper scripts can live here.
- `docker-compose.yml` — Postgres service definition.

## Database notes
- Uses `ankane/pgvector:pg15` so embeddings can be stored later.
- Init hook `docker/postgres/init/00_extensions.sql` enables `pgvector` on first start.
- Data volume is persisted via `pgdata` volume (see compose file).

## Next steps (suggested)
- Wire up an ingestion script to load Vietnamese legal texts into Postgres.
- Decide on embedding model + chunking strategy; create a `documents` + `embeddings` schema.
- Add API layer (e.g., FastAPI) and retrieval pipeline that leverages `pgvector`.
