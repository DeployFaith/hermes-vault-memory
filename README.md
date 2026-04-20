# hermes-vault-memory

Local semantic memory for Hermes across three mounted note roots.

This repo is meant to be deployed from Docker Compose or Dokploy on a host that can see the configured mounts. It is not a cloud sync toy. The roots must be mounted into the container.

## What it does

- runs **Qdrant** in Docker on port **6333**
- runs a Python **memory-service** that:
  - scans Markdown notes from all three vaults
  - chunks them structurally
  - embeds them locally
  - stores vectors in Qdrant
  - exposes MCP tools over HTTP
  - exposes a `/health` endpoint for Dokploy

## Stack

- Qdrant for vector storage
- FastMCP for the tool layer
- FastAPI for the HTTP wrapper and health checks
- `fastembed` for local embeddings

## Compose layout

The default `docker-compose.yml` expects three host mounts provided via environment variables:

- `HVM_HOST_VAULT_1` → `/vault/root-1`
- `HVM_HOST_VAULT_2` → `/vault/root-2`
- `HVM_HOST_VAULT_3` → `/vault/root-3`

If you deploy somewhere else, set those mount variables to match your machine. If the container cannot see the configured roots, startup should fail instead of pretending everything is fine.

## Run locally

```bash
docker compose up --build
```

Then check:

- health: `http://localhost:8787/health`
- MCP: `http://localhost:8787/mcp`
- Qdrant: `http://localhost:6333`

## MCP tools

The server exposes both the long and short tool names:

- `search_vault` / `search`
- `get_note_context` / `get`
- `sync_vault` / `sync`
- `memory_status` / `status`
- `rebuild_vault_index` / `rebuild`

## Dokploy notes

Use the repo root as the build context. Set the deployment port to `8787`.
The important part is the bind mounts and a persistent `/data` volume.

If Dokploy runs on a different machine than the vaults, this repo will not magically fix that. You would need a sync/mirror layer first.

## Validation

Run the checks before pushing or deploying:

```bash
python -m unittest discover -s tests -v
python scripts/smoke.py
```
