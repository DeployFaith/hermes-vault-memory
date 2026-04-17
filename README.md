# hermes-vault-memory

Local semantic memory for Hermes across all three vaults:
- `/workspace/vaults/agent-main`
- `/workspace/vaults/psalmbox-main`
- `/workspace/vaults/katana-main`

This repo is meant to be deployed from Docker Compose or Dokploy on the host that can see those vault paths. It is not a cloud sync toy. The vaults must be mounted into the container.

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

The default `docker-compose.yml` expects these host mounts:

- `/workspace/vaults/agent-main` → `/vault/agent-main`
- `/workspace/vaults/psalmbox-main` → `/vault/psalmbox-main`
- `/workspace/vaults/katana-main` → `/vault/katana-main`

If you deploy somewhere else, change the bind mounts. If the container cannot see the vaults, startup should fail instead of pretending everything is fine.

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
The important part is the vault bind mounts and a persistent `/data` volume.

If Dokploy runs on a different machine than the vaults, this repo will not magically fix that. You would need a sync/mirror layer first.

## Validation

Run the checks before pushing or deploying:

```bash
python -m unittest discover -s tests -v
python scripts/smoke.py
```
