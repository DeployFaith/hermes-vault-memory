# hermes-vault-memory

Local semantic memory for Hermes across mounted Markdown note roots.

This repo is meant to be deployed from Docker Compose or Dokploy on a host that can see the configured mounts. It is not a cloud sync toy. The roots must be mounted into the container.

## What it does

- runs Qdrant privately on the Docker Compose network
- runs a Python memory-service that:
  - scans Markdown notes from configured vaults
  - chunks them structurally while skipping YAML frontmatter and heading-only sections
  - embeds them locally
  - stores vectors in Qdrant
  - searches with semantic candidates plus lexical title/path reranking
  - exposes MCP tools over HTTP at `/mcp`
  - exposes health endpoints for Compose, Dokploy, and reverse proxies
  - continuously polls mounted vault roots for Markdown changes and kicks off incremental syncs
  - performs a full reconciliation sync on a fixed interval so deletions and missed updates get cleaned up

## Stack

- Qdrant for vector storage
- FastMCP for the tool layer
- FastAPI for the HTTP wrapper and health checks
- `fastembed` for local embeddings

## Compose layout

The default `docker-compose.yml` runs two services:

- `qdrant`: internal vector database. Its port is not published by default.
- `memory-service`: HTTP/MCP service published on `127.0.0.1:8787:8787` by default.

The localhost-only bind is intentional. If you expose the service publicly, put it behind auth and a trusted network boundary such as a reverse proxy, Tailscale, or equivalent private network. Do not publish Qdrant unless you are debugging locally and understand the exposure.

The default compose file mounts `./fixtures/sample-vault` into all three container vault paths so this works out of the box:

```bash
docker compose up --build
```

For a real deployment, set the host mount variables in `.env` or your deployment UI:

- `HVM_HOST_VAULT_1` -> `/vault/root-1`
- `HVM_HOST_VAULT_2` -> `/vault/root-2`
- `HVM_HOST_VAULT_3` -> `/vault/root-3`

If the container cannot see the configured roots, startup should fail instead of pretending everything is fine.

## Configuration

Common environment variables:

- `HVM_VAULT_ROOTS`: container paths to scan, separated by the platform path separator. In the Linux container this is `:`. Default: `/vault/root-1:/vault/root-2:/vault/root-3`.
- `HVM_VAULTS`: optional named vault mapping. Format: `name:/container/path,name2:/container/path2`. When set, it takes precedence over `HVM_VAULT_ROOTS` and controls vault names in status/results.
- `HVM_DATA_DIR`: persistent service data directory. Default in Docker: `/data`.
- `HVM_QDRANT_URL`: Qdrant URL. Default in Compose: `http://qdrant:6333`.
- `HVM_QDRANT_PATH`: local embedded-Qdrant path when running without `HVM_QDRANT_URL`. Default: `/data/qdrant` in Docker.
- `HVM_MANIFEST_PATH`: sync manifest path. Default: `/data/manifest.json` in Docker.
- `HVM_SYNC_POLL_SECONDS`: incremental polling cadence. Default: `60`.
- `HVM_SYNC_FULL_RESYNC_SECONDS`: full reconciliation cadence. Default: `21600` (6 hours).
- `HVM_COLLECTION_NAME`: Qdrant collection name. Default: `hermes_vault_memory`.
- `HVM_EMBEDDING_MODEL`: fastembed model. Default: `BAAI/bge-small-en-v1.5`.
- `HVM_CHUNK_SIZE`: target chunk size. Default: `1600`.
- `HVM_CHUNK_OVERLAP`: chunk overlap. Default: `200`.
- `HVM_EXCLUDE_GLOBS`: comma-separated relative-path glob patterns to skip. Compose defaults to `Archives/OpenCode Sessions/**` so raw agent transcript archives do not dominate the index.
- `HVM_MAX_FILE_BYTES`: optional maximum Markdown file size to index. Compose defaults to `2097152` (2 MiB); set empty to disable.
- `HVM_AUTH_TOKEN`: optional bearer auth value for HTTP/MCP access. If set, non-health endpoints require an `Authorization` header containing `Bearer YOUR_TOKEN`.
- `HVM_ENABLE_MUTATION_TOOLS`: enables MCP tools that mutate the vector index (`sync_vault`/`sync`, `rebuild_vault_index`/`rebuild`). Default: `false`.

Mutation tool warning: leave `HVM_ENABLE_MUTATION_TOOLS=false` for read-only clients and public-facing deployments unless you explicitly need remote sync/rebuild control. Enabling it lets an MCP client trigger index writes/rebuilds.

## Health endpoints

- `GET /live`: process liveness. Exempt from bearer-token auth.
- `GET /ready`: readiness check. Exempt from bearer-token auth.
- `GET /health`: compatibility health check for Compose/Dokploy. Exempt from bearer-token auth.
- `GET /`: basic service metadata. Requires bearer auth when `HVM_AUTH_TOKEN` is set.
- `/mcp`: MCP HTTP endpoint. Requires bearer auth when `HVM_AUTH_TOKEN` is set.

Local checks:

```bash
curl http://127.0.0.1:8787/health
curl -H "Authorization: Bearer YOUR_TOKEN" http://127.0.0.1:8787/
```

## Hermes MCP config example

Example local Hermes MCP server entry:

```json
{
  "mcpServers": {
    "hermes-vault-memory": {
      "type": "http",
      "url": "http://127.0.0.1:8787/mcp",
      "headers": {
        "Authorization": "Bearer ${HVM_AUTH_TOKEN}"
      }
    }
  }
}
```

If `HVM_AUTH_TOKEN` is unset, omit the `headers` block. For public or shared networks, set `HVM_AUTH_TOKEN` and terminate TLS/auth at a reverse proxy or private overlay network.

## Retrieval quality

Indexing intentionally avoids low-signal Markdown structure:

- leading YAML frontmatter is stripped before chunking/search indexing, while source line numbers stay stable
- heading-only sections are not embedded as standalone chunks
- search asks Qdrant for a wider semantic candidate set, adds lightweight lexical candidates, then reranks with keyword coverage, exact title/file-name matches, title/path bigrams, path matches, and section matches
- `find_notes` provides a dedicated manifest-backed title/path lookup for exact note discovery, including zero-chunk notes that are intentionally absent from vector search
- an internal index schema version forces unchanged files to reindex after chunking/retrieval-quality changes

This keeps agent searches from returning frontmatter blocks, empty headings, or broad context notes ahead of precise operational files.

## MCP tools

Read-only tools are registered by default:

- `search_vault` / `search` for semantic + lexical chunk search
- `find_notes` / `find_note` / `find` for exact/fuzzy note title/path lookup without vector search
- `get_note_context` / `get`
- `memory_status` / `status`

Mutation tools are registered only when `HVM_ENABLE_MUTATION_TOOLS=true`:

- `sync_vault` / `sync`
- `rebuild_vault_index` / `rebuild`

## Dokploy notes

Use the repo root as the build context. Set the deployment port to `8787`.

The important parts are:

- bind the memory service through Dokploy/reverse proxy to port `8787`
- configure the vault bind mounts (`HVM_HOST_VAULT_1`, `HVM_HOST_VAULT_2`, `HVM_HOST_VAULT_3`) or equivalent Dokploy volume mounts
- keep a persistent `/data` volume
- set `HVM_AUTH_TOKEN` before exposing `/mcp` beyond localhost/private networking
- keep Qdrant private; the memory service reaches it over the Docker network

The Compose file preserves the external `dokploy-network` declaration for Dokploy compatibility. If Dokploy runs on a different machine than the vaults, this repo will not magically fix that. You need a sync/mirror layer first.

If exposing this service publicly, put it behind auth plus a reverse proxy, Tailscale, or another trusted private network boundary. Avoid direct unauthenticated internet exposure.

## Troubleshooting

- Compose refuses to start because vault env vars are missing: current compose defaults should work locally with `./fixtures/sample-vault`; for real data, create `.env` from `.env.example` and set absolute `HVM_HOST_VAULT_*` paths.
- Startup fails with missing vault mounts: verify the host paths exist and are mounted into `/vault/root-*` or update `HVM_VAULT_ROOTS`/`HVM_VAULTS` to match the container paths.
- `/mcp` returns 401: set an `Authorization` header containing `Bearer YOUR_TOKEN` matching `HVM_AUTH_TOKEN`, or unset `HVM_AUTH_TOKEN` for local-only testing.
- Sync/rebuild tools are missing: set `HVM_ENABLE_MUTATION_TOOLS=true` and restart. They are intentionally disabled by default.
- Qdrant is not reachable from the host: this is expected. It is private by default. For temporary local debugging, add a localhost-only Qdrant port mapping such as `127.0.0.1:6333:6333`.
- Permission errors under `/data` or cache directories: the image runs as a non-root `hvm` user. Ensure any bind-mounted data/cache directories are writable by UID/GID `10001`, or use Docker named volumes.
- Health check is unhealthy: inspect `docker compose logs memory-service qdrant`, confirm Qdrant is running, and confirm vault mounts are readable.

## Validation

Run the checks before pushing or deploying:

```bash
python3 -m unittest discover -s tests -v
python3 scripts/smoke.py
```
