# Hermes Vault Memory Hardening and Reliability Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Make hermes-vault-memory secure-by-default, safer under concurrent sync/rebuild operations, easier to deploy, and faster during indexing.

**Architecture:** Keep the existing single-process FastAPI/FastMCP service, but move lifecycle startup into FastAPI lifespan, add a global sync operation lock, gate mutating MCP tools behind an explicit env flag, and expose clearer health semantics. Keep retrieval behavior unchanged except for faster batched embedding writes.

**Tech Stack:** Python 3.11, FastAPI, FastMCP, Qdrant client, fastembed, Docker Compose, unittest.

---

## Task 1: Add settings validation and security switches

**Objective:** Add validated config fields for mutation tools, bearer-token auth, and explicit vault names.

**Files:**
- Modify: `src/hermes_vault_memory/config.py`
- Test: `tests/test_config.py`

**Requirements:**
- Add `auth_token: str | None`
- Add `enable_mutation_tools: bool`, default false
- Add support for `HVM_VAULTS=name:/path,name2:/path2`
- Keep `HVM_VAULT_ROOTS` fallback compatible
- Validate chunk settings and sync cadence
- Reject duplicate vault names
- Reject duplicate vault roots

**Verification:**
- `python3 -m unittest tests.test_config -v`
- `python3 -m unittest discover -s tests -v`

---

## Task 2: Gate mutation MCP tools and add bearer auth middleware

**Objective:** Make remote use safer by requiring explicit opt-in for sync/rebuild tools and optional bearer-token auth.

**Files:**
- Modify: `src/hermes_vault_memory/service.py`
- Test: `tests/test_server.py` or new `tests/test_security.py`

**Requirements:**
- `build_mcp_server()` should register search/get/status always.
- Register sync/rebuild aliases only when `settings.enable_mutation_tools` is true.
- `build_fastapi_app()` should add HTTP middleware when `settings.auth_token` is configured.
- Bearer auth should not block `/health`, `/live`, or `/ready`.
- Bearer auth should protect `/mcp` and other non-health routes.
- Unauthorized responses should be 401 JSON.

**Verification:**
- Tests prove mutation tools hidden by default and visible when enabled.
- Tests prove auth accepts `Authorization: Bearer <token>` and rejects missing/wrong tokens.
- `python3 -m unittest discover -s tests -v`

---

## Task 3: Add sync/rebuild operation lock and lifecycle-safe monitor control

**Objective:** Prevent overlapping sync/rebuild operations and stop monitor threads cleanly.

**Files:**
- Modify: `src/hermes_vault_memory/service.py`
- Test: `tests/test_index_search.py`

**Requirements:**
- Add a dedicated operation lock for full sync/rebuild critical sections.
- `sync()` and `rebuild()` must reject concurrent operations with a clear RuntimeError.
- `start_background_sync()` must not schedule when an operation is already running.
- Add `stop_sync_monitor()`.
- Move background startup from app construction into FastAPI lifespan startup; stop monitor on shutdown.
- Preserve existing test behavior where direct service methods still work.

**Verification:**
- Test overlapping direct sync calls fail instead of racing.
- Existing background tests still pass.
- `python3 -m unittest discover -s tests -v`

---

## Task 4: Improve health semantics

**Objective:** Split liveness/readiness from general health and avoid over-reporting readiness before useful index state.

**Files:**
- Modify: `src/hermes_vault_memory/service.py`
- Test: `tests/test_server.py`

**Requirements:**
- Add service-level `health()` method.
- Add `/live` endpoint: process alive, returns 200.
- Add `/ready` endpoint: ready only when collection exists, sync not failed, and either initial sync completed or manifest has indexed files.
- Keep `/health` backward compatible but include `live` and `ready` booleans.
- Avoid reading private service fields directly from FastAPI route bodies where practical.

**Verification:**
- Tests cover empty/pre-sync not ready, post-sync ready, failed sync not ready.
- `python3 -m unittest discover -s tests -v`

---

## Task 5: Batch embeddings during document upserts

**Objective:** Avoid one embedder call per chunk by encoding batches in `_upsert_document()`.

**Files:**
- Modify: `src/hermes_vault_memory/service.py`
- Test: `tests/test_index_search.py`

**Requirements:**
- `_point_from_chunk` should accept a precomputed vector or add a sibling helper.
- `_upsert_document()` should call `_encode()` once per batch, not once per chunk.
- Preserve payload structure and point IDs.

**Verification:**
- Add test using a multi-chunk note and a counting fake embedder/client to prove one encode call per batch.
- `python3 -m unittest discover -s tests -v`

---

## Task 6: Harden Docker/Compose defaults and docs

**Objective:** Make deploy defaults match secure-by-default service behavior and document usage.

**Files:**
- Modify: `Dockerfile`
- Modify: `docker-compose.yml`
- Modify: `.env.example`
- Modify: `README.md`
- Test: `tests/test_compose.py`, `scripts/smoke.py`

**Requirements:**
- Bind service port to `127.0.0.1:8787:8787` by default.
- Do not publish Qdrant port by default.
- Add env docs for `HVM_AUTH_TOKEN`, `HVM_ENABLE_MUTATION_TOOLS`, `HVM_VAULTS`.
- Run container as non-root user where feasible.
- Update compose tests/smoke expectations.
- README includes Hermes MCP config example, auth notes, mutation tool warning, health endpoints, and troubleshooting.

**Verification:**
- `python3 -m unittest discover -s tests -v`
- `python3 scripts/smoke.py`

---

## Task 7: Final integration review and cleanup

**Objective:** Ensure the complete patch is consistent and deployable.

**Files:**
- Review all changed files.

**Requirements:**
- Run full tests and smoke.
- Inspect git diff.
- Remove generated `__pycache__` files from working tree if any.
- Confirm no secrets added.

**Verification:**
- `python3 -m unittest discover -s tests -v`
- `python3 scripts/smoke.py`
- `git diff --stat`
