from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastmcp import FastMCP
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding

from .chunking import ParsedDocument, parse_markdown_file
from .config import ScanTarget, Settings, vault_targets


MANIFEST_VERSION = 1


@dataclass(slots=True)
class IndexSummary:
    scanned_files: int = 0
    indexed_files: int = 0
    skipped_files: int = 0
    deleted_files: int = 0
    changed_files: int = 0
    indexed_chunks: int = 0
    deleted_chunks: int = 0
    upserted_chunks: int = 0
    removed_orphans: int = 0
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SearchHit:
    id: str
    score: float
    vault: str
    relative_path: str
    title: str
    section_path: list[str]
    start_line: int
    end_line: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VaultMemoryService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.ensure_dirs()
        if self.settings.qdrant_url:
            self.client = QdrantClient(url=self.settings.qdrant_url)
            self._wait_for_qdrant_ready()
        else:
            self.client = QdrantClient(path=str(self.settings.qdrant_path))
        self.embedder = TextEmbedding(model_name=self.settings.embedding_model)
        self.vector_size = self._probe_vector_size()
        self._lock = threading.RLock()
        self._sync_thread: threading.Thread | None = None
        self._sync_state = "idle"
        self._sync_error: str | None = None
        self._last_sync_summary: dict[str, Any] | None = None
        self._vault_targets = tuple(
            target for target in vault_targets(self.settings) if target.root.exists() and target.root.is_dir()
        )
        if not self._vault_targets:
            configured = ", ".join(str(target.root) for target in vault_targets(self.settings))
            raise FileNotFoundError(
                "No configured vault mount was found. Mount your Markdown vault and set HVM_VAULT_ROOTS "
                f"(configured: {configured})."
            )
        self._manifest = self._load_manifest()
        self._ensure_collection()

    @property
    def manifest(self) -> dict[str, Any]:
        return self._manifest

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _probe_vector_size(self) -> int:
        sample = next(iter(self.embedder.embed(["vector size probe"])))
        return len(sample)

    def _wait_for_qdrant_ready(self, timeout_seconds: int = 30) -> None:
        deadline = time.monotonic() + timeout_seconds
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self.client.get_collections()
                return
            except Exception as exc:  # pragma: no cover - retry loop for container startup only
                last_error = exc
                time.sleep(1)
        raise RuntimeError(
            f"Timed out waiting for Qdrant at {self.settings.qdrant_url!r} to become ready"
        ) from last_error

    def _load_manifest(self) -> dict[str, Any]:
        if not self.settings.manifest_path.exists():
            return {
                "version": MANIFEST_VERSION,
                "collection_name": self.settings.collection_name,
                "embedding_model": self.settings.embedding_model,
                "chunk_size": self.settings.chunk_size,
                "chunk_overlap": self.settings.chunk_overlap,
                "last_sync": None,
                "files": {},
            }
        with self.settings.manifest_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if "files" not in data:
            data["files"] = {}
        return data

    def _save_manifest(self) -> None:
        tmp_path = self.settings.manifest_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(self._manifest, fh, indent=2, sort_keys=True)
            fh.write("\n")
        tmp_path.replace(self.settings.manifest_path)

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.settings.collection_name):
            self.client.create_collection(
                collection_name=self.settings.collection_name,
                vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE),
            )
        else:
            info = self.client.get_collection(self.settings.collection_name)
            current = getattr(info.config.params.vectors, "size", None)
            if current not in {None, self.vector_size}:
                raise RuntimeError(
                    f"Qdrant collection {self.settings.collection_name!r} already exists with vector size {current}, "
                    f"but this service expects {self.vector_size}."
                )


    def _sync_worker(self, paths: Sequence[str | Path] | None = None) -> None:
        try:
            summary = self.sync(paths=paths)
        except Exception as exc:
            with self._lock:
                self._sync_state = "failed"
                self._sync_error = str(exc)
                self._sync_thread = None
            return
        with self._lock:
            self._sync_state = "complete"
            self._sync_error = None
            self._last_sync_summary = summary
            self._sync_thread = None

    def start_background_sync(self, paths: Sequence[str | Path] | None = None) -> bool:
        with self._lock:
            if self._sync_thread and self._sync_thread.is_alive():
                return False
            self._sync_state = "running"
            self._sync_error = None
            worker = threading.Thread(target=self._sync_worker, kwargs={"paths": paths}, daemon=True)
            self._sync_thread = worker
            worker.start()
            return True

    def _document_key(self, vault: str, relative_path: str) -> str:
        return f"{vault}/{relative_path}"

    def _resolve_target(self, path: str | Path, vault: str | None = None) -> tuple[ScanTarget, Path]:
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            for target in self._vault_targets:
                try:
                    rel = candidate.resolve().relative_to(target.root.resolve())
                except Exception:
                    continue
                return target, target.root / rel
            raise ValueError(f"Path {str(path)!r} is not inside a configured vault")
        if vault:
            for target in self._vault_targets:
                if target.vault == vault:
                    return target, target.root / candidate
            raise ValueError(f"Unknown vault {vault!r}")
        for target in self._vault_targets:
            absolute = target.root / candidate
            if absolute.exists():
                return target, absolute
        raise ValueError(f"Could not resolve path {str(path)!r} against configured vaults")

    def _discover_files(self, paths: Sequence[str | Path] | None = None) -> list[tuple[ScanTarget, Path]]:
        if paths:
            discovered: list[tuple[ScanTarget, Path]] = []
            for raw in paths:
                target, absolute = self._resolve_target(raw)
                if absolute.is_file() and absolute.suffix.lower() in {".md", ".markdown", ".mdown"}:
                    discovered.append((target, absolute))
            return discovered

        discovered = []
        for target in self._vault_targets:
            for path in target.root.rglob("*"):
                if path.is_file() and path.suffix.lower() in {".md", ".markdown", ".mdown"}:
                    discovered.append((target, path))
        return discovered

    def _encode(self, texts: Iterable[str]) -> list[list[float]]:
        embeddings = self.embedder.embed(list(texts))
        return [list(map(float, embedding)) for embedding in embeddings]

    def _point_from_chunk(self, document: ParsedDocument, chunk) -> models.PointStruct:
        payload = {
            "document_key": self._document_key(document.vault, document.relative_path),
            "vault": document.vault,
            "root": str(document.root),
            "relative_path": document.relative_path,
            "title": document.title,
            "file_hash": document.file_hash,
            "chunk_index": chunk.chunk_index,
            "chunk_id": chunk.chunk_id,
            "section_path": list(chunk.section_path),
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "char_count": chunk.char_count,
            "content_hash": chunk.content_hash,
            "text": chunk.text,
            "source_mtime": document.mtime,
            "source_size": document.size,
        }
        return models.PointStruct(id=chunk.chunk_id, vector=self._encode([chunk.text])[0], payload=payload)

    def _upsert_document(self, document: ParsedDocument) -> None:
        batch_size = 128
        for i in range(0, len(document.chunks), batch_size):
            batch = document.chunks[i : i + batch_size]
            points = [self._point_from_chunk(document, chunk) for chunk in batch]
            self.client.upsert(collection_name=self.settings.collection_name, points=points)

    def _delete_chunk_ids(self, chunk_ids: Sequence[str]) -> int:
        if not chunk_ids:
            return 0
        self.client.delete(
            collection_name=self.settings.collection_name,
            points_selector=models.PointIdsList(points=list(chunk_ids)),
        )
        return len(chunk_ids)

    def sync(self, paths: Sequence[str | Path] | None = None) -> dict[str, Any]:
        start = datetime.now(timezone.utc)
        self._ensure_collection()
        summary = IndexSummary()
        discovered = self._discover_files(paths)
        summary.scanned_files = len(discovered)
        current_keys: set[str] = set()

        for target, path in discovered:
            if not path.exists():
                continue
            rel_path = path.resolve().relative_to(target.root.resolve()).as_posix()
            key = self._document_key(target.vault, rel_path)
            current_keys.add(key)
            stat = path.stat()
            with self._lock:
                files_manifest = self._manifest.setdefault("files", {})
                previous = files_manifest.get(key)
            if previous and previous.get("size") == stat.st_size and float(previous.get("mtime", -1)) == stat.st_mtime:
                summary.skipped_files += 1
                continue

            document = parse_markdown_file(
                path=path,
                vault=target.vault,
                root=target.root,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            if previous:
                summary.deleted_chunks += self._delete_chunk_ids(previous.get("chunk_ids", []))
            self._upsert_document(document)
            summary.changed_files += 1
            summary.indexed_files += 1
            summary.indexed_chunks += len(document.chunks)
            summary.upserted_chunks += len(document.chunks)
            with self._lock:
                files_manifest = self._manifest.setdefault("files", {})
                files_manifest[key] = {
                    "vault": document.vault,
                    "root": str(document.root),
                    "relative_path": document.relative_path,
                    "title": document.title,
                    "file_hash": document.file_hash,
                    "size": document.size,
                    "mtime": document.mtime,
                    "chunk_ids": [chunk.chunk_id for chunk in document.chunks],
                    "chunk_count": len(document.chunks),
                    "updated_at": self._now(),
                }

        removed_keys: list[str] = []
        if paths is None:
            with self._lock:
                files_manifest = self._manifest.setdefault("files", {})
                removed_keys = [key for key in list(files_manifest.keys()) if key not in current_keys]
            for key in removed_keys:
                with self._lock:
                    files_manifest = self._manifest.setdefault("files", {})
                    entry = files_manifest.pop(key, None)
                if not entry:
                    continue
                summary.deleted_files += 1
                summary.deleted_chunks += self._delete_chunk_ids(entry.get("chunk_ids", []))
                summary.removed_orphans += 1

        with self._lock:
            self._manifest.update(
                {
                    "version": MANIFEST_VERSION,
                    "collection_name": self.settings.collection_name,
                    "embedding_model": self.settings.embedding_model,
                    "chunk_size": self.settings.chunk_size,
                    "chunk_overlap": self.settings.chunk_overlap,
                    "last_sync": self._now(),
                    "vault_roots": [str(target.root) for target in self._vault_targets],
                }
            )
            self._save_manifest()
        summary.duration_seconds = (datetime.now(timezone.utc) - start).total_seconds()
        return summary.to_dict()

    def rebuild(self) -> dict[str, Any]:
        with self._lock:
            if self.client.collection_exists(self.settings.collection_name):
                self.client.delete_collection(self.settings.collection_name)
            self.client.create_collection(
                collection_name=self.settings.collection_name,
                vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE),
            )
            self._manifest = {
                "version": MANIFEST_VERSION,
                "collection_name": self.settings.collection_name,
                "embedding_model": self.settings.embedding_model,
                "chunk_size": self.settings.chunk_size,
                "chunk_overlap": self.settings.chunk_overlap,
                "last_sync": None,
                "files": {},
            }
            self._save_manifest()
        return self.sync()

    def _point_to_hit(self, point) -> SearchHit:
        payload = point.payload or {}
        return SearchHit(
            id=str(point.id),
            score=float(point.score or 0.0),
            vault=str(payload.get("vault", "")),
            relative_path=str(payload.get("relative_path", "")),
            title=str(payload.get("title", "")),
            section_path=list(payload.get("section_path", [])),
            start_line=int(payload.get("start_line", 0)),
            end_line=int(payload.get("end_line", 0)),
            text=str(payload.get("text", "")),
        )

    def search(self, query: str, limit: int = 5, vault: str | None = None) -> dict[str, Any]:
        with self._lock:
            self._ensure_collection()
            query_vector = self._encode([query])[0]
            qfilter = None
            if vault:
                qfilter = models.Filter(
                    must=[models.FieldCondition(key="vault", match=models.MatchValue(value=vault))]
                )
            results = self.client.query_points(
                collection_name=self.settings.collection_name,
                query=query_vector,
                limit=limit,
                query_filter=qfilter,
                with_payload=True,
                with_vectors=False,
            )
            points = getattr(results, "points", results)
            return {
                "query": query,
                "limit": limit,
                "vault": vault,
                "results": [self._point_to_hit(result).to_dict() for result in points],
            }
    def get(self, path: str | None = None, chunk_id: str | None = None, vault: str | None = None) -> dict[str, Any]:
        with self._lock:
            self._ensure_collection()
            if chunk_id:
                points = self.client.retrieve(
                    collection_name=self.settings.collection_name,
                    ids=[chunk_id],
                    with_payload=True,
                    with_vectors=False,
                )
                if not points:
                    raise KeyError(f"Chunk {chunk_id!r} not found")
                point = points[0]
                payload = point.payload or {}
                document_key = str(payload.get("document_key", ""))
                entry = self._manifest.get("files", {}).get(document_key)
                return {
                    "document": entry,
                    "chunk": self._point_to_hit(point).to_dict(),
                }

            if not path:
                raise ValueError("Either path or chunk_id must be provided")

            target, absolute = self._resolve_target(path, vault=vault)
            rel_path = absolute.resolve().relative_to(target.root.resolve()).as_posix()
            key = self._document_key(target.vault, rel_path)
            entry = self._manifest.get("files", {}).get(key)
            if not entry:
                raise KeyError(f"Document {key!r} not indexed")
            points = self.client.retrieve(
                collection_name=self.settings.collection_name,
                ids=entry.get("chunk_ids", []),
                with_payload=True,
                with_vectors=False,
            )
            chunks = [self._point_to_hit(point).to_dict() for point in points]
            chunks.sort(key=lambda item: item["start_line"])
            return {
                "document": entry,
                "chunks": chunks,
            }

    def status(self) -> dict[str, Any]:
        with self._lock:
            collection_exists = self.client.collection_exists(self.settings.collection_name)
            point_count = 0
            if collection_exists:
                try:
                    point_count = int(self.client.count(self.settings.collection_name, exact=True).count)
                except Exception:
                    point_count = 0
            files = self._manifest.get("files", {})
            chunk_count = sum(int(entry.get("chunk_count", 0)) for entry in files.values())
            vault_counts: dict[str, dict[str, int]] = {}
            for entry in files.values():
                vault = str(entry.get("vault", ""))
                bucket = vault_counts.setdefault(vault, {"files": 0, "chunks": 0})
                bucket["files"] += 1
                bucket["chunks"] += int(entry.get("chunk_count", 0))
            return {
                "ok": collection_exists,
                "collection_name": self.settings.collection_name,
                "embedding_model": self.settings.embedding_model,
                "vector_size": self.vector_size,
                "qdrant_path": str(self.settings.qdrant_path),
                "manifest_path": str(self.settings.manifest_path),
                "vault_roots": [str(target.root) for target in self._vault_targets],
                "files_indexed": len(files),
                "chunks_indexed": chunk_count,
                "points_indexed": point_count,
                "last_sync": self._manifest.get("last_sync"),
                "vault_counts": vault_counts,
                "sync_state": self._sync_state,
                "sync_error": self._sync_error,
                "sync_thread_alive": bool(self._sync_thread and self._sync_thread.is_alive()),
                "last_sync_summary": self._last_sync_summary,
            }



def load_settings() -> Settings:
    return Settings.load()



def build_mcp_server(service: VaultMemoryService | None = None) -> FastMCP:
    service = service or VaultMemoryService(load_settings())
    mcp = FastMCP("Hermes Vault Memory")

    @mcp.tool(name="search_vault")
    def search_vault(query: str, limit: int = 5, vault: str | None = None) -> dict[str, Any]:
        """Search the indexed vault memories."""
        return service.search(query=query, limit=limit, vault=vault)

    @mcp.tool(name="search")
    def search(query: str, limit: int = 5, vault: str | None = None) -> dict[str, Any]:
        return search_vault(query=query, limit=limit, vault=vault)

    @mcp.tool(name="get_note_context")
    def get_note_context(path: str | None = None, chunk_id: str | None = None, vault: str | None = None) -> dict[str, Any]:
        """Fetch an indexed note or chunk by path or chunk id."""
        return service.get(path=path, chunk_id=chunk_id, vault=vault)

    @mcp.tool(name="get")
    def get(path: str | None = None, chunk_id: str | None = None, vault: str | None = None) -> dict[str, Any]:
        return get_note_context(path=path, chunk_id=chunk_id, vault=vault)

    @mcp.tool(name="sync_vault")
    def sync_vault(paths: list[str] | None = None) -> dict[str, Any]:
        """Scan vault markdown and sync changed files into Qdrant."""
        return service.sync(paths=paths)

    @mcp.tool(name="sync")
    def sync(paths: list[str] | None = None) -> dict[str, Any]:
        return sync_vault(paths=paths)

    @mcp.tool(name="memory_status")
    def memory_status() -> dict[str, Any]:
        """Report collection, manifest, and vault index state."""
        return service.status()

    @mcp.tool(name="status")
    def status() -> dict[str, Any]:
        return memory_status()

    @mcp.tool(name="rebuild_vault_index")
    def rebuild_vault_index() -> dict[str, Any]:
        """Drop and rebuild the full index from the configured vaults."""
        return service.rebuild()

    @mcp.tool(name="rebuild")
    def rebuild() -> dict[str, Any]:
        return rebuild_vault_index()

    return mcp



def build_fastapi_app(service: VaultMemoryService | None = None) -> FastAPI:
    service = service or VaultMemoryService(load_settings())
    mcp = build_mcp_server(service)
    mcp_app = mcp.http_app(path="/")

    app = FastAPI(lifespan=mcp_app.lifespan)
    app.mount("/mcp", mcp_app)
    service.start_background_sync()

    @app.get("/health")
    def health() -> JSONResponse:
        collection_exists = service.client.collection_exists(service.settings.collection_name)
        ready = bool(collection_exists) and service._sync_state != "failed"
        return JSONResponse(
            status_code=200 if ready else 503,
            content={
                "status": "ok" if ready else "not-ready",
                "ready": ready,
                "service": {
                    "ok": collection_exists,
                    "collection_name": service.settings.collection_name,
                    "sync_state": service._sync_state,
                    "sync_error": service._sync_error,
                },
            },
        )

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "service": "hermes-vault-memory",
            "health": "/health",
            "mcp": "/mcp",
        }

    return app


def build_stdio_server(service: VaultMemoryService | None = None) -> FastMCP:
    return build_mcp_server(service)
