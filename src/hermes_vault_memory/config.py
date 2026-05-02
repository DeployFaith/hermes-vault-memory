from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Sequence

DEFAULT_VAULT_ROOTS = (
    Path("/vault/root-1"),
    Path("/vault/root-2"),
    Path("/vault/root-3"),
)


@dataclass(slots=True)
class ScanTarget:
    vault: str
    root: Path


def _split_env_paths(value: str | None) -> tuple[Path, ...]:
    if not value:
        return DEFAULT_VAULT_ROOTS
    paths: list[Path] = []
    for item in value.split(os.pathsep):
        item = item.strip()
        if item:
            paths.append(Path(item).expanduser().resolve())
    return tuple(paths) if paths else DEFAULT_VAULT_ROOTS


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _split_env_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_optional_positive_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    parsed = int(value)
    if parsed < 1:
        raise ValueError("max_file_bytes must be >= 1")
    return parsed


def _validate_targets(targets: Sequence[ScanTarget]) -> tuple[ScanTarget, ...]:
    seen_names: set[str] = set()
    seen_roots: set[Path] = set()
    for target in targets:
        if not target.vault:
            raise ValueError("vault name must not be empty")
        if target.vault in seen_names:
            raise ValueError(f"duplicate vault name: {target.vault}")
        if target.root in seen_roots:
            raise ValueError(f"duplicate vault root: {target.root}")
        seen_names.add(target.vault)
        seen_roots.add(target.root)
    return tuple(targets)


def _targets_from_roots(roots: Sequence[Path]) -> tuple[ScanTarget, ...]:
    return _validate_targets(tuple(ScanTarget(vault=root.name, root=root) for root in roots))


def _split_env_vaults(value: str | None) -> tuple[ScanTarget, ...] | None:
    if not value:
        return None
    targets: list[ScanTarget] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"invalid vault entry: {item!r}")
        name, root = item.split(":", 1)
        name = name.strip()
        root = root.strip()
        if not name:
            raise ValueError("vault name must not be empty")
        if not root:
            raise ValueError(f"vault root must not be empty for {name!r}")
        targets.append(ScanTarget(vault=name, root=Path(root).expanduser().resolve()))
    if not targets:
        return None
    return _validate_targets(targets)


def _validate_chunk_settings(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size")


def _validate_sync_settings(sync_poll_seconds: int, sync_full_resync_seconds: int) -> None:
    if sync_poll_seconds < 1:
        raise ValueError("sync_poll_seconds must be >= 1")
    if sync_full_resync_seconds < sync_poll_seconds:
        raise ValueError("sync_full_resync_seconds must be >= sync_poll_seconds")


@dataclass(slots=True)
class Settings:
    repo_root: Path
    vault_roots: tuple[Path, ...]
    data_dir: Path
    qdrant_path: Path
    qdrant_url: str | None
    manifest_path: Path
    collection_name: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    sync_poll_seconds: int
    sync_full_resync_seconds: int
    auth_token: str | None
    enable_mutation_tools: bool
    vaults: tuple[ScanTarget, ...]
    exclude_globs: tuple[str, ...]
    max_file_bytes: int | None

    @classmethod
    def load(cls) -> "Settings":
        repo_root = Path(__file__).resolve().parents[2]
        data_dir = Path(os.environ.get("HVM_DATA_DIR", repo_root / "data")).expanduser().resolve()
        vaults = _split_env_vaults(os.environ.get("HVM_VAULTS"))
        if vaults is None:
            vault_roots = _split_env_paths(os.environ.get("HVM_VAULT_ROOTS"))
            vaults = _targets_from_roots(vault_roots)
        else:
            vault_roots = tuple(target.root for target in vaults)
        collection_name = os.environ.get("HVM_COLLECTION_NAME", "hermes_vault_memory")
        embedding_model = os.environ.get("HVM_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        chunk_size = int(os.environ.get("HVM_CHUNK_SIZE", "1600"))
        chunk_overlap = int(os.environ.get("HVM_CHUNK_OVERLAP", "200"))
        qdrant_path = Path(os.environ.get("HVM_QDRANT_PATH", data_dir / "qdrant")).expanduser().resolve()
        qdrant_url = os.environ.get("HVM_QDRANT_URL", "http://qdrant:6333").strip() or None
        manifest_path = Path(os.environ.get("HVM_MANIFEST_PATH", data_dir / "manifest.json")).expanduser().resolve()
        sync_poll_seconds = int(os.environ.get("HVM_SYNC_POLL_SECONDS", "60"))
        sync_full_resync_seconds = int(os.environ.get("HVM_SYNC_FULL_RESYNC_SECONDS", "21600"))
        exclude_globs = _split_env_csv(os.environ.get("HVM_EXCLUDE_GLOBS"))
        max_file_bytes = _parse_optional_positive_int(os.environ.get("HVM_MAX_FILE_BYTES"))
        _validate_chunk_settings(chunk_size, chunk_overlap)
        _validate_sync_settings(sync_poll_seconds, sync_full_resync_seconds)
        auth_token = os.environ.get("HVM_AUTH_TOKEN", "").strip() or None
        enable_mutation_tools = _parse_bool(os.environ.get("HVM_ENABLE_MUTATION_TOOLS"), default=False)
        return cls(
            repo_root=repo_root,
            vault_roots=vault_roots,
            data_dir=data_dir,
            qdrant_path=qdrant_path,
            qdrant_url=qdrant_url,
            manifest_path=manifest_path,
            collection_name=collection_name,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            sync_poll_seconds=sync_poll_seconds,
            sync_full_resync_seconds=sync_full_resync_seconds,
            auth_token=auth_token,
            enable_mutation_tools=enable_mutation_tools,
            vaults=vaults,
            exclude_globs=exclude_globs,
            max_file_bytes=max_file_bytes,
        )

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.qdrant_path.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)


def vault_targets(settings: Settings) -> tuple[ScanTarget, ...]:
    return settings.vaults
