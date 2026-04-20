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


def _split_env_paths(value: str | None) -> tuple[Path, ...]:
    if not value:
        return DEFAULT_VAULT_ROOTS
    paths: list[Path] = []
    for item in value.split(os.pathsep):
        item = item.strip()
        if item:
            paths.append(Path(item).expanduser().resolve())
    return tuple(paths) if paths else DEFAULT_VAULT_ROOTS


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

    @classmethod
    def load(cls) -> "Settings":
        repo_root = Path(__file__).resolve().parents[2]
        data_dir = Path(os.environ.get("HVM_DATA_DIR", repo_root / "data")).expanduser().resolve()
        vault_roots = _split_env_paths(os.environ.get("HVM_VAULT_ROOTS"))
        collection_name = os.environ.get("HVM_COLLECTION_NAME", "hermes_vault_memory")
        embedding_model = os.environ.get("HVM_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        chunk_size = int(os.environ.get("HVM_CHUNK_SIZE", "1600"))
        chunk_overlap = int(os.environ.get("HVM_CHUNK_OVERLAP", "200"))
        qdrant_path = Path(os.environ.get("HVM_QDRANT_PATH", data_dir / "qdrant")).expanduser().resolve()
        qdrant_url = os.environ.get("HVM_QDRANT_URL", "http://qdrant:6333").strip() or None
        manifest_path = Path(os.environ.get("HVM_MANIFEST_PATH", data_dir / "manifest.json")).expanduser().resolve()
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
        )

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.qdrant_path.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class ScanTarget:
    vault: str
    root: Path


def vault_targets(settings: Settings) -> tuple[ScanTarget, ...]:
    targets: list[ScanTarget] = []
    for root in settings.vault_roots:
        targets.append(ScanTarget(vault=root.name, root=root))
    return tuple(targets)
