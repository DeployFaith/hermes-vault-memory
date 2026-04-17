from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / 'src'):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from hermes_vault_memory.config import Settings
from tests.support.fakes import load_service_module


def assert_contains(text: str, fragment: str, label: str) -> None:
    if fragment not in text:
        raise AssertionError(f"Missing {label}: {fragment}")


def main() -> int:
    compose_text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    for fragment in (
        'qdrant:',
        'memory-service:',
        'HVM_VAULT_ROOTS: /vault/agent-main:/vault/psalmbox-main:/vault/katana-main',
        'HVM_QDRANT_URL: http://qdrant:6333',
        '- /workspace/vaults/agent-main:/vault/agent-main:ro',
        '- /workspace/vaults/psalmbox-main:/vault/psalmbox-main:ro',
        '- /workspace/vaults/katana-main:/vault/katana-main:ro',
        '- hermes-vault-memory-data:/data',
        '- qdrant-storage:/qdrant/storage',
        'restart: unless-stopped',
    ):
        assert_contains(compose_text, fragment, "compose fragment")

    service_module = load_service_module()
    fixture_vault = ROOT / "fixtures" / "sample-vault"

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        vault = root / "vault"
        data_dir = root / "data"
        shutil.copytree(fixture_vault, vault)

        with patch.dict(
            "os.environ",
            {
                "HVM_VAULT_ROOTS": str(vault),
                "HVM_DATA_DIR": str(data_dir),
                "HVM_QDRANT_PATH": str(data_dir / "qdrant"),
                "HVM_MANIFEST_PATH": str(data_dir / "manifest.json"),
            },
            clear=True,
        ):
            settings = Settings.load()
            service = service_module.VaultMemoryService(settings)

        try:
            with patch.dict(
                "os.environ",
                {
                    "HVM_VAULT_ROOTS": str(root / "missing-vault"),
                    "HVM_DATA_DIR": str(data_dir),
                    "HVM_QDRANT_PATH": str(data_dir / "qdrant"),
                    "HVM_MANIFEST_PATH": str(data_dir / "manifest.json"),
                },
                clear=True,
            ):
                missing_settings = Settings.load()
            try:
                service_module.VaultMemoryService(missing_settings)
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("Expected startup to fail when the vault mount is missing")

            summary = service.sync()
            assert summary["indexed_files"] == 2, summary
            assert summary["upserted_chunks"] == 2, summary

            search = service.search("vault mount", limit=5)
            assert search["query"] == "vault mount"
            assert search["results"][0]["relative_path"] == "dokploy-setup.md"

            status = service.status()
            assert status["ok"] is True
            assert status["files_indexed"] == 2
            assert status["vault_counts"]["vault"]["files"] == 2
        finally:
            pass

    print("Smoke check passed: compose layout, config loading, startup validation, and search flow are healthy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
