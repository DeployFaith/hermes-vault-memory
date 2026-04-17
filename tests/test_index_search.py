from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import shutil
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hermes_vault_memory.config import Settings
from tests.support.fakes import load_service_module


class IndexSearchTests(unittest.TestCase):
    def test_startup_rejects_missing_vault_mount(self) -> None:
        service_module = load_service_module()
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing_vault = root / "missing-vault"
            data_dir = root / "data"

            with patch.dict(
                "os.environ",
                {
                    "HVM_VAULT_ROOTS": str(missing_vault),
                    "HVM_DATA_DIR": str(data_dir),
                    "HVM_QDRANT_PATH": str(data_dir / "qdrant"),
                    "HVM_MANIFEST_PATH": str(data_dir / "manifest.json"),
                },
                clear=True,
            ):
                settings = Settings.load()

            with self.assertRaises(FileNotFoundError):
                service_module.VaultMemoryService(settings)

    def test_sync_and_search_flow_on_sample_markdown(self) -> None:
        service_module = load_service_module()
        fixture_vault = Path(__file__).resolve().parents[1] / "fixtures" / "sample-vault"

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
                    "HVM_COLLECTION_NAME": "demo_collection",
                },
                clear=True,
            ):
                settings = Settings.load()
                service = service_module.VaultMemoryService(settings)

            summary = service.sync()
            self.assertEqual(summary["indexed_files"], 2)
            self.assertEqual(summary["upserted_chunks"], 2)

            results = service.search("vault mount", limit=5)
            self.assertEqual(results["query"], "vault mount")
            self.assertGreaterEqual(len(results["results"]), 1)
            self.assertEqual(results["results"][0]["relative_path"], "dokploy-setup.md")
            self.assertIn("vault", results["results"][0]["text"].lower())

            status = service.status()
            self.assertTrue(status["ok"])
            self.assertEqual(status["files_indexed"], 2)
            self.assertEqual(status["vault_counts"]["vault"]["files"], 2)
