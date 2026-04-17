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


class ServiceStartupTests(unittest.TestCase):
    def test_service_ready_state_uses_mount_and_manifest_locations(self) -> None:
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
                },
                clear=True,
            ):
                settings = Settings.load()
                service = service_module.VaultMemoryService(settings)

            self.assertEqual(service.settings.vault_roots, (vault.resolve(),))
            self.assertEqual(service.settings.manifest_path, (data_dir / "manifest.json").resolve())
            self.assertEqual(service.settings.qdrant_path, (data_dir / "qdrant").resolve())
            self.assertEqual(service.status()["vault_roots"], [str(vault.resolve())])
