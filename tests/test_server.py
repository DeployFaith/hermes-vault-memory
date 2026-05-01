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
    def _build_service(self, service_module, root: Path):
        fixture_vault = Path(__file__).resolve().parents[1] / "fixtures" / "sample-vault"
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
        return service, vault, data_dir

    def test_service_ready_state_uses_mount_and_manifest_locations(self) -> None:
        service_module = load_service_module()

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service, vault, data_dir = self._build_service(service_module, root)

            self.assertEqual(service.settings.vault_roots, (vault.resolve(),))
            self.assertEqual(service.settings.manifest_path, (data_dir / "manifest.json").resolve())
            self.assertEqual(service.settings.qdrant_path, (data_dir / "qdrant").resolve())
            self.assertEqual(service.status()["vault_roots"], [str(vault.resolve())])

    def test_service_health_is_not_ready_until_initial_sync_finishes(self) -> None:
        service_module = load_service_module()

        with TemporaryDirectory() as temp_dir:
            service, _vault, _data_dir = self._build_service(service_module, Path(temp_dir))

            health = service.health()
            self.assertTrue(health["live"])
            self.assertFalse(health["ready"])
            self.assertEqual(health["status"], "not-ready")
            self.assertTrue(health["service"]["ok"])

            service.sync()
            health = service.health()
            self.assertTrue(health["live"])
            self.assertTrue(health["ready"])
            self.assertEqual(health["status"], "ok")
            self.assertGreater(health["service"]["files_indexed"], 0)

    def test_service_health_ready_when_manifest_has_indexed_files(self) -> None:
        service_module = load_service_module()

        with TemporaryDirectory() as temp_dir:
            service, _vault, _data_dir = self._build_service(service_module, Path(temp_dir))

            service.manifest["files"] = {
                "default/example.md": {"vault": "default", "chunk_count": 1}
            }
            service.manifest["last_sync"] = None

            health = service.health()
            self.assertTrue(health["ready"])
            self.assertEqual(health["status"], "ok")

    def test_service_health_not_ready_when_sync_failed(self) -> None:
        service_module = load_service_module()

        with TemporaryDirectory() as temp_dir:
            service, _vault, _data_dir = self._build_service(service_module, Path(temp_dir))
            service.sync()
            service._sync_state = "failed"
            service._sync_error = "boom"

            health = service.health()
            self.assertFalse(health["ready"])
            self.assertEqual(health["status"], "not-ready")
            self.assertEqual(health["service"]["sync_error"], "boom")

    def test_fastapi_live_ready_and_health_routes(self) -> None:
        service_module = load_service_module()

        with TemporaryDirectory() as temp_dir:
            service, _vault, _data_dir = self._build_service(service_module, Path(temp_dir))
            app = service_module.build_fastapi_app(service)

            live_response = app.routes[("GET", "/live")]()
            self.assertEqual(live_response.status_code, 200)
            self.assertEqual(live_response.content, {"status": "ok", "live": True})

            ready_response = app.routes[("GET", "/ready")]()
            self.assertEqual(ready_response.status_code, 503)
            self.assertFalse(ready_response.content["ready"])

            health_response = app.routes[("GET", "/health")]()
            self.assertEqual(health_response.status_code, 503)
            self.assertTrue(health_response.content["live"])
            self.assertFalse(health_response.content["ready"])

            service.sync()
            ready_response = app.routes[("GET", "/ready")]()
            self.assertEqual(ready_response.status_code, 200)
            self.assertTrue(ready_response.content["ready"])
