from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hermes_vault_memory.config import Settings


class SettingsTests(unittest.TestCase):
    def test_loads_environment_configuration(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vault = root / 'vault'
            vault.mkdir()
            data_dir = root / 'data'
            qdrant_dir = data_dir / 'qdrant'
            manifest_path = data_dir / 'manifest.json'

            with patch.dict(
                'os.environ',
                {
                    'HVM_VAULT_ROOTS': str(vault),
                    'HVM_DATA_DIR': str(data_dir),
                    'HVM_QDRANT_PATH': str(qdrant_dir),
                    'HVM_QDRANT_URL': 'http://demo-qdrant:6333',
                    'HVM_MANIFEST_PATH': str(manifest_path),
                    'HVM_COLLECTION_NAME': 'demo_collection',
                    'HVM_EMBEDDING_MODEL': 'demo-model',
                    'HVM_CHUNK_SIZE': '900',
                    'HVM_CHUNK_OVERLAP': '123',
                },
                clear=True,
            ):
                settings = Settings.load()

            self.assertEqual(settings.vault_roots, (vault.resolve(),))
            self.assertEqual(settings.data_dir, data_dir.resolve())
            self.assertEqual(settings.qdrant_path, qdrant_dir.resolve())
            self.assertEqual(settings.qdrant_url, 'http://demo-qdrant:6333')
            self.assertEqual(settings.manifest_path, manifest_path.resolve())
            self.assertEqual(settings.collection_name, 'demo_collection')
            self.assertEqual(settings.embedding_model, 'demo-model')
            self.assertEqual(settings.chunk_size, 900)
            self.assertEqual(settings.chunk_overlap, 123)
            self.assertEqual(settings.sync_poll_seconds, 60)
            self.assertEqual(settings.sync_full_resync_seconds, 21600)

    def test_defaults_point_at_generic_mount_paths(self) -> None:
        with patch.dict('os.environ', {}, clear=True):
            settings = Settings.load()

        self.assertEqual(settings.vault_roots, (Path('/vault/root-1'), Path('/vault/root-2'), Path('/vault/root-3')))
        self.assertTrue(str(settings.data_dir).endswith('data'))
        self.assertTrue(str(settings.qdrant_path).endswith('data/qdrant'))
        self.assertEqual(settings.qdrant_url, 'http://qdrant:6333')
        self.assertEqual(settings.sync_poll_seconds, 60)
        self.assertEqual(settings.sync_full_resync_seconds, 21600)
