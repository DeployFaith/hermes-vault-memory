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

from hermes_vault_memory.config import Settings, vault_targets


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
                    'HVM_EXCLUDE_GLOBS': 'Archives/OpenCode Sessions/**, tmp/*.md ',
                    'HVM_MAX_FILE_BYTES': '2048',
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
            self.assertEqual(settings.exclude_globs, ('Archives/OpenCode Sessions/**', 'tmp/*.md'))
            self.assertEqual(settings.max_file_bytes, 2048)
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
        self.assertIsNone(settings.auth_token)
        self.assertFalse(settings.enable_mutation_tools)
        self.assertEqual(settings.exclude_globs, ())
        self.assertIsNone(settings.max_file_bytes)

    def test_loads_named_vaults_from_environment(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / 'first-vault'
            second = root / 'second-vault'
            first.mkdir()
            second.mkdir()

            with patch.dict(
                'os.environ',
                {
                    'HVM_VAULTS': f'alpha:{first}, beta:{second}',
                    'HVM_VAULT_ROOTS': str(root / 'ignored'),
                    'HVM_AUTH_TOKEN': ' secret-token ',
                    'HVM_ENABLE_MUTATION_TOOLS': 'true',
                },
                clear=True,
            ):
                settings = Settings.load()

            self.assertEqual(settings.vault_roots, (first.resolve(), second.resolve()))
            self.assertEqual(
                [(target.vault, target.root) for target in vault_targets(settings)],
                [('alpha', first.resolve()), ('beta', second.resolve())],
            )
            self.assertEqual(settings.auth_token, 'secret-token')
            self.assertTrue(settings.enable_mutation_tools)

    def test_rejects_invalid_chunk_settings(self) -> None:
        invalid_envs = (
            {'HVM_CHUNK_SIZE': '0'},
            {'HVM_CHUNK_SIZE': '100', 'HVM_CHUNK_OVERLAP': '-1'},
            {'HVM_CHUNK_SIZE': '100', 'HVM_CHUNK_OVERLAP': '100'},
        )
        for env in invalid_envs:
            with self.subTest(env=env):
                with patch.dict('os.environ', env, clear=True):
                    with self.assertRaises(ValueError):
                        Settings.load()

    def test_rejects_invalid_sync_settings(self) -> None:
        invalid_envs = (
            {'HVM_SYNC_POLL_SECONDS': '0'},
            {'HVM_SYNC_POLL_SECONDS': '60', 'HVM_SYNC_FULL_RESYNC_SECONDS': '59'},
        )
        for env in invalid_envs:
            with self.subTest(env=env):
                with patch.dict('os.environ', env, clear=True):
                    with self.assertRaises(ValueError):
                        Settings.load()

    def test_rejects_invalid_max_file_bytes(self) -> None:
        with patch.dict('os.environ', {'HVM_MAX_FILE_BYTES': '0'}, clear=True):
            with self.assertRaises(ValueError):
                Settings.load()

    def test_rejects_duplicate_named_vault_names(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / 'first'
            second = root / 'second'
            with patch.dict('os.environ', {'HVM_VAULTS': f'dupe:{first},dupe:{second}'}, clear=True):
                with self.assertRaises(ValueError):
                    Settings.load()

    def test_rejects_duplicate_named_vault_roots(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vault = root / 'vault'
            with patch.dict('os.environ', {'HVM_VAULTS': f'one:{vault},two:{vault}'}, clear=True):
                with self.assertRaises(ValueError):
                    Settings.load()

    def test_rejects_duplicate_vault_roots_fallback(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vault = root / 'vault'
            with patch.dict('os.environ', {'HVM_VAULT_ROOTS': f'{vault}:{vault}'}, clear=True):
                with self.assertRaises(ValueError):
                    Settings.load()

    def test_rejects_duplicate_vault_names_fallback(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / 'one' / 'vault'
            second = root / 'two' / 'vault'
            with patch.dict('os.environ', {'HVM_VAULT_ROOTS': f'{first}:{second}'}, clear=True):
                with self.assertRaises(ValueError):
                    Settings.load()
