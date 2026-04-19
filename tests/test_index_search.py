from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from types import SimpleNamespace
import shutil
import sys
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hermes_vault_memory.config import Settings
from tests.support.fakes import load_service_module


class QueryOnlyQdrantClient:
    def __init__(self, path: str | None = None, url: str | None = None, **_: object):
        self.path = Path(path) if path else None
        self.url = url
        self._collections: dict[str, dict[str, object]] = {}

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self._collections

    def create_collection(self, collection_name: str, vectors_config) -> None:
        self._collections[collection_name] = {'vectors_config': vectors_config, 'points': {}}

    def delete_collection(self, collection_name: str) -> None:
        self._collections.pop(collection_name, None)

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name=name) for name in self._collections])

    def get_collection(self, collection_name: str):
        collection = self._collections[collection_name]
        size = collection['vectors_config'].size
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=size))))

    def upsert(self, collection_name: str, points) -> None:
        stored = self._collections[collection_name]['points']
        for point in points:
            stored[point.id] = SimpleNamespace(id=point.id, vector=list(point.vector), payload=point.payload or {})

    def delete(self, collection_name: str, points_selector) -> None:
        stored = self._collections[collection_name]['points']
        for point_id in points_selector.points:
            stored.pop(point_id, None)

    def count(self, collection_name: str, exact: bool = True):
        return SimpleNamespace(count=len(self._collections[collection_name]['points']))

    def retrieve(self, collection_name: str, ids, with_payload: bool = True, with_vectors: bool = False):
        stored = self._collections[collection_name]['points']
        results = []
        for point_id in ids:
            point = stored.get(point_id)
            if point is not None:
                results.append(SimpleNamespace(id=point.id, payload=point.payload, score=1.0))
        return results

    def query_points(self, collection_name: str, query, limit: int, query_filter=None, with_payload: bool = True, with_vectors: bool = False, **_: object):
        points = list(self._collections[collection_name]['points'].values())
        ranked = sorted(points, key=lambda point: (-self._score(query, point.vector), point.id))
        return SimpleNamespace(points=[SimpleNamespace(id=point.id, payload=point.payload, score=self._score(query, point.vector)) for point in ranked[:limit]])

    @staticmethod
    def _score(query, vector) -> float:
        return sum(float(left) * float(right) for left, right in zip(query, vector))


class IndexSearchTests(unittest.TestCase):
    def test_startup_rejects_missing_vault_mount(self) -> None:
        service_module = load_service_module()
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing_vault = root / 'missing-vault'
            data_dir = root / 'data'

            with patch.dict(
                'os.environ',
                {
                    'HVM_VAULT_ROOTS': str(missing_vault),
                    'HVM_DATA_DIR': str(data_dir),
                    'HVM_QDRANT_PATH': str(data_dir / 'qdrant'),
                    'HVM_MANIFEST_PATH': str(data_dir / 'manifest.json'),
                },
                clear=True,
            ):
                settings = Settings.load()

            with self.assertRaises(FileNotFoundError):
                service_module.VaultMemoryService(settings)

    def test_sync_and_search_flow_on_sample_markdown(self) -> None:
        service_module = load_service_module()
        fixture_vault = Path(__file__).resolve().parents[1] / 'fixtures' / 'sample-vault'

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vault = root / 'vault'
            data_dir = root / 'data'
            shutil.copytree(fixture_vault, vault)

            with patch.dict(
                'os.environ',
                {
                    'HVM_VAULT_ROOTS': str(vault),
                    'HVM_DATA_DIR': str(data_dir),
                    'HVM_QDRANT_PATH': str(data_dir / 'qdrant'),
                    'HVM_MANIFEST_PATH': str(data_dir / 'manifest.json'),
                    'HVM_COLLECTION_NAME': 'demo_collection',
                },
                clear=True,
            ):
                settings = Settings.load()
                service = service_module.VaultMemoryService(settings)

            summary = service.sync()
            self.assertEqual(summary['indexed_files'], 2)
            self.assertEqual(summary['upserted_chunks'], 2)

            results = service.search('vault mount', limit=5)
            self.assertEqual(results['query'], 'vault mount')
            self.assertGreaterEqual(len(results['results']), 1)
            self.assertEqual(results['results'][0]['relative_path'], 'dokploy-setup.md')
            self.assertIn('vault', results['results'][0]['text'].lower())

            status = service.status()
            self.assertTrue(status['ok'])
            self.assertEqual(status['files_indexed'], 2)
            self.assertEqual(status['vault_counts']['vault']['files'], 2)

    def test_search_uses_query_points_api(self) -> None:
        service_module = load_service_module()
        fixture_vault = Path(__file__).resolve().parents[1] / 'fixtures' / 'sample-vault'

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vault = root / 'vault'
            data_dir = root / 'data'
            shutil.copytree(fixture_vault, vault)

            with patch.dict(
                'os.environ',
                {
                    'HVM_VAULT_ROOTS': str(vault),
                    'HVM_DATA_DIR': str(data_dir),
                    'HVM_QDRANT_PATH': str(data_dir / 'qdrant'),
                    'HVM_MANIFEST_PATH': str(data_dir / 'manifest.json'),
                    'HVM_COLLECTION_NAME': 'demo_collection',
                },
                clear=True,
            ):
                with patch.object(service_module, 'QdrantClient', QueryOnlyQdrantClient):
                    settings = Settings.load()
                    service = service_module.VaultMemoryService(settings)

            service.sync()
            results = service.search('vault mount', limit=5)
            self.assertEqual(results['results'][0]['relative_path'], 'dokploy-setup.md')
            self.assertGreaterEqual(len(results['results']), 1)

    def test_background_sync_transitions_to_complete_without_blocking(self) -> None:
        service_module = load_service_module()
        fixture_vault = Path(__file__).resolve().parents[1] / 'fixtures' / 'sample-vault'

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vault = root / 'vault'
            data_dir = root / 'data'
            shutil.copytree(fixture_vault, vault)

            with patch.dict(
                'os.environ',
                {
                    'HVM_VAULT_ROOTS': str(vault),
                    'HVM_DATA_DIR': str(data_dir),
                    'HVM_QDRANT_PATH': str(data_dir / 'qdrant'),
                    'HVM_MANIFEST_PATH': str(data_dir / 'manifest.json'),
                    'HVM_COLLECTION_NAME': 'demo_collection',
                },
                clear=True,
            ):
                settings = Settings.load()
                service = service_module.VaultMemoryService(settings)

        started = threading.Event()
        release = threading.Event()

        def slow_sync(paths=None):
            started.set()
            release.wait(timeout=5)
            return {'indexed_files': 1}

        service.sync = slow_sync  # type: ignore[method-assign]

        launched = service.start_background_sync()
        self.assertTrue(launched)
        self.assertTrue(started.wait(1))
        self.assertEqual(service.status()['sync_state'], 'running')

        release.set()
        for _ in range(50):
            if service.status()['sync_state'] == 'complete':
                break
            time.sleep(0.05)

        self.assertEqual(service.status()['sync_state'], 'complete')
        self.assertIsNone(service.status()['sync_error'])
