from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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

    def scroll(
        self,
        collection_name: str,
        limit: int = 10,
        offset: str | None = None,
        with_payload: bool | list[str] = True,
        with_vectors: bool = False,
        **_: object,
    ):
        points = sorted(self._collections[collection_name]['points'].values(), key=lambda point: point.id)
        start = 0
        if offset is not None:
            ids = [point.id for point in points]
            if offset in ids:
                start = ids.index(offset) + 1
        page = points[start : start + limit]
        next_offset = page[-1].id if start + limit < len(points) and page else None
        return [SimpleNamespace(id=point.id, payload=point.payload) for point in page], next_offset

    def query_points(self, collection_name: str, query, limit: int, query_filter=None, with_payload: bool = True, with_vectors: bool = False, **_: object):
        points = list(self._collections[collection_name]['points'].values())
        ranked = sorted(points, key=lambda point: (-self._score(query, point.vector), point.id))
        return SimpleNamespace(points=[SimpleNamespace(id=point.id, payload=point.payload, score=self._score(query, point.vector)) for point in ranked[:limit]])

    @staticmethod
    def _score(query, vector) -> float:
        return sum(float(left) * float(right) for left, right in zip(query, vector))


class CountingTextEmbedding:
    instances: list['CountingTextEmbedding'] = []

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.embed_calls: list[list[str]] = []
        self.instances.append(self)

    def embed(self, texts):
        batch = list(texts)
        self.embed_calls.append(batch)
        return [[float((len(text) % 17) + 1), 1.0, 0.5] for text in batch]


class CountingQdrantClient(QueryOnlyQdrantClient):
    instances: list['CountingQdrantClient'] = []

    def __init__(self, path: str | None = None, url: str | None = None, **kwargs: object):
        super().__init__(path=path, url=url, **kwargs)
        self.upsert_batches: list[list[object]] = []
        self.instances.append(self)

    def upsert(self, collection_name: str, points) -> None:
        batch = list(points)
        self.upsert_batches.append(batch)
        super().upsert(collection_name=collection_name, points=batch)


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

    def test_sync_respects_exclude_globs_and_max_file_bytes(self) -> None:
        service_module = load_service_module()
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vault = root / 'vault'
            data_dir = root / 'data'
            archive = vault / 'Archives' / 'OpenCode Sessions'
            archive.mkdir(parents=True)
            (vault / 'keep.md').write_text('# Keep\nUseful durable note.\n')
            (archive / 'skip.md').write_text('# Skip\nRaw transcript dump.\n')
            (vault / 'large.md').write_text('# Large\n' + ('x' * 80))

            with patch.dict(
                'os.environ',
                {
                    'HVM_VAULT_ROOTS': str(vault),
                    'HVM_DATA_DIR': str(data_dir),
                    'HVM_QDRANT_PATH': str(data_dir / 'qdrant'),
                    'HVM_MANIFEST_PATH': str(data_dir / 'manifest.json'),
                    'HVM_COLLECTION_NAME': 'demo_collection',
                    'HVM_EXCLUDE_GLOBS': 'Archives/OpenCode Sessions/**',
                    'HVM_MAX_FILE_BYTES': '64',
                },
                clear=True,
            ):
                settings = Settings.load()
                service = service_module.VaultMemoryService(settings)

            summary = service.sync()
            self.assertEqual(summary['scanned_files'], 1)
            self.assertEqual(summary['indexed_files'], 1)
            self.assertEqual(list(service.manifest['files']), ['vault/keep.md'])

            service.client.upsert(
                collection_name=settings.collection_name,
                points=[
                    service_module.models.PointStruct(
                        id='orphan-point',
                        vector=[0.0] * service.vector_size,
                        payload={'document_key': 'vault/Archives/OpenCode Sessions/skip.md'},
                    )
                ],
            )
            self.assertEqual(service.status()['points_indexed'], 2)

            cleanup = service.sync()
            self.assertEqual(cleanup['removed_orphans'], 1)
            self.assertEqual(service.status()['points_indexed'], 1)

    def test_plan_sync_detects_modified_files(self) -> None:
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

            service.sync()
            note = vault / 'dokploy-setup.md'
            note.write_text(note.read_text() + '\nAdded sync probe line.\n')

            plan = service.plan_sync()
            self.assertIn(str(note), plan['changed_paths'])
            self.assertFalse(plan['needs_full_resync'])

    def test_plan_sync_requests_full_rescan_for_deleted_files(self) -> None:
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

            service.sync()
            note = vault / 'dokploy-setup.md'
            note.unlink()

            plan = service.plan_sync()
            self.assertTrue(plan['needs_full_resync'])

    def test_sync_monitor_tick_dispatches_incremental_and_full_sync(self) -> None:
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

            service.sync()
            note = vault / 'dokploy-setup.md'
            note.write_text(note.read_text() + '\nAdded sync probe line.\n')

            calls: list[list[str] | None] = []

            def fake_start_background_sync(paths=None):
                calls.append(paths)
                return True

            with patch.object(service, 'start_background_sync', side_effect=fake_start_background_sync):
                dispatched = service._sync_monitor_tick()

            self.assertTrue(dispatched)
            self.assertEqual(calls, [[str(note)]])

            note.unlink()
            calls.clear()
            with patch.object(service, 'start_background_sync', side_effect=fake_start_background_sync):
                dispatched = service._sync_monitor_tick()

            self.assertTrue(dispatched)
            self.assertEqual(calls, [None])

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

    def test_document_upsert_batches_embeddings_once_for_multi_chunk_note(self) -> None:
        service_module = load_service_module()
        CountingTextEmbedding.instances.clear()
        CountingQdrantClient.instances.clear()

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vault = root / 'vault'
            data_dir = root / 'data'
            vault.mkdir()
            note = vault / 'multi-chunk.md'
            note.write_text(
                '# Batch Embedding Probe\n\n'
                + '\n\n'.join(
                    f'Paragraph {index} ' + ('batch embedding text ' * 20)
                    for index in range(3)
                ),
                encoding='utf-8',
            )

            with patch.dict(
                'os.environ',
                {
                    'HVM_VAULT_ROOTS': str(vault),
                    'HVM_DATA_DIR': str(data_dir),
                    'HVM_QDRANT_PATH': str(data_dir / 'qdrant'),
                    'HVM_MANIFEST_PATH': str(data_dir / 'manifest.json'),
                    'HVM_COLLECTION_NAME': 'demo_collection',
                    'HVM_CHUNK_SIZE': '250',
                    'HVM_CHUNK_OVERLAP': '0',
                },
                clear=True,
            ):
                with patch.object(service_module, 'TextEmbedding', CountingTextEmbedding), patch.object(
                    service_module, 'QdrantClient', CountingQdrantClient
                ):
                    settings = Settings.load()
                    service = service_module.VaultMemoryService(settings)

            embedder = CountingTextEmbedding.instances[-1]
            client = CountingQdrantClient.instances[-1]
            embedder.embed_calls.clear()  # Ignore the vector-size probe from service startup.

            summary = service.sync(paths=[note])

            self.assertEqual(summary['indexed_files'], 1)
            self.assertGreaterEqual(summary['upserted_chunks'], 2)
            self.assertEqual(len(embedder.embed_calls), 1)
            self.assertEqual(len(embedder.embed_calls[0]), summary['upserted_chunks'])
            self.assertEqual(len(client.upsert_batches), 1)
            self.assertEqual(len(client.upsert_batches[0]), summary['upserted_chunks'])
            for point in client.upsert_batches[0]:
                self.assertEqual(point.id, point.payload['chunk_id'])
                self.assertIn('document_key', point.payload)
                self.assertIn('text', point.payload)

    def test_read_only_calls_return_while_background_sync_is_busy(self) -> None:
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

            sync_started = threading.Event()
            release_sync = threading.Event()
            original_upsert = service._upsert_document

            def blocking_upsert(document):
                sync_started.set()
                release_sync.wait(timeout=5)
                return original_upsert(document)

            with patch.object(service, "_upsert_document", side_effect=blocking_upsert):
                self.assertTrue(service.start_background_sync())
                self.assertTrue(sync_started.wait(5))

                with ThreadPoolExecutor(max_workers=1) as executor:
                    search_future = executor.submit(service.search, "vault mount", 5)
                    search_result = search_future.result(timeout=5)
                self.assertEqual(search_result["query"], "vault mount")

                with ThreadPoolExecutor(max_workers=1) as executor:
                    status_future = executor.submit(service.status)
                    status_result = status_future.result(timeout=5)
                self.assertEqual(status_result["sync_state"], "running")
                self.assertTrue(status_result["ok"])

            release_sync.set()
            if service._sync_thread:
                service._sync_thread.join(timeout=10)
            self.assertEqual(service._sync_state, 'complete')
            self.assertIsNone(service._sync_error)

    def test_mutating_operations_reject_concurrent_sync(self) -> None:
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

            sync_started = threading.Event()
            release_sync = threading.Event()
            original_upsert = service._upsert_document

            def blocking_upsert(document):
                sync_started.set()
                release_sync.wait(timeout=5)
                return original_upsert(document)

            with patch.object(service, '_upsert_document', side_effect=blocking_upsert):
                self.assertTrue(service.start_background_sync())
                self.assertTrue(sync_started.wait(5))

                self.assertFalse(service.start_background_sync())
                with self.assertRaisesRegex(RuntimeError, 'sync already running'):
                    service.sync()
                with self.assertRaisesRegex(RuntimeError, 'sync already running'):
                    service.rebuild()

                release_sync.set()
                if service._sync_thread:
                    service._sync_thread.join(timeout=10)

            self.assertEqual(service._sync_state, 'complete')
            self.assertIsNone(service._sync_error)

    def test_sync_monitor_can_be_stopped(self) -> None:
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
                    'HVM_SYNC_POLL_SECONDS': '60',
                },
                clear=True,
            ):
                settings = Settings.load()
                service = service_module.VaultMemoryService(settings)

            self.assertTrue(service.start_sync_monitor())
            self.assertFalse(service.start_sync_monitor())
            self.assertTrue(service.status()['sync_monitor_alive'])
            self.assertTrue(service.stop_sync_monitor(timeout=1))
            self.assertFalse(service.status()['sync_monitor_alive'])

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
