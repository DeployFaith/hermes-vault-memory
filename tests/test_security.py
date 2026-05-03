from __future__ import annotations

import asyncio
from types import SimpleNamespace
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tests.support.fakes import load_service_module


class DummyService:
    def __init__(self, *, enable_mutation_tools: bool = False, auth_token: str | None = None):
        self.settings = SimpleNamespace(
            enable_mutation_tools=enable_mutation_tools,
            auth_token=auth_token,
            collection_name="test_collection",
        )
        self.client = SimpleNamespace(collection_exists=lambda _collection_name: True)
        self._sync_state = "idle"
        self._sync_error = None
        self._sync_monitor_thread = None
        self._last_full_sync_completed_at = None
        self.background_sync_starts = 0
        self.monitor_starts = 0
        self.monitor_stops = 0

    def search(self, **kwargs):
        return {"search": kwargs}

    def get(self, **kwargs):
        return {"get": kwargs}

    def find_notes(self, **kwargs):
        return {"find_notes": kwargs}

    def status(self):
        return {"ok": True, "sync_state": self._sync_state, "files_indexed": 1, "last_sync": "now"}

    def health(self):
        return {"status": "ok", "live": True, "ready": True, "service": self.status()}

    def sync(self, **kwargs):
        return {"sync": kwargs}

    def rebuild(self):
        return {"rebuild": True}

    def start_background_sync(self):
        self.background_sync_starts += 1
        return True

    def start_sync_monitor(self):
        self.monitor_starts += 1
        self._sync_monitor_thread = SimpleNamespace(is_alive=lambda: True)
        return True

    def stop_sync_monitor(self):
        self.monitor_stops += 1
        self._sync_monitor_thread = None
        return True


class McpToolSecurityTests(unittest.TestCase):
    def test_mutation_tools_are_not_registered_by_default(self) -> None:
        service_module = load_service_module()
        mcp = service_module.build_mcp_server(DummyService(enable_mutation_tools=False))

        self.assertEqual(
            set(mcp.tools),
            {"search_vault", "search", "get_note_context", "get", "find_notes", "find_note", "find", "memory_status", "status"},
        )

    def test_mutation_tools_are_registered_when_enabled(self) -> None:
        service_module = load_service_module()
        mcp = service_module.build_mcp_server(DummyService(enable_mutation_tools=True))

        self.assertIn("sync_vault", mcp.tools)
        self.assertIn("sync", mcp.tools)
        self.assertIn("rebuild_vault_index", mcp.tools)
        self.assertIn("rebuild", mcp.tools)
        self.assertIn("search", mcp.tools)
        self.assertIn("get", mcp.tools)
        self.assertIn("find_note", mcp.tools)
        self.assertIn("find", mcp.tools)
        self.assertIn("status", mcp.tools)


class BearerAuthMiddlewareTests(unittest.TestCase):
    def _middleware(self, token: str | None = "secret"):
        service_module = load_service_module()
        app = service_module.build_fastapi_app(DummyService(auth_token=token))
        return app.middlewares[0][1] if app.middlewares else None

    def _request(self, path: str, authorization: str | None = None):
        headers = {}
        if authorization is not None:
            headers["authorization"] = authorization
        return SimpleNamespace(url=SimpleNamespace(path=path), headers=headers)

    def test_auth_middleware_is_not_added_without_token(self) -> None:
        self.assertIsNone(self._middleware(token=None))

    def test_auth_middleware_allows_health_routes_without_token(self) -> None:
        middleware = self._middleware()

        async def call_next(request):
            return SimpleNamespace(status_code=204, path=request.url.path)

        for path in ("/health", "/live", "/ready"):
            with self.subTest(path=path):
                response = asyncio.run(middleware(self._request(path), call_next))
                self.assertEqual(response.status_code, 204)
                self.assertEqual(response.path, path)

    def test_auth_middleware_rejects_non_health_routes_without_bearer_token(self) -> None:
        middleware = self._middleware()

        async def call_next(_request):
            self.fail("call_next should not run for unauthorized requests")

        for path in ("/", "/mcp", "/other"):
            with self.subTest(path=path):
                response = asyncio.run(middleware(self._request(path), call_next))
                self.assertEqual(response.status_code, 401)
                self.assertEqual(response.content, {"detail": "Unauthorized"})

    def test_auth_middleware_allows_valid_bearer_token(self) -> None:
        middleware = self._middleware()

        async def call_next(request):
            return SimpleNamespace(status_code=200, path=request.url.path)

        response = asyncio.run(middleware(self._request("/mcp", "Bearer secret"), call_next))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.path, "/mcp")

    def test_fastapi_lifespan_starts_and_stops_sync_monitor(self) -> None:
        service_module = load_service_module()
        service = DummyService(auth_token=None)
        app = service_module.build_fastapi_app(service)

        self.assertEqual(service.background_sync_starts, 0)
        self.assertEqual(service.monitor_starts, 0)

        async def run_lifespan():
            async with app.lifespan(app):
                self.assertEqual(service.background_sync_starts, 1)
                self.assertEqual(service.monitor_starts, 1)
                self.assertEqual(service.monitor_stops, 0)

        asyncio.run(run_lifespan())

        self.assertEqual(service.monitor_stops, 1)


if __name__ == "__main__":
    unittest.main()
