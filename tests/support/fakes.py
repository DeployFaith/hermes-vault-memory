from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from hashlib import sha256
from math import sqrt
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
import re
import sys

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
EMBED_DIMENSIONS = 128


@dataclass(slots=True)
class PointStruct:
    id: str
    vector: list[float]
    payload: dict[str, Any] | None = None


@dataclass(slots=True)
class PointIdsList:
    points: list[str]


@dataclass(slots=True)
class MatchValue:
    value: Any


@dataclass(slots=True)
class FieldCondition:
    key: str
    match: MatchValue


@dataclass(slots=True)
class Filter:
    must: list[FieldCondition]


@dataclass(slots=True)
class VectorParams:
    size: int
    distance: str


class Distance:
    COSINE = "cosine"


@dataclass(slots=True)
class StoredPoint:
    id: str
    vector: list[float]
    payload: dict[str, Any] | None
    score: float = 0.0


class QdrantClient:
    def __init__(self, path: str | None = None, url: str | None = None, **_: Any):
        self.path = Path(path) if path else None
        self.url = url
        self._collections: dict[str, dict[str, Any]] = {}


    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self._collections

    def create_collection(self, collection_name: str, vectors_config: VectorParams) -> None:
        self._collections[collection_name] = {
            "vectors_config": vectors_config,
            "points": {},
        }

    def delete_collection(self, collection_name: str) -> None:
        self._collections.pop(collection_name, None)

    def get_collections(self) -> SimpleNamespace:
        return SimpleNamespace(collections=[SimpleNamespace(name=name) for name in self._collections])

    def get_collection(self, collection_name: str) -> SimpleNamespace:
        collection = self._collections[collection_name]
        size = collection["vectors_config"].size
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=size))))

    def upsert(self, collection_name: str, points: list[PointStruct]) -> None:
        collection = self._collections[collection_name]
        stored = collection["points"]
        for point in points:
            stored[point.id] = StoredPoint(id=point.id, vector=list(point.vector), payload=point.payload or {})

    def delete(self, collection_name: str, points_selector: PointIdsList) -> None:
        collection = self._collections[collection_name]
        stored = collection["points"]
        for point_id in points_selector.points:
            stored.pop(point_id, None)

    def count(self, collection_name: str, exact: bool = True) -> SimpleNamespace:
        return SimpleNamespace(count=len(self._collections[collection_name]["points"]))

    def retrieve(self, collection_name: str, ids: list[str], with_payload: bool = True, with_vectors: bool = False) -> list[SimpleNamespace]:
        stored = self._collections[collection_name]['points']
        results: list[SimpleNamespace] = []
        for point_id in ids:
            point = stored.get(point_id)
            if not point:
                continue
            results.append(SimpleNamespace(id=point.id, payload=point.payload))
        return results

    def scroll(
        self,
        collection_name: str,
        limit: int = 10,
        offset: str | None = None,
        with_payload: bool | list[str] = True,
        with_vectors: bool = False,
        **_: Any,
    ) -> tuple[list[SimpleNamespace], str | None]:
        points = sorted(self._collections[collection_name]['points'].values(), key=lambda point: point.id)
        start = 0
        if offset is not None:
            ids = [point.id for point in points]
            if offset in ids:
                start = ids.index(offset) + 1
        page = points[start : start + limit]
        next_offset = page[-1].id if start + limit < len(points) and page else None
        return [SimpleNamespace(id=point.id, payload=point.payload) for point in page], next_offset

    def query_points(self, collection_name: str, query: list[float], limit: int, query_filter: Filter | None = None, with_payload: bool = True, with_vectors: bool = False, **_: Any) -> SimpleNamespace:
        points = list(self._collections[collection_name]['points'].values())
        filtered: list[StoredPoint] = []
        for point in points:
            if query_filter and not self._matches_filter(point, query_filter):
                continue
            filtered.append(point)

        ranked = [
            StoredPoint(id=point.id, vector=point.vector, payload=point.payload, score=self._cosine_similarity(query, point.vector))
            for point in filtered
        ]
        ranked.sort(key=lambda item: (-item.score, item.id))
        return SimpleNamespace(points=[SimpleNamespace(id=item.id, payload=item.payload, score=item.score) for item in ranked[:limit]])

    def search(self, collection_name: str, query_vector: list[float], limit: int, query_filter: Filter | None = None, with_payload: bool = True, with_vectors: bool = False) -> list[SimpleNamespace]:
        return list(self.query_points(collection_name, query_vector, limit, query_filter, with_payload, with_vectors).points)

    @staticmethod
    def _matches_filter(point: StoredPoint, qfilter: Filter) -> bool:
        payload = point.payload or {}
        for condition in qfilter.must:
            if payload.get(condition.key) != condition.match.value:
                return False
        return True

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        numerator = sum(l * r for l, r in zip(left, right))
        left_norm = sqrt(sum(v * v for v in left))
        right_norm = sqrt(sum(v * v for v in right))
        if not left_norm or not right_norm:
            return 0.0
        return numerator / (left_norm * right_norm)
class TextEmbedding:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def embed(self, texts):
        return [self._vectorize(text) for text in texts]

    @staticmethod
    def _vectorize(text: str) -> list[float]:
        vector = [0.0] * EMBED_DIMENSIONS
        for token in TOKEN_RE.findall(text.lower()):
            digest = sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % EMBED_DIMENSIONS
            vector[index] += 1.0
            vector[(index + 13) % EMBED_DIMENSIONS] += 0.25
        norm = sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector


class FastAPI:
    def __init__(self, lifespan=None):
        self.lifespan = lifespan
        self.routes: dict[tuple[str, str], Any] = {}
        self.mounted: list[tuple[str, Any]] = []
        self.middlewares: list[tuple[str, Any]] = []

    def mount(self, path: str, app: Any) -> None:
        self.mounted.append((path, app))

    def middleware(self, middleware_type: str):
        def decorator(func):
            self.middlewares.append((middleware_type, func))
            return func

        return decorator

    def get(self, path: str):
        def decorator(func):
            self.routes[("GET", path)] = func
            return func

        return decorator


@dataclass(slots=True)
class JSONResponse:
    status_code: int
    content: Any


class FastMCP:
    def __init__(self, name: str):
        self.name = name
        self.tools: dict[str, Any] = {}

    def tool(self, name: str | None = None):
        def decorator(func):
            self.tools[name or func.__name__] = func
            return func

        return decorator

    def http_app(self, path: str = "/mcp"):
        @asynccontextmanager
        async def lifespan(app):
            yield

        return SimpleNamespace(path=path, lifespan=lifespan)

    def run(self) -> None:
        return None


def combine_lifespans(*lifespans):
    def wrapper(app):
        @asynccontextmanager
        async def lifespan(_):
            yield

        return lifespan(app)

    return wrapper


def install_fake_dependencies() -> None:
    fastapi_module = ModuleType("fastapi")
    fastapi_module.FastAPI = FastAPI
    fastapi_responses = ModuleType("fastapi.responses")
    fastapi_responses.JSONResponse = JSONResponse
    fastapi_module.responses = fastapi_responses

    fastmcp_module = ModuleType("fastmcp")
    fastmcp_module.FastMCP = FastMCP
    fastmcp_utilities = ModuleType("fastmcp.utilities")
    fastmcp_lifespan = ModuleType("fastmcp.utilities.lifespan")
    fastmcp_lifespan.combine_lifespans = combine_lifespans
    fastmcp_utilities.lifespan = fastmcp_lifespan
    fastmcp_module.utilities = fastmcp_utilities

    qdrant_models = ModuleType("qdrant_client.models")
    qdrant_models.PointStruct = PointStruct
    qdrant_models.PointIdsList = PointIdsList
    qdrant_models.Filter = Filter
    qdrant_models.FieldCondition = FieldCondition
    qdrant_models.MatchValue = MatchValue
    qdrant_models.VectorParams = VectorParams
    qdrant_models.Distance = Distance

    qdrant_module = ModuleType("qdrant_client")
    qdrant_module.QdrantClient = QdrantClient
    qdrant_module.models = qdrant_models

    fastembed_module = ModuleType("fastembed")
    fastembed_module.TextEmbedding = TextEmbedding

    sys.modules.update(
        {
            "fastapi": fastapi_module,
            "fastapi.responses": fastapi_responses,
            "fastmcp": fastmcp_module,
            "fastmcp.utilities": fastmcp_utilities,
            "fastmcp.utilities.lifespan": fastmcp_lifespan,
            "qdrant_client": qdrant_module,
            "qdrant_client.models": qdrant_models,
            "fastembed": fastembed_module,
        }
    )


def load_service_module():
    import importlib

    install_fake_dependencies()
    sys.modules.pop("hermes_vault_memory.service", None)
    return importlib.import_module("hermes_vault_memory.service")
