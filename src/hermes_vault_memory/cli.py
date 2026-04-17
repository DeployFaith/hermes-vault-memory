from __future__ import annotations

import argparse
import json
from typing import Any

import uvicorn

from .service import VaultMemoryService, build_fastapi_app, build_mcp_server, load_settings


def _service() -> VaultMemoryService:
    return VaultMemoryService(load_settings())


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(prog="hermes-vault-memory")
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve", help="Run the HTTP service with /health and /mcp")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8787)

    subparsers.add_parser("stdio", help="Run the MCP server on stdio")
    subparsers.add_parser("status", help="Print service status as JSON")
    sync_parser = subparsers.add_parser("sync", help="Sync the configured vaults")
    sync_parser.add_argument("paths", nargs="*", help="Optional files to sync")
    rebuild_parser = subparsers.add_parser("rebuild", help="Drop and rebuild the index")
    rebuild_parser.add_argument("--serve-after", action="store_true", help="Rebuild then start HTTP server")
    rebuild_parser.add_argument("--host", default="0.0.0.0")
    rebuild_parser.add_argument("--port", type=int, default=8787)

    args = parser.parse_args()
    command = args.command or "serve"
    service = _service()

    if command == "serve":
        app = build_fastapi_app(service)
        uvicorn.run(app, host=args.host, port=args.port)
        return

    if command == "stdio":
        mcp = build_mcp_server(service)
        mcp.run()
        return

    if command == "status":
        _print_json(service.status())
        return

    if command == "sync":
        _print_json(service.sync(paths=args.paths or None))
        return

    if command == "rebuild":
        summary = service.rebuild()
        _print_json(summary)
        if args.serve_after:
            app = build_fastapi_app(service)
            uvicorn.run(app, host=args.host, port=args.port)
        return

    raise SystemExit(f"Unknown command: {command}")
