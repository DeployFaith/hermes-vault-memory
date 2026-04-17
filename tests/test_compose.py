from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class ComposeLayoutTests(unittest.TestCase):
    def test_compose_layout_matches_runtime_assumptions(self) -> None:
        compose_text = Path(__file__).resolve().parents[1].joinpath("docker-compose.yml").read_text(encoding="utf-8")

        expected_fragments = [
            'qdrant:',
            'image: qdrant/qdrant:v1.11.5',
            'memory-service:',
            'build: .',
            'command: ["hermes-vault-memory", "serve", "--host", "0.0.0.0", "--port", "8787"]',
            'HVM_VAULT_ROOTS: /vault/agent-main:/vault/psalmbox-main:/vault/katana-main',
            'HVM_DATA_DIR: /data',
            'HVM_QDRANT_URL: http://qdrant:6333',
            'HVM_QDRANT_PATH: /data/qdrant',
            'HVM_MANIFEST_PATH: /data/manifest.json',
            '- "6333:6333"',
            '- "8787:8787"',
            '- /workspace/vaults/agent-main:/vault/agent-main:ro',
            '- /workspace/vaults/psalmbox-main:/vault/psalmbox-main:ro',
            '- /workspace/vaults/katana-main:/vault/katana-main:ro',
            '- hermes-vault-memory-data:/data',
            '- qdrant-storage:/qdrant/storage',
            'restart: unless-stopped',
        ]

        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, compose_text)
