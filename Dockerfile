FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HVM_VAULT_ROOTS=/vault/agent-main:/vault/psalmbox-main:/vault/katana-main \
    HVM_DATA_DIR=/data \
    HVM_QDRANT_URL=http://qdrant:6333 \
    HVM_QDRANT_PATH=/data/qdrant \
    HVM_MANIFEST_PATH=/data/manifest.json

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

ARG CACHEBUST=1
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

EXPOSE 8787
CMD ["hermes-vault-memory", "serve", "--host", "0.0.0.0", "--port", "8787"]
