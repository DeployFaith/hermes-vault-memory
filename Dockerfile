FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/hvm \
    XDG_CACHE_HOME=/home/hvm/.cache \
    HF_HOME=/home/hvm/.cache/huggingface \
    HVM_VAULT_ROOTS=/vault/root-1:/vault/root-2:/vault/root-3 \
    HVM_DATA_DIR=/data \
    HVM_QDRANT_URL=http://qdrant:6333 \
    HVM_QDRANT_PATH=/data/qdrant \
    HVM_MANIFEST_PATH=/data/manifest.json \
    HVM_SYNC_POLL_SECONDS=60 \
    HVM_SYNC_FULL_RESYNC_SECONDS=21600

WORKDIR /app

COPY . /app/src-build/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir /app/src-build/ && \
    rm -rf /app/src-build/ && \
    groupadd --system --gid 10001 hvm && \
    useradd --system --uid 10001 --gid hvm --home-dir /home/hvm --create-home hvm && \
    mkdir -p /data /vault/root-1 /vault/root-2 /vault/root-3 /home/hvm/.cache/huggingface && \
    chown -R hvm:hvm /app /data /home/hvm /vault

USER hvm

EXPOSE 8787
CMD ["hermes-vault-memory", "serve", "--host", "0.0.0.0", "--port", "8787"]
