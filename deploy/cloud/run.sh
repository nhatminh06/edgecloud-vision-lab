#!/usr/bin/env bash
set -euo pipefail

image_name="${EDGECLOUD_IMAGE:-edgecloud-vision-worker:phase9}"
container_name="${EDGECLOUD_CONTAINER:-edgecloud-vision-worker}"
bind_host="${EDGECLOUD_BIND_HOST:-0.0.0.0}"
port="${EDGECLOUD_PORT:-8000}"
worker_id="${EDGECLOUD_WORKER_ID:-cloud-cpu-1}"
device="${EDGECLOUD_DEVICE:-cpu}"
confidence="${EDGECLOUD_CONFIDENCE:-0.5}"

if docker container inspect "${container_name}" >/dev/null 2>&1; then
  echo "error: container ${container_name} already exists; stop or remove it first" >&2
  exit 2
fi

docker run --detach \
  --name "${container_name}" \
  --restart unless-stopped \
  --publish "${bind_host}:${port}:8000" \
  --volume edgecloud-model-cache:/root/.cache/torch \
  "${image_name}" \
  edgecloud-serve \
  --host 0.0.0.0 \
  --port 8000 \
  --worker-id "${worker_id}" \
  --device "${device}" \
  --confidence "${confidence}"

echo "started ${container_name} on ${bind_host}:${port}"
