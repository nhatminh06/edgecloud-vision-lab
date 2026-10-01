#!/usr/bin/env bash
set -euo pipefail

image_name="${EDGECLOUD_IMAGE:-edgecloud-vision-worker:phase9}"
project_dir="${EDGECLOUD_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

if ! command -v docker >/dev/null 2>&1; then
  echo "error: Docker is required; install it using the VM operating system documentation" >&2
  exit 2
fi

docker build --tag "${image_name}" "${project_dir}"
docker volume create edgecloud-model-cache >/dev/null
echo "built ${image_name} and prepared edgecloud-model-cache"
