#!/usr/bin/env bash
set -euo pipefail

container_name="${EDGECLOUD_CONTAINER:-edgecloud-vision-worker}"

if ! docker container inspect "${container_name}" >/dev/null 2>&1; then
  echo "container ${container_name} does not exist"
  exit 0
fi

docker stop "${container_name}"
if [[ "${EDGECLOUD_REMOVE_CONTAINER:-0}" == "1" ]]; then
  docker rm "${container_name}"
fi
