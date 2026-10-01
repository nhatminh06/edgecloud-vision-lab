from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Any

import psutil


def collect_environment(
    *, backend: str, execution_resource: str, gpu_name: str | None = None
) -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "total_memory_bytes": int(psutil.virtual_memory().total),
        "gpu_name": gpu_name,
        "inference_backend": backend,
        "execution_resource": execution_resource,
        "project_version": "0.1.0",
        "git_commit": _git_commit(),
        "python_executable": sys.executable,
    }


def _cpu_model() -> str | None:
    model = platform.processor().strip()
    if model:
        return model
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", maxsplit=1)[1].strip()
    except OSError:
        return None
    return None


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None
