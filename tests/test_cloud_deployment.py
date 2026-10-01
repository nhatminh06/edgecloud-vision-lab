from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPTS = tuple((ROOT / "deploy" / "cloud").glob("*.sh"))


@pytest.mark.parametrize("script", SCRIPTS)
def test_cloud_deployment_scripts_have_valid_bash_syntax(script: Path) -> None:
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_cloud_run_script_uses_existing_service_and_restricted_configuration() -> None:
    script = (ROOT / "deploy" / "cloud" / "run.sh").read_text(encoding="utf-8")

    assert "edgecloud-serve" in script
    assert "EDGECLOUD_WORKER_ID" in script
    assert "EDGECLOUD_PORT" in script
    assert "EDGECLOUD_DEVICE" in script
    assert "aws" not in script.lower()
    assert "azure" not in script.lower()
    assert "gcloud" not in script.lower()
