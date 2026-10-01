from __future__ import annotations

import json
from pathlib import Path

from edgecloud.experiments.replay import export_replay, load_and_validate_jsonl

ROOT = Path(__file__).parent.parent


def test_synthetic_fixture_stays_valid(tmp_path: Path) -> None:
    fixture = ROOT / "tests/fixtures/replay/sample-synthetic.jsonl"
    generated = tmp_path / "replay.json"
    synthetic = export_replay(fixture, generated)

    assert synthetic["run"]["notes"]["synthetic_fixture"] is True
    assert any(event.get("fallback", {}).get("used") for event in synthetic["events"])
    assert len(load_and_validate_jsonl(fixture)) == 13


def test_public_replay_is_real_capture() -> None:
    replay = json.loads((ROOT / "demo-data/adaptive-failover.json").read_text())
    frames = [event for event in replay["events"] if event["event_type"] == "inference"]

    assert replay["run"]["source"]["type"] == "video"
    assert replay["run"].get("notes", {}).get("synthetic_fixture") is not True
    assert len(frames) == 100
    assert any(event["fallback"]["used"] for event in frames)


def test_site_is_static_and_loads_committed_replay() -> None:
    html = (ROOT / "site/index.html").read_text()
    javascript = (ROOT / "site/app.js").read_text()
    styles = (ROOT / "site/styles.css").read_text()

    assert '<script src="app.js" defer></script>' in html
    assert "fetch(DATA_URL" in javascript
    assert "data/adaptive-failover.json" in javascript
    assert "Unsupported replay schema" in javascript
    assert "Sample replay" in html
    assert "Captured run" in html
    assert "[hidden] { display: none !important; }" in styles
    assert not any(name in html.lower() for name in ("react", "next.js", "vite"))


def test_readme_uses_a_real_viewer_screenshot() -> None:
    readme = (ROOT / "README.md").read_text()
    screenshot = ROOT / "docs/assets/edgecloud-replay.png"

    assert "docs/assets/edgecloud-replay.png" in readme
    assert screenshot.stat().st_size > 100_000


def test_pages_workflow_deploys_only_static_site_and_replay() -> None:
    workflow = (ROOT / ".github/workflows/pages.yml").read_text()

    assert "actions/configure-pages@" in workflow
    assert "actions/upload-pages-artifact@" in workflow
    assert "actions/deploy-pages@" in workflow
    assert "cp demo-data/adaptive-failover.json site/data/adaptive-failover.json" in workflow
    assert "path: site" in workflow
