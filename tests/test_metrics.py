import pytest

from edgecloud.metrics import StreamMetrics


def test_stream_fps_uses_elapsed_wall_time() -> None:
    metrics = StreamMetrics(started_at=10.0, frame_count=5)

    assert metrics.fps(now=12.0) == pytest.approx(2.5)
