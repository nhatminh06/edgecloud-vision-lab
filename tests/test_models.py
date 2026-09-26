import pytest

from edgecloud.inference.models import BoundingBox, Detection, InferenceResult, TimingMetrics


def test_result_serialization_includes_detection_count() -> None:
    result = InferenceResult(
        detections=(Detection(BoundingBox(1, 2, 3, 4), "cat", 0.75, 17),),
        timing=TimingMetrics(1.0, 2.0, 3.0, 6.0),
        source="image.jpg",
        frame_index=0,
    )

    serialized = result.to_dict()

    assert serialized["detection_count"] == 1
    assert serialized["detections"][0]["box"] == {"x1": 1, "y1": 2, "x2": 3, "y2": 4}
    assert serialized["timing"]["total_ms"] == 6.0


@pytest.mark.parametrize("score", [-0.1, 1.1])
def test_detection_rejects_invalid_score(score: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        Detection(BoundingBox(0, 0, 1, 1), "object", score, 1)
