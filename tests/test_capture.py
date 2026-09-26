from pathlib import Path

import cv2
import numpy as np
import pytest

from edgecloud.capture import image_frames


def test_image_frames_reads_one_frame(tmp_path: Path) -> None:
    path = tmp_path / "input.png"
    assert cv2.imwrite(str(path), np.zeros((5, 7, 3), dtype=np.uint8))

    frames = list(image_frames(path))

    assert len(frames) == 1
    assert frames[0].image.shape == (5, 7, 3)
    assert frames[0].index == 0


def test_image_frames_reports_unreadable_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="could not read image"):
        list(image_frames(tmp_path / "missing.png"))
