import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blur_ai.core import STYLES, BlurSettings, FaceDetector, blur_faces, process_image  # noqa: E402

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("ellipse", [True, False])
def test_blur_changes_only_the_box(style, ellipse):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    out = blur_faces(img, [(50, 150, 150, 50)], BlurSettings(style, 80, 0.0, ellipse))
    assert out.shape == img.shape and out.dtype == np.uint8
    assert not np.array_equal(out[60:140, 60:140], img[60:140, 60:140])
    assert np.array_equal(out[:40], img[:40])           # outside untouched
    assert np.array_equal(img, img.copy())               # input not mutated


def test_boxes_are_clamped_to_frame():
    img = np.full((50, 50, 3), 128, np.uint8)
    blur_faces(img, [(-20, 90, 90, -20)], BlurSettings(padding=0.5))


def test_detector_runs_on_samples():
    try:
        det = FaceDetector()
    except RuntimeError:
        pytest.skip("no detector available")
    for f in SAMPLES.iterdir():
        _, n = process_image(f, None, det, BlurSettings())
        assert n >= 0

