"""Face detection and anonymisation primitives shared by the GUI and CLI."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # (top, right, bottom, left)

STYLES = ("gaussian", "pixelate", "solid")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


@dataclass
class BlurSettings:
    style: str = "gaussian"
    strength: int = 60        # 1-100, higher = more anonymous
    padding: float = 0.15     # grow each box by this fraction of its size
    ellipse: bool = True      # blur an oval instead of a rectangle


class FaceDetector:
    """Wraps `face_recognition` (dlib) when available, else OpenCV's Haar cascade."""

    def __init__(self, backend: str = "auto", detect_width: int = 640):
        self.detect_width = detect_width
        self._fr = None
        self._cascade = None
        if backend in ("auto", "dlib"):
            try:
                import face_recognition  # type: ignore

                self._fr = face_recognition
            except ImportError:
                if backend == "dlib":
                    raise
        if self._fr is None:
            if not hasattr(cv2, "CascadeClassifier"):
                raise RuntimeError("No face detector available: install face_recognition, or use opencv-python<5 (OpenCV 5 removed Haar cascades).")
            path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            self._cascade = cv2.CascadeClassifier(str(path))
            if self._cascade.empty():
                raise RuntimeError(f"Could not load Haar cascade at {path}")

    @property
    def name(self) -> str:
        return "dlib (face_recognition)" if self._fr else "OpenCV Haar cascade"

    def detect(self, bgr: np.ndarray) -> list[Box]:
        h, w = bgr.shape[:2]
        scale = min(1.0, self.detect_width / max(w, 1))
        small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else bgr
        if self._fr is not None:
            rgb = np.ascontiguousarray(small[:, :, ::-1])
            boxes = self._fr.face_locations(rgb)
        else:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            found = self._cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            boxes = [(y, x + bw, y + bh, x) for x, y, bw, bh in found]
        return [tuple(int(v / scale) for v in b) for b in boxes]  # type: ignore[misc]


def _expand(box: Box, pad: float, w: int, h: int) -> Box:
    top, right, bottom, left = box
    dy, dx = int((bottom - top) * pad), int((right - left) * pad)
    return max(0, top - dy), min(w, right + dx), min(h, bottom + dy), max(0, left - dx)


def _anonymise_roi(roi: np.ndarray, s: BlurSettings) -> np.ndarray:
    h, w = roi.shape[:2]
    amount = max(1, min(100, s.strength)) / 100
    if s.style == "pixelate":
        blocks = max(2, int(max(w, h) * (1 - amount) / 4) + 2)
        small = cv2.resize(roi, (max(1, w * blocks // max(w, h)), max(1, h * blocks // max(w, h))), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    if s.style == "solid":
        return np.full_like(roi, 0)
    k = int(max(w, h) * amount * 0.5) | 1  # odd kernel scaled to the face size
    return cv2.GaussianBlur(roi, (max(3, k), max(3, k)), 0)


def blur_faces(frame: np.ndarray, boxes: Sequence[Box], s: BlurSettings) -> np.ndarray:
    """Return a copy of `frame` (BGR) with every box anonymised."""
    out = frame.copy()
    fh, fw = out.shape[:2]
    for box in boxes:
        top, right, bottom, left = _expand(box, s.padding, fw, fh)
        if bottom - top < 2 or right - left < 2:
            continue
        roi = out[top:bottom, left:right]
        processed = _anonymise_roi(roi, s)
        if s.ellipse:
            mask = np.zeros(roi.shape[:2], np.uint8)
            cv2.ellipse(mask, ((right - left) // 2, (bottom - top) // 2),
                        ((right - left) // 2, (bottom - top) // 2), 0, 0, 360, 255, -1)
            mask = cv2.GaussianBlur(mask, (7, 7), 0)[..., None] / 255.0
            processed = (processed * mask + roi * (1 - mask)).astype(np.uint8)
        out[top:bottom, left:right] = processed
    return out


def read_image(path: str | Path) -> np.ndarray:
    # imdecode copes with non-ASCII paths and formats like .jfif that imread can reject
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def write_image(path: str | Path, img: np.ndarray) -> None:
    ext = Path(path).suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"Could not encode image as {ext}")
    buf.tofile(str(path))


def process_image(src: str | Path, dst: str | Path | None, detector: FaceDetector, s: BlurSettings) -> tuple[np.ndarray, int]:
    img = read_image(src)
    boxes = detector.detect(img)
    result = blur_faces(img, boxes, s)
    if dst:
        write_image(dst, result)
    return result, len(boxes)


def process_video(
    src: str | Path,
    dst: str | Path,
    detector: FaceDetector,
    s: BlurSettings,
    on_progress: Callable[[float, np.ndarray], None] | None = None,
    cancel: threading.Event | None = None,
    detect_every: int = 2,
) -> int:
    """Blur faces in a video. Returns the number of frames written.

    Detection runs every `detect_every` frames and the last boxes are reused in
    between, which roughly doubles throughput with little visible lag.
    """
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {src}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not out.isOpened():
        cap.release()
        raise OSError(f"Could not create output: {dst}")
    n, boxes = 0, []
    try:
        while not (cancel and cancel.is_set()):
            ok, frame = cap.read()
            if not ok:
                break
            if n % max(1, detect_every) == 0:
                boxes = detector.detect(frame)
            result = blur_faces(frame, boxes, s)
            out.write(result)
            n += 1
            if on_progress and (n % 5 == 0 or n == total):
                on_progress(n / total if total else 0.0, result)
    finally:
        cap.release()
        out.release()
    return n


def webcam_frames(index: int, detector: FaceDetector, get_settings: Callable[[], BlurSettings],
                  stop: threading.Event) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise OSError(f"Could not open camera {index}")
    try:
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            yield blur_faces(frame, detector.detect(frame), get_settings())
    finally:
        cap.release()

