"""Command-line interface: `python -m blur_ai blur INPUT [-o OUTPUT]`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core import IMAGE_EXTS, STYLES, VIDEO_EXTS, BlurSettings, FaceDetector, process_image, process_video


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="blur_ai", description="Blur faces in images and videos.")
    p.add_argument("input", nargs="?", help="image or video file (omit to launch the GUI)")
    p.add_argument("-o", "--output", help="output path (default: <input>_blurred.<ext>)")
    p.add_argument("--style", choices=STYLES, default="gaussian")
    p.add_argument("--strength", type=int, default=60, help="1-100 (default 60)")
    p.add_argument("--padding", type=float, default=0.15, help="grow boxes by this fraction (default 0.15)")
    p.add_argument("--rect", action="store_true", help="use rectangles instead of ovals")
    p.add_argument("--backend", choices=("auto", "dlib", "opencv"), default="auto")
    args = p.parse_args(argv)

    if not args.input:
        from .gui import run
        run()
        return 0

    src = Path(args.input)
    if not src.exists():
        p.error(f"file not found: {src}")
    ext = src.suffix.lower()
    if ext not in IMAGE_EXTS | VIDEO_EXTS:
        p.error(f"unsupported file type: {ext}")
    dst = Path(args.output) if args.output else src.with_name(f"{src.stem}_blurred{'.mp4' if ext in VIDEO_EXTS else ext}")

    settings = BlurSettings(args.style, args.strength, args.padding, not args.rect)
    detector = FaceDetector("opencv" if args.backend == "opencv" else args.backend)
    print(f"Detector: {detector.name}")
    if ext in IMAGE_EXTS:
        _, n = process_image(src, dst, detector, settings)
        print(f"Blurred {n} face(s) -> {dst}")
    else:
        frames = process_video(src, dst, detector, settings,
                               on_progress=lambda f, _: print(f"\r{f:6.1%}", end="", file=sys.stderr))
        print(f"\nWrote {frames} frames -> {dst}  (note: audio is not copied)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
