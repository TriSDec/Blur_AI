# Blur AI

Automatically find and anonymise faces in **images, videos, and live webcam** feeds — through a desktop GUI or the command line.

## Features

- **GUI** (CustomTkinter): Image / Video / Webcam tabs, live preview that updates as you move the sliders, light/dark theme.
- **Three styles**: Gaussian blur, pixelate, solid block. Adjustable strength, padding, and oval or rectangular masks.
- **Two detectors**: `face_recognition` (dlib, more accurate) if installed, otherwise OpenCV's built-in Haar cascade — no extra setup required.
- **Video**: processed on a background thread with a progress bar and cancel; detection runs every other frame for speed.
- **Unicode-safe I/O** and a single shared core used by both GUI and CLI (no duplicated blur code).

## Install

```sh
pip install -r requirements.txt
# optional, better accuracy:
pip install face_recognition
```

## Use

```sh
python -m blur_ai                      # launch the GUI
python -m blur_ai photo.jpg            # -> photo_blurred.jpg
python -m blur_ai clip.mp4 --style pixelate --strength 80 -o out.mp4
python -m blur_ai --help
```

Try it on the files in `samples/`.

## Layout

```
blur_ai/core.py   detection + blur primitives
blur_ai/gui.py    desktop app
blur_ai/cli.py    command line
tests/            pytest suite
samples/          example images
```

## Notes

- Output video is written with OpenCV's `mp4v` codec and **does not carry the original audio**; re-mux with `ffmpeg -i out.mp4 -i original.mp4 -map 0:v -map 1:a -c copy final.mp4` if you need it.
- Automatic detection can miss faces (small, turned away, occluded). Review the result before publishing sensitive footage.
