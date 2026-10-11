"""CustomTkinter front-end: Image / Video / Webcam tabs with live settings."""
from __future__ import annotations

import queue
import threading
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image

from .core import (IMAGE_EXTS, STYLES, VIDEO_EXTS, BlurSettings, FaceDetector, blur_faces,
                   process_video, read_image, webcam_frames, write_image)

PREVIEW_MAX = (860, 560)


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Blur AI")
        self.geometry("1180x720")
        self.minsize(900, 600)
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        self.detector: FaceDetector | None = None
        self.settings = BlurSettings()
        self.events: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.source_image: np.ndarray | None = None   # original, for live re-blur
        self.boxes: list = []
        self.result: np.ndarray | None = None
        self.video_path: Path | None = None
        self._photo: ctk.CTkImage | None = None

        self._build_sidebar()
        self._build_main()
        self._on_mode("Image")
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(50, self._poll)

    # ---- layout ---------------------------------------------------------
    def _build_sidebar(self) -> None:
        bar = ctk.CTkFrame(self, width=280, corner_radius=0)
        bar.pack(side="left", fill="y")
        bar.pack_propagate(False)
        ctk.CTkLabel(bar, text="Blur AI", font=ctk.CTkFont(size=26, weight="bold")).pack(pady=(24, 2))
        ctk.CTkLabel(bar, text="Anonymise faces automatically", text_color="gray").pack(pady=(0, 18))

        self.mode = ctk.CTkSegmentedButton(bar, values=["Image", "Video", "Webcam"], command=self._on_mode)
        self.mode.pack(padx=20, fill="x")
        self.mode.set("Image")

        self._section(bar, "Style")
        self.style_var = ctk.StringVar(value="gaussian")
        ctk.CTkOptionMenu(bar, values=list(STYLES), variable=self.style_var, command=lambda _: self._settings_changed()
                          ).pack(padx=20, fill="x")

        self.strength_label = self._section(bar, "Strength: 60")
        self.strength = ctk.CTkSlider(bar, from_=1, to=100, number_of_steps=99, command=lambda _: self._settings_changed())
        self.strength.set(60)
        self.strength.pack(padx=20, fill="x")

        self.padding_label = self._section(bar, "Padding: 15%")
        self.padding = ctk.CTkSlider(bar, from_=0, to=60, number_of_steps=60, command=lambda _: self._settings_changed())
        self.padding.set(15)
        self.padding.pack(padx=20, fill="x")

        self.ellipse_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(bar, text="Oval mask", variable=self.ellipse_var, command=self._settings_changed
                      ).pack(padx=20, pady=(18, 0), anchor="w")

        self._section(bar, "Detector")
        self.backend_var = ctk.StringVar(value="auto")
        ctk.CTkOptionMenu(bar, values=["auto", "dlib", "opencv"], variable=self.backend_var,
                          command=lambda _: self._reset_detector()).pack(padx=20, fill="x")
        self.detector_label = ctk.CTkLabel(bar, text="", text_color="gray", font=ctk.CTkFont(size=11))
        self.detector_label.pack(padx=20, pady=(4, 0), anchor="w")

        ctk.CTkOptionMenu(bar, values=["System", "Light", "Dark"], width=100,
                          command=lambda v: ctk.set_appearance_mode(v.lower())).pack(side="bottom", pady=16)

    def _section(self, parent, text: str) -> ctk.CTkLabel:
        lbl = ctk.CTkLabel(parent, text=text, anchor="w", font=ctk.CTkFont(weight="bold"))
        lbl.pack(padx=20, pady=(18, 4), fill="x")
        return lbl

    def _build_main(self) -> None:
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(side="left", fill="both", expand=True, padx=16, pady=16)

        top = ctk.CTkFrame(main, fg_color="transparent")
        top.pack(fill="x")
        self.open_btn = ctk.CTkButton(top, text="Open…", width=110, command=self._open)
        self.open_btn.pack(side="left")
        self.run_btn = ctk.CTkButton(top, text="Start", width=110, command=self._run, state="disabled")
        self.run_btn.pack(side="left", padx=8)
        self.save_btn = ctk.CTkButton(top, text="Save…", width=110, command=self._save, state="disabled",
                                      fg_color="transparent", border_width=2, text_color=("gray10", "gray90"))
        self.save_btn.pack(side="left")
        self.file_label = ctk.CTkLabel(top, text="No file loaded", text_color="gray", anchor="w")
        self.file_label.pack(side="left", padx=14, fill="x", expand=True)

        self.preview = ctk.CTkLabel(main, text="Open an image or video to get started",
                                    fg_color=("gray88", "gray17"), corner_radius=12)
        self.preview.pack(fill="both", expand=True, pady=14)

        bottom = ctk.CTkFrame(main, fg_color="transparent")
        bottom.pack(fill="x")
        self.progress = ctk.CTkProgressBar(bottom)
        self.progress.set(0)
        self.progress.pack(fill="x")
        self.status = ctk.CTkLabel(bottom, text="Ready", anchor="w", text_color="gray")
        self.status.pack(fill="x", pady=(6, 0))

    # ---- helpers --------------------------------------------------------
    def _get_detector(self) -> FaceDetector:
        if self.detector is None:
            self.status.configure(text="Loading face detector…")
            self.update_idletasks()
            self.detector = FaceDetector(self.backend_var.get())
            self.detector_label.configure(text=f"Using {self.detector.name}")
        return self.detector

    def _reset_detector(self) -> None:
        self.detector = None
        self.detector_label.configure(text="")
        if self.source_image is not None:
            self._detect_and_show()

    def _settings_changed(self) -> None:
        self.settings = BlurSettings(self.style_var.get(), int(self.strength.get()),
                                     self.padding.get() / 100, self.ellipse_var.get())
        self.strength_label.configure(text=f"Strength: {self.settings.strength}")
        self.padding_label.configure(text=f"Padding: {int(self.padding.get())}%")
        if self.mode.get() == "Image" and self.source_image is not None:
            self._render_image()

    def _show(self, bgr: np.ndarray) -> None:
        h, w = bgr.shape[:2]
        scale = min(PREVIEW_MAX[0] / w, PREVIEW_MAX[1] / h, 1.0)
        size = (max(1, int(w * scale)), max(1, int(h * scale)))
        small = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
        pil = Image.fromarray(small[:, :, ::-1])
        self._photo = ctk.CTkImage(light_image=pil, dark_image=pil, size=size)
        self.preview.configure(image=self._photo, text="")

    def _busy(self, busy: bool) -> None:
        self.open_btn.configure(state="disabled" if busy else "normal")
        self.mode.configure(state="disabled" if busy else "normal")

    # ---- mode handling --------------------------------------------------
    def _on_mode(self, mode: str) -> None:
        self._stop_worker()
        self.source_image = self.result = None
        self.video_path = None
        self.preview.configure(image=None, text={
            "Image": "Open an image to blur its faces",
            "Video": "Open a video to blur its faces",
            "Webcam": "Press Start to blur faces live from your camera"}[mode])
        self.file_label.configure(text="No file loaded" if mode != "Webcam" else "Default camera")
        self.open_btn.configure(state="disabled" if mode == "Webcam" else "normal")
        self.run_btn.configure(text="Start" if mode == "Webcam" else "Blur video",
                               state="normal" if mode == "Webcam" else "disabled")
        if mode == "Image":
            self.run_btn.pack_forget()
        else:
            self.run_btn.pack(side="left", padx=8, after=self.open_btn)
        self.save_btn.configure(state="disabled")
        self.progress.set(0)
        self.status.configure(text="Ready")

    def _open(self) -> None:
        mode = self.mode.get()
        exts = IMAGE_EXTS if mode == "Image" else VIDEO_EXTS
        pattern = " ".join(f"*{e}" for e in sorted(exts))
        path = filedialog.askopenfilename(title=f"Select {mode.lower()}", filetypes=[(mode, pattern), ("All files", "*.*")])
        if not path:
            return
        try:
            if mode == "Image":
                self.source_image = read_image(path)
                self.file_label.configure(text=Path(path).name)
                self._detect_and_show()
            else:
                self.video_path = Path(path)
                cap = cv2.VideoCapture(path)
                ok, frame = cap.read()
                cap.release()
                if not ok:
                    raise OSError("Could not read the video")
                self.file_label.configure(text=self.video_path.name)
                self._show(frame)
                self.run_btn.configure(state="normal")
                self.status.configure(text="Ready – press “Blur video”")
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            messagebox.showerror("Blur AI", str(exc))

    def _detect_and_show(self) -> None:
        self.boxes = self._get_detector().detect(self.source_image)
        self._render_image()

    def _render_image(self) -> None:
        self.result = blur_faces(self.source_image, self.boxes, self.settings)
        self._show(self.result)
        self.save_btn.configure(state="normal")
        n = len(self.boxes)
        self.status.configure(text=f"{n} face{'s' if n != 1 else ''} detected" if n else "No faces detected")

    # ---- run / save -----------------------------------------------------
    def _run(self) -> None:
        if self.worker and self.worker.is_alive():
            self._stop_worker()
            return
        mode = self.mode.get()
        try:
            det = self._get_detector()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Blur AI", str(exc))
            return
        self.stop_event.clear()
        if mode == "Webcam":
            self.worker = threading.Thread(target=self._webcam_loop, args=(det,), daemon=True)
            self.run_btn.configure(text="Stop")
            self.status.configure(text="Live")
        else:
            out = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 video", "*.mp4")],
                                               initialfile=f"{self.video_path.stem}_blurred.mp4")
            if not out:
                return
            self.worker = threading.Thread(target=self._video_job, args=(det, out), daemon=True)
            self.run_btn.configure(text="Cancel")
            self.status.configure(text="Processing…")
            self._busy(True)
        self.worker.start()

    def _webcam_loop(self, det: FaceDetector) -> None:
        try:
            for frame in webcam_frames(0, det, lambda: self.settings, self.stop_event):
                self.events.put(("frame", frame))
            self.events.put(("done", "Camera stopped"))
        except Exception as exc:  # noqa: BLE001
            self.events.put(("error", str(exc)))

    def _video_job(self, det: FaceDetector, out: str) -> None:
        try:
            n = process_video(self.video_path, out, det, self.settings,
                              on_progress=lambda f, fr: self.events.put(("progress", (f, fr))),
                              cancel=self.stop_event)
            msg = f"Cancelled after {n} frames" if self.stop_event.is_set() else f"Saved {n} frames to {out} (audio is not copied)"
            self.events.put(("done", msg))
        except Exception as exc:  # noqa: BLE001
            self.events.put(("error", str(exc)))

    def _stop_worker(self) -> None:
        self.stop_event.set()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=2)

    def _save(self) -> None:
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("WebP", "*.webp")])
        if path:
            write_image(path, self.result)
            self.status.configure(text=f"Saved {Path(path).name}")

    def _poll(self) -> None:
        latest = None
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "frame":
                    latest = payload
                elif kind == "progress":
                    frac, latest = payload
                    self.progress.set(frac)
                    self.status.configure(text=f"Processing… {frac:.0%}")
                else:
                    self._finish(kind, payload)
        except queue.Empty:
            pass
        if latest is not None:
            self._show(latest)
        self.after(30, self._poll)

    def _finish(self, kind: str, msg: str) -> None:
        self._busy(False)
        self.run_btn.configure(text="Start" if self.mode.get() == "Webcam" else "Blur video")
        if kind == "error":
            messagebox.showerror("Blur AI", msg)
            msg = "Error"
        elif self.mode.get() == "Video":
            self.progress.set(0 if self.stop_event.is_set() else 1)
        self.status.configure(text=msg)

    def _close(self) -> None:
        self._stop_worker()
        self.destroy()


def run() -> None:
    App().mainloop()


if __name__ == "__main__":
    run()
