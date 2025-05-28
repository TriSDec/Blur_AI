import face_recognition
import numpy as np
from tkinter import filedialog
from PIL import Image
import cv2
import math



def gaussian_kernel(size, sigma):
    """Generate a normalized 2D Gaussian kernel."""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

def video():
    file_path = filedialog.askopenfilename(title="Select an image")
    def blur_face_region(frame, top, right, bottom, left, kernel_size=31, sigma=10):
        """Apply Gaussian blur to a region of interest (face)."""
        face = frame[top:bottom, left:right]
        kernel = gaussian_kernel(kernel_size, sigma)
        for i in range(3):  # RGB channels
            face[..., i] = cv2.filter2D(face[..., i], -1, kernel)
        frame[top:bottom, left:right] = face
        return frame

    # === Load video ===
    input_path = file_path  # Change this to your video filename
    output_path = 'blurred_output.mp4'

    video = cv2.VideoCapture(input_path)
    if not video.isOpened():
        raise IOError("❌ Could not open input video")

    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        raise IOError("❌ Could not open output file")

    print("✅ Starting face-blur processing...")

    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break

        rgb = frame[:, :, ::-1]
        faces = face_recognition.face_locations(rgb)

        for top, right, bottom, left in faces:
            frame = blur_face_region(frame, top, right, bottom, left)

        out.write(frame)
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    video.release()
    out.release()
    cv2.destroyAllWindows()
    print("🎉 Done. Output saved to:", output_path)

def image():
    def blur_face_region(image_array, top, right, bottom, left, kernel_size=15, sigma=5):
        """Apply Gaussian blur to a specific rectangular region in an image."""
        face_region = image_array[top:bottom, left:right]

        # Create the Gaussian kernel
        kernel = gaussian_kernel(kernel_size, sigma)

        # Blur each channel (R, G, B)
        for i in range(3):
            face_region[..., i] = cv2.filter2D(face_region[..., i], -1, kernel)

        image_array[top:bottom, left:right] = face_region
        return image_array

    # === Load image ===
    file_path = filedialog.askopenfilename(title="Select an image")
    image = face_recognition.load_image_file(file_path)
    face_locations = face_recognition.face_locations(image)

    if not face_locations:
        print("❌ No faces detected in the image.")
        exit()

    print(f"✅ Detected {len(face_locations)} face(s).")

    # === Blur faces ===
    image_blurred = image.copy()

    for face in face_locations:
        top, right, bottom, left = face
        image_blurred = blur_face_region(image_blurred, top, right, bottom, left, kernel_size=31, sigma=10)

    # === Show result ===
    blurred_pil = Image.fromarray(image_blurred)
    blurred_pil.show()

def webcam() :
    def blur_face_region(image_array, top, right, bottom, left, kernel_size=31, sigma=10):
        """Apply Gaussian blur to a specific rectangular region in an image."""
        face_region = image_array[top:bottom, left:right]
        kernel = gaussian_kernel(kernel_size, sigma)

        # Apply convolution to each channel
        for i in range(3):
            face_region[..., i] = cv2.filter2D(face_region[..., i], -1, kernel)

        image_array[top:bottom, left:right] = face_region
        return image_array

    # Start webcam
    video_capture = cv2.VideoCapture(0)
    print("🎥 Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convert to RGB (face_recognition expects RGB)
        rgb_frame = frame[:, :, ::-1]

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)

        # Apply blur to each detected face
        for (top, right, bottom, left) in face_locations:
            rgb_frame = blur_face_region(rgb_frame, top, right, bottom, left)

        # Convert back to BGR for OpenCV display
        output_frame = rgb_frame[:, :, ::-1]
        cv2.imshow('Live Face Blurring', output_frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    print("🛑 Webcam stopped.")

mode = input("Do you want live recognition or picture recognition?")

if mode == 'live':
    webcam()
elif mode == 'img':
    image()
else:
    video()