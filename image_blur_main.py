from tkinter import filedialog
from PIL import Image
import numpy as np
import math
import scipy.ndimage

def gaussian_kernel(size, sigma):
    """Generate a 2D Gaussian kernel."""
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

# Select and load image
file = filedialog.askopenfilename(title="Find image")
img = Image.open(file).convert("RGB")  # Or "RGBA" if alpha is needed

# Convert to NumPy array
img_np = np.array(img)
height, width = img_np.shape[:2]

# Set blur parameters
blur_strength = 0
mean = height // 25
sigma = {0: mean/8, 1: mean/4, 2: mean/2, 3: mean, 4: mean*2}[blur_strength]
kernel_size = int(2 * mean + 1)

# Precompute Gaussian kernel
kernel = gaussian_kernel(kernel_size, sigma)

# Apply convolution on each channel separately
blurred = np.zeros_like(img_np)
for c in range(3):  # For R, G, B channels
    blurred[..., c] = scipy.ndimage.convolve(img_np[..., c], kernel, mode='reflect')

# Convert back to image
blurred_img = Image.fromarray(blurred)
blurred_img.show()