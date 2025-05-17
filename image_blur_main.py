from tkinter import *
from tkinter import filedialog
from PIL import Image
import math

blur_strength = 2
mean = 1
a = 255

def gaussian(num, mean, blur_strength):
    return (1 / math.sqrt(2 * math.pi * (blur_strength * 0.5))) * math.pow(math.e, -((num - mean) ** 2) / blur_strength)


file = filedialog.askopenfilename(title="Find image")
img = Image.open(file)
img.show()

pixels = img.load()

width, height = img.size
for y in range(height):
    for x in range(width):
        try:
            r, g, b, a = pixels[x, y]
        except:
            r, g, b = pixels[x, y]

pixel_values = list(img.getdata())

for y in range(height):
    for x in range(width):
        new_r = 0
        new_g = 0
        new_b = 0
        n = 0
        for i in range(mean*2+1):
            for j in range(mean*2+1):
                if (x - mean + i >= 0 and y - mean + j >= 0 and x - mean + i < width and y - mean + j < height):
                    pixel = pixels[x - mean + i, y - mean + j]
                    n = n + 1
                    #print(pixel)
                    try:
                        temp_r, temp_g, temp_b, a = pixel
                    except ValueError:
                        temp_r, temp_g, temp_b = pixel
                    new_r = int(new_r + temp_r * gaussian(n, mean, blur_strength))
                    new_g = int(new_g + temp_g * gaussian(n, mean, blur_strength))
                    new_b = int(new_b + temp_b * gaussian(n, mean, blur_strength))
        try:
            pixels[x, y] = (new_r, new_g, new_b)
        except:
            pixels[x, y] = (new_r, new_g, new_b, a)
        #print(new_r, new_g, new_b)

img.show()