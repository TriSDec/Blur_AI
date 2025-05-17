from tkinter import *
from tkinter import filedialog
from PIL import Image
import math

blur_strength = 0
std_dev = 1

mean = 2

if blur_strength == 0:
    std_dev = mean/4

def gaussian(x, mean, blur_strength):
    return (1 / math.sqrt(2 * math.pi * (std_dev * 0.5)))*math.pow(math.e, 0 -(((x - mean) ** 2) / std_dev))

for i in range(mean*2+1):
    print(gaussian(i,mean,2))

for y in range(40):
    for x in range(40):
        n=0
        for i in range(mean*2+1):
            for j in range(mean*2+1):
                n=n+1
                if (x - mean + i >= 0 and y - mean + j >= 0 and x - mean + i < 40 and y - mean + j < 40):
                    print(f"x = {x}, y = {y}")
                    print(gaussian(n, mean, blur_strength))