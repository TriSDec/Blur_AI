from tkinter import * # type: ignore
from tkinter import filedialog
from PIL import Image
import math
import turtle

blur_strength = 0
std_dev = 1.0

mean = 2.0

match blur_strength:
    case 0:
        std_dev = mean/4
    case 1:
        std_dev = mean/2
    case 2:
        std_dev = mean


def gaussian(x, y, mean, blur_strength):
    if y == -1:
        return (1 / math.sqrt(2 * math.pi * (std_dev * 0.5)))*math.pow(math.e, 0 -(((x - mean) ** 2) / std_dev))
    else:
        return (1 / math.sqrt(2 * math.pi * (std_dev * 0.5)))*math.pow(math.e, 0 -(((x - mean) ** 2) / std_dev)) * gaussian(y, -1, mean, blur_strength)

turtle.colormode(255)
screen = turtle.Screen()
t = turtle.Turtle()
t.speed(100000000000)
t.hideturtle()
n=0
m=0
for i in range(int(mean)*2+1):
    n=0
    t.goto(0,i*-10*(10-mean))
    for j in range(int(mean)*2+1):
        r = g = b = 255*gaussian(n, m, mean, blur_strength) # type: ignore
        n=n+1
        t.fillcolor(int(r),int(g),int(b))
        t.begin_fill()
        for _ in range(4):
            t.forward(10*(10-mean))
            t.left(90)
        t.forward(10*(10-mean))
        t.end_fill()
    m=m+1
screen.exitonclick()