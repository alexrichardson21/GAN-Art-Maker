import cv2
import skimage.transform
import glob
import numpy as np
from PIL import Image
import random
import math

def load_images():
    imgs = []
    for filepath in glob.iglob('./select_train/*4.jpg'):
        img = Image.open(filepath)
        # resize image
        w, h = img.size
        
        m = random.random()*2 - 1
        xshift = abs(m) * w
        new_width = w + int(round(xshift))
        img = img.transform(
            (new_width, h), Image.AFFINE,
            (1, m, -xshift if m > 0 else 0, 0, 1, 0), Image.BICUBIC)
        
        w = new_width

        theta = random.randint(-180,180)
        off_x = w/2
        off_y = h/2
        corners = [(0, 0), (0, h), (w, 0), (w, h)]
        new_corners = []
        for x, y in corners:
            # translate point to origin
            tempX = x - off_x
            tempY = y - off_y

            # now apply rotation
            rotatedX = tempX*math.cos(theta) - tempY*math.sin(theta)
            rotatedY = tempX*math.sin(theta) + tempY*math.cos(theta)

            # translate back
            x = rotatedX + off_x
            y = rotatedY + off_y
            new_corners.append({'x': x, 'y': y})

        # rand_crop = []
        # rand_crop.append(random.randint(int(new_corners[0]['x']), int(new_corners[1]['x'])))
        # rand_crop.append(random.randint(int(new_corners[0]['y']), int(new_corners[1]['y'])))
        # rand_crop
        # rand_crop.append(random.randint(int(new_corners[2][0]), int(new_corners[3][0])))
        # rand_crop.append(random.randint(int(new_corners[2][1]), int(new_corners[3][1])))
        img = img.rotate(theta)

        rand_points = []

        while len(rand_points) < 4:
            # rand top left corner
            rand_x, rand_y = random.randint(0, int(w/2)), random.randint(0, int(h/2))
            # rand side length greater than 100
            rand_side = random.randint(100, min((w-rand_x),(h-rand_y)))
            rand_points = [(rand_x, rand_y), (rand_x+rand_side, rand_y), 
                (rand_x, rand_y+rand_side), (rand_x+rand_side, rand_y+rand_side)]
            
            for point in rand_points:
                if img.getpixel((point[0], point[1])) == (0,0,0):
                    rand_points.remove(point)
        
        box = (rand_x, rand_y, rand_x+rand_side, rand_y+rand_side)
        
        img = img.resize(
            (500, 500),
            Image.LANCZOS,
            box)
        img.show()

load_images()
# print(imgs.shape)
