import matplotlib.pyplot as plt
import random
import sys
import cv2
import skimage.transform
from PIL import Image, ImageEnhance
import math

import numpy as np
import pandas as pd
import sqlite3
import glob

class ImageNormalizer():
    def load_images(self, shape=(100, 100, 3), folder='select_train', epochs=20):
        imgs = []
        for filepath in glob.iglob('./%s/*.jpg' % folder):
            print(filepath)
            img = Image.open(filepath)  # .convert('L')

            for epoch in range(epochs):
                new_img = self.transform_image(img, shape)
                if new_img:
                    print('.', end='', flush=True)
                    new_img.save(filepath.replace(folder, folder + '_transformations').replace('.jpg', '_%d.jpg' % epoch))
                    # img_data = list(new_img.getdata())
                    # imgs.append(np.array(img_data).reshape(shape))
                else:
                    print('x', end='', flush=True)
            print()
        return np.array(imgs)

    def transform_image(self, img, shape):
        # Skew
        ################
        w, h = img.size
        skew = random.random()*2 - 1
        xshift = abs(skew) * w
        new_width = w + int(round(xshift))
        img = img.transform(
            (new_width, h), Image.AFFINE,
            (1, skew, -xshift if skew > 0 else 0, 0, 1, 0), Image.BICUBIC)
        # Rotate
        ################
        theta = random.randint(-180, 180)
        img = img.rotate(theta)
        # Random Crop
        #################
        w, h = img.size
        rand_points = []
        c = 0
        while len(rand_points) < 4:
            if c >= 20:
                return None
            # rand top left corner
            rand_x, rand_y = random.randint(
                0, int(w/3)), random.randint(0, int(h/3))
            # rand side length greater than 150
            rand_side = random.randint(150, min((w-rand_x), (h-rand_y)) - 1)
            rand_points = [(rand_x, rand_y), (rand_x+rand_side, rand_y),
                           (rand_x, rand_y+rand_side), (rand_x+rand_side, rand_y+rand_side)]
            # checks if all corners are part of picture
            for x, y in rand_points:
                if img.getpixel((x, y)) == (0, 0, 0):
                    rand_points.remove((x, y))
            c += 1
        # Sharpening, Brightness, and Contrast
        #################
        sharpener = ImageEnhance.Sharpness(img)
        img = sharpener.enhance(1 + (random.random() - .5) / 2)

        brightener = ImageEnhance.Brightness(img)
        img = brightener.enhance(1 + (random.random() - .5) / 2)

        contraster = ImageEnhance.Contrast(img)
        img = contraster.enhance(1 + (random.random() - .5) / 2)
        # Final Resize
        ##################
        box = (rand_x, rand_y, rand_x+rand_side, rand_y+rand_side)
        img = img.resize(
            (shape[0], shape[1]),
            Image.LANCZOS,
            box)
        # img.show()

        return img
    
if __name__ == '__main__':
    im = ImageNormalizer()
    im.load_images(shape=(300,300,3), epochs=50)
