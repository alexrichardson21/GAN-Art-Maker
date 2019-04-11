import matplotlib.pyplot as plt
import random
import cv2
import skimage.transform
from PIL import Image, ImageEnhance
import numpy as np
import pandas as pd
import glob

class ImageNormalizer():
    
    def load_and_transform_images(self, shape=(100, 100, 3), folder='./select_train', epochs=20, save_rate=10):

        num_imgs = len(glob.glob('%s/*.jpg' % folder))
        
        # Init empty array to allow the the maximum number of transformations
        imgs = np.zeros((epochs * num_imgs, shape[0], shape[1], shape[2]))
        # Keep track of actual number of images in numpy array
        len_imgs = 0
        
        for filepath in glob.iglob('%s/*.jpg' % folder):
            
            print(filepath)
            img = Image.open(filepath)  # .convert('L')
            
            h, w = img.size

            # If height or width <450 then don't use image 
            if (h > 450 and w > 450):
                # Copies and randomly transforms images n times
                for epoch in range(epochs):
                    
                    new_img = self.transform_image(img, shape)
                    
                    if new_img:
                        print('.', end='', flush=True)
                        if (epoch % save_rate == 0):
                            new_img.save(filepath.replace(folder, folder + '_transformations').replace('.jpg', '_%d.jpg' % epoch))
                        img_data = np.array(list(new_img.getdata())).reshape(shape)
                        imgs[len_imgs] = img_data
                        len_imgs += 1
                        print('.', end='', flush=True)
                    
                    else:
                        print('x', end='', flush=True)
            print() 
        
        # Returns numpy array of image data without extra zeros at end of array
        return imgs[:len_imgs]

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
            
            # rand side length greater than 300
            rand_side = random.randint(300, min((w-rand_x), (h-rand_y)) - 1)
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
    
# if __name__ == '__main__':
#     im = ImageNormalizer()
#     im.load_images(shape=(300,300,3), epochs=50)
