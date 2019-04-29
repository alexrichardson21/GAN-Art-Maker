import glob
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.transform
from PIL import Image, ImageEnhance


class ImageNormalizer():
    
    def load_images(self, shape, folder):
        num_imgs = len(glob.glob('%s/*.jpg' % folder))

        # Init empty array to hold half of maximum number of images
        imgs = np.zeros(
            (num_imgs, shape[0], shape[1], shape[2]), dtype=np.float16
        )
        x, y, _ = shape
        len_imgs = 0
        
        for filepath in glob.iglob('%s/*.jpg' % folder):

            img = Image.open(filepath)
            if img.size == (x, y):
                img_data = np.array(
                    list(img.getdata()), dtype=np.dtype(np.uint16)
                )
                # Make pixel values -1 to 1
                img_data = (img_data.astype(np.float16) - 127.5) / 127.5
                imgs[len_imgs] = img_data.reshape(shape)
                len_imgs += 1
        
        return imgs[:len_imgs]

    def load_and_transform_images(self, shape, folder, epochs=50, save_rate=10):

        num_imgs = len(glob.glob('%s/*.jpg' % folder))
        
        # Init empty array to hold half of maximum number of images 
        imgs = np.zeros((epochs * num_imgs // 2, shape[0], shape[1], shape[2]), dtype=np.float16)
        
        # Keep track of actual number of images in numpy array
        len_imgs = 0
        
        for i, filepath in enumerate(glob.iglob('%s/*.jpg' % folder)):
            
            img = Image.open(filepath)
            print('Transforming image: %d / %d' % (i+1, num_imgs))
            print('\t' + filepath)
            print('\t Dimensions: ' + str(img.size))
            
            # Copies and randomly transforms images n times
            for epoch in range(epochs):
                # If array is full => return
                if len_imgs >= len(imgs):
                    print('Training set is full at size: %d\n' % len_imgs)
                    return imgs
                
                try:
                    # New randomly transformed image
                    new_img = self.transform_image(img, shape, trials=80)
                    
                    # If properly transformed and cropped
                    if new_img:
                    
                        # Save if epoch is multiple of save_rate
                        if (epoch % save_rate == 0):
                            new_img.save(
                                filepath.replace(folder, folder + '_transformations').replace('.jpg', '_%d.jpg' % epoch)
                            )
                        
                        # Save image data into imgs array
                        img_data = np.array(
                            list(new_img.getdata()), dtype=np.dtype(np.uint16)
                        )
                        # Make pixel values -1 to 1
                        img_data = (img_data.astype(np.float16) - 127.5) / 127.5
                        # Add img data to imgs array
                        imgs[len_imgs] = img_data.reshape(shape)
                        len_imgs += 1
                        print('.', end='', flush=True)
                    
                    # If unable to crop 
                    else:
                        print('x', end='', flush=True)
                
                # If error
                except:
                    print('e', end='', flush=True)
            
            print('\n\n')
        
        # Returns numpy array of image data without extra zeros at end of array
        print('Training set is size: %d\n' % len_imgs)
        return imgs[:len_imgs]

    def transform_image(self, img, shape, trials=80):
        # Skew
        ################
        w, h = img.size
        skew = random.random()*2 - 1
        xshift = abs(skew) * w
        new_width = w + int(round(xshift))
        img = img.transform(
            (new_width, h), 
            Image.AFFINE,
            (1, skew, -xshift if skew > 0 else 0, 0, 1, 0), 
            Image.BICUBIC
        )
        
        # Rotate
        ################
        theta = random.randint(-180, 180)
        img = img.rotate(theta)
        
        # Random Crop
        #################
        w, h = img.size
        rand_points = []
        c = 0
        
        while c < trials:
            try: 
                # rand top left corner
                rand_x, rand_y = random.randint(
                    0, int(w/3)), random.randint(0, int(h/3)
                )
                
                # rand side length greater than half shape dimensions
                rand_side = random.randint(
                    min(shape[:2]), min((w-rand_x), (h-rand_y)) - 1
                )
                
                # generates the set of corners for random crop
                rand_points = [(rand_x, rand_y), (rand_x+rand_side, rand_y),
                            (rand_x, rand_y+rand_side), (rand_x+rand_side, rand_y+rand_side)]
                
                # if all corners are part of image => move on
                for x, y in rand_points:
                    if img.getpixel((x, y)) == (0, 0, 0):
                            rand_points.remove((x, y))
                if len(rand_points) == 4:
                    break
                
                # Try again
                c+=1
            
            # If exception occurred => try again
            except:
                c+=1
        
        # If couldn't generate image in n trials => return None
        if c >= trials:
            return None
       
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
