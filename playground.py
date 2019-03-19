import cv2
import skimage.transform
import glob
import numpy as np
from PIL import Image
import random

im = Image.open("./bam_train/182189.jpg")
im.show()
im_0 = im.rotate(45)
im_0.show()
im_1 = im.rotate(90)
im_1.show()
im_2 = im.rotate(180)
im_2.show()
im.transpose(Image.FLIP_LEFT_RIGHT)

def load_images():
    imgs = []
    for filepath in glob.iglob('./bam_train/*.jpg'):
        img = cv2.imread(filepath, 3)
        # resize image
        img = skimage.transform.resize(img, (200, 200))
        imgs.append(img)
    return np.array(imgs)

# imgs = load_images()
# print(imgs.shape)
