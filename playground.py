import cv2
import skimage.transform
import glob
import numpy as np
im = cv2.imread("./bam_train/182189.jpg", 3)


def load_images():
    imgs = []
    for filepath in glob.iglob('./bam_train/*.jpg'):
        img = cv2.imread(filepath, 3)
        # resize image
        img = skimage.transform.resize(img, (200, 200))
        imgs.append(img)
    return np.array(imgs)

imgs = load_images()
print(imgs.shape)
