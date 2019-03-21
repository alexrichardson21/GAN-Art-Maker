from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import random
import sys
import cv2
import skimage.transform
from PIL import Image
import math


import numpy as np
import pandas as pd
import sqlite3
import glob


class GAN():
    def __init__(self):
        self.img_rows = 500
        self.img_cols = 500
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        noise_shape = (100,)

        model = Sequential()

        model.add(Dense(256, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()

        model.add(Flatten(input_shape=img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        X_train = self.load_images(100)

        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        # X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(
                imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(
                gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                  (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 3, 3
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/art_%d.png" % epoch)
        plt.close()

    def load_images(self, epochs):
        imgs = []
        for filepath in glob.iglob('./select_train/*.jpg'):
            print(filepath)
            img = Image.open(filepath) #.convert('L')
            
            for _ in range(epochs):
                new_img = self.transform_image(img)
                if new_img:
                    print('.', end='', flush=True)
                    img_data = list(new_img.getdata())
                    imgs.append(np.array(img_data).reshape(
                        self.img_rows, self.img_cols, self.channels))
                else:
                    print('x', end='', flush=True)
            print()
        return np.array(imgs)

    def transform_image(self, img):
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
            rand_x, rand_y = random.randint(0, int(w/3)), random.randint(0, int(h/3))
            # rand side length greater than 150
            rand_side = random.randint(150, min((w-rand_x), (h-rand_y)) - 1)
            rand_points = [(rand_x, rand_y), (rand_x+rand_side, rand_y),
                           (rand_x, rand_y+rand_side), (rand_x+rand_side, rand_y+rand_side)]
            # checks if all corners are part of picture
            for x, y in rand_points:
                if img.getpixel((x, y)) == (0, 0, 0):
                    rand_points.remove((x, y))
            c += 1
        # Final Resize
        ##################
        box = (rand_x, rand_y, rand_x+rand_side, rand_y+rand_side)
        img = img.resize(
            (self.img_rows, self.img_cols),
            Image.LANCZOS,
            box)
        # img.show()
        
        return img

if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=20000, batch_size=32, save_interval=100)
