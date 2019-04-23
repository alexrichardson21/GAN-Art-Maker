from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Activation, Reshape, Flatten, Dropout, Cropping2D
from keras.layers import MaxPooling2D, GlobalAveragePooling2D,BatchNormalization, Activation, ZeroPadding2D, Lambda
from keras.activations import relu
from keras.layers.advanced_activations import LeakyReLU

from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import cv2
import skimage.transform
from PIL import Image

import numpy as np
import pandas as pd

from image_normalization import ImageNormalizer
from wikiart_scraper import WikiartScraper
class GAN():
    def __init__(self):
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.noise = 100

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
        z = Input((self.noise,))
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

        k = 3

        model = Sequential()

        # First Layer
        model.add(Reshape((1,1,self.noise), input_shape=(self.noise,)))
        model.add(
            Conv2DTranspose(
                filters=2560, 
                kernel_size=k,
                use_bias=False,
            )
        )
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        # Layer 2
        model.add(
            Conv2DTranspose(
                filters=1280, 
                kernel_size=k,
                strides=2, 
                use_bias=False,
            )
        )
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        # Layer 3
        model.add(
            Conv2DTranspose(
                filters=640, 
                kernel_size=k,
                strides=2, 
                use_bias=False,
            )
        )
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        # Layer 4
        model.add(
            Conv2DTranspose(
                filters=320, 
                kernel_size=k,
                strides=2, 
                use_bias=False,
            )
        )
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        # Layer 5
        model.add(
            Conv2DTranspose(
                filters=160, 
                kernel_size=k,
                strides=2, 
                # output_padding=1, 
                use_bias=False,
            )
        )
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        # Output Layer
        model.add(
            Conv2DTranspose(
                filters=self.channels, 
                kernel_size=k+1,
                strides=2, 
                use_bias=False,
            )
        )
        # model.add(Cropping2D(((1, 0), (1, 0))))
        model.add(Activation('tanh'))

        model.summary()

        noise = Input(shape=(self.noise,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)
        k = 4

        model = Sequential()
        
        # First Layer
        model.add(
            Conv2D(
                filters=40, 
                kernel_size=k,
                strides=2, 
                input_shape=img_shape
            )
        )
        model.add(LeakyReLU(alpha=0.2))
        model.add(ZeroPadding2D(padding=1))

        # Layer 2
        model.add(
            Conv2D(
                filters=80, 
                kernel_size=k, 
                strides=2
            )
        )
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(ZeroPadding2D(padding=1))

        # Layer 3
        model.add(
            Conv2D(
                filters=160, 
                kernel_size=k, 
                strides=2
            )
        )
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(ZeroPadding2D(padding=1))

        # Layer 4
        model.add(
            Conv2D(
                filters=320, 
                kernel_size=k, 
                strides=2
            )
        )
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(ZeroPadding2D(padding=1))
        # Layer 5
        model.add(
            Conv2D(
                filters=640, 
                kernel_size=k, 
                strides=2
            )
        )
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        # Final Layer
        model.add(Dropout(0.3))
        model.add(
            Conv2D(
                filters=1,
                kernel_size=k-1,
                activation='sigmoid',
            )
        )
        model.add(Reshape((1,)))
        
        # model.add(Dense(1, activation='sigmoid'))
        
        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50, training_dir='./select_train' , wikiart_scrape_url=None):
        
        # ---------------------
        #  Preprocessing
        # ---------------------

        # Scrape from wikiart profile to output directory if url given
        if wikiart_scrape_url:
            ws = WikiartScraper()
            ws.scrape_art(
                wikiart_scrape_url, 
                training_dir,
            )
        
        # Load from training_dir and normalize dataset
        im = ImageNormalizer()
        X_train = im.load_and_transform_images(
            self.img_shape, 
            training_dir, 
            epochs=100, 
            save_rate=1,
        )
        
        # Make pixel values -1 to 1
        # X_train = (X_train.astype(np.float16) - 127.5) / 127.5

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]
            
            noise = np.random.normal(0, 1, (half_batch, self.noise))

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

            noise = np.random.normal(0, 1, (batch_size, self.noise))

            # The generator wants the discriminator to label the generated samples as valid (ones)
            valid_y = np.ones((batch_size, 1))

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                  (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
        

        # Save all three models
        self.discriminator.save('art_gan_discriminator.h5')
        self.generator.save('art_gan_generator.h5')
        self.combined.save('art_gan_combined.h5')

    def save_imgs(self, epoch):
        r, c = 3, 3
        noise = np.random.normal(0, 1, (r * c, self.noise))
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

if __name__ == '__main__':
    
    wikiart_profile = 'https://www.wikiart.org/en/profile/5c9ba655edc2c9b87424edfe/albums/favourites'
    
    gan = GAN()
    gan.train(
        epochs=40000, 
        batch_size=32, 
        training_dir='./testing_paintings', 
        save_interval=100,
        # wikiart_scrape_url=wikiart_profile,
    ) 
        
