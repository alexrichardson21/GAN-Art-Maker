from __future__ import division, print_function

import argparse

import matplotlib.pyplot as plt
import numpy as np
from keras.activations import relu
from keras.layers import (Activation, BatchNormalization, Dense, Dropout,
                          Flatten, Input, Reshape)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model, Sequential
from keras.optimizers import Adam

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

        k = 5
        s = 2

        model = Sequential()

        # First Layer
        model.add(Dense(4 * 4 * 1024, input_shape=(self.noise,)))
        model.add(Reshape(target_shape=(4, 4, 1024)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        # Layer 2
        model.add(Conv2DTranspose(
            filters=512,
            kernel_size=k,
            strides=s,
            padding='same',
            use_bias=False,
        ))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        # Layer 3
        model.add(Conv2DTranspose(
            filters=256,
            kernel_size=k,
            strides=s,
            padding='same',
            use_bias=False,
        ))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        # Layer 4
        model.add(Conv2DTranspose(
            filters=128,
            kernel_size=k,
            strides=s,
            padding='same',
            use_bias=False,
        ))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        # Layer 5
        model.add(Conv2DTranspose(
            filters=64,
            kernel_size=k,
            strides=s,
            padding='same',
            use_bias=False,
        ))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        # Output Layer
        model.add(
            Conv2DTranspose(
                filters=self.channels,
                kernel_size=k,
                strides=s,
                padding='same',
                use_bias=False,
            )
        )
        model.add(Activation('tanh'))

        model.summary()

        noise = Input(shape=(self.noise,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        k = 4
        s = 2

        model = Sequential()

        # First Layer
        model.add(
            Conv2D(
                filters=64//2,
                kernel_size=k,
                strides=s,
                padding='same',
                use_bias=False,
                input_shape=self.img_shape,
            )
        )
        model.add(LeakyReLU(alpha=0.2))

        # Layer 2
        model.add(
            Conv2D(
                filters=128//2,
                kernel_size=k,
                strides=s,
                padding='same',
                use_bias=False,
            )
        )
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        # Layer 3
        model.add(
            Conv2D(
                filters=256//2,
                kernel_size=k,
                strides=s,
                padding='same',
                use_bias=False,
            )
        )
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        # Layer 4
        model.add(
            Conv2D(
                filters=512//2,
                kernel_size=k,
                strides=s,
                padding='same',
                use_bias=False,
            )
        )
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        # Layer 5
        model.add(
            Conv2D(
                filters=1024//2,
                kernel_size=k,
                strides=s,
                padding='same',
                use_bias=False,
            )
        )
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        # Final Layer
        model.add(Flatten())
        model.add(Dropout(.3))
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, training_dir, epochs, batch_size=32, save_interval=100, transform=0, wikiart_scrape_url=None):

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
        if transform:
            X_train = im.load_and_transform_images(
                self.img_shape,
                training_dir,
                epochs=40,
                save_rate=1,
            )
        else:
            X_train = im.load_images(
                self.img_shape,
                training_dir,
            )

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


def parse_command_line_args():
    parser = argparse.ArgumentParser(description='AI Generated Art Bitch')
    parser.add_argument('epochs', type=int,
                        help='number of epochs')
    parser.add_argument('training_dir', type=str,
                        help='filepath of training set (if wikiart url is given then filepath becomes the save dir)')
    parser.add_argument('-b', '--batchsize',
                        default=32, type=int, help='size of batches per epoch')
    parser.add_argument('-s', '--saveinterval',
                        type=int, default=100, help='interval to save sample images')
    parser.add_argument('-w', '--wikiart', type=str, default=None,
                        help='url of wikiart profile to dowload from')
    parser.add_argument('-t', '--transform', type=int, default=0,
                        help='number of transformations applied to each picture (default: no transformation)')
    return vars(parser.parse_args())
    # print(args)


if __name__ == '__main__':

    wikiart_profile = 'https://www.wikiart.org/en/profile/5c9ba655edc2c9b87424edfe/albums/favourites'

    args = parse_command_line_args()
    gan = GAN()
    gan.train(
        training_dir=args['training_dir'],
        epochs=args['epochs'],
        transform=args['transform'],
        batch_size=args['batchsize'],
        save_interval=args['saveinterval'],
        wikiart_scrape_url=args['wikiart'],
    )
