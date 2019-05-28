from __future__ import division, print_function

import argparse
import datetime

import matplotlib.pyplot as plt
import numpy as np
import os

from image_god import ImageGod
from keras.activations import relu
from keras.initializers import RandomNormal
from keras.layers import Activation, Dense, Dropout, Flatten, Input, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras_contrib.layers.normalization.instancenormalization import \
    InstanceNormalization
from scrapers.pexel_downloader import PexelDownloader
from scrapers.wikiart_scraper import WikiartScraper


class CycleGAN():
    def __init__(self):
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        self.ngf = 64
        self.ndf = 32
        
        # Loss weights
        self.lambda_cycle = 10.0                    # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle    # Identity loss


        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.dis_A = self.build_discriminator()
        self.dis_B = self.build_discriminator()
        self.dis_A.compile(loss='mse',
                           optimizer=optimizer,
                           metrics=['accuracy'])
        self.dis_B.compile(loss='mse',
                           optimizer=optimizer,
                           metrics=['accuracy'])

        # -------------------------
        # Construct Computational
        #   Graph of Generators
        # -------------------------

        # Build the generators
        self.gen_AtoB = self.build_generator()
        self.gen_BtoA = self.build_generator()

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.gen_AtoB(img_A)
        fake_A = self.gen_BtoA(img_B)
        # Translate images back to original domain
        reconstr_A = self.gen_BtoA(fake_B)
        reconstr_B = self.gen_AtoB(fake_A)
        # Identity mapping of images
        img_A_id = self.gen_BtoA(img_A)
        img_B_id = self.gen_AtoB(img_B)

        # For the combined model we will only train the generators
        self.dis_A.trainable = False
        self.dis_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.dis_A(fake_A)
        valid_B = self.dis_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[valid_A, valid_B,
                                       reconstr_A, reconstr_B,
                                       img_A_id, img_B_id])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                              loss_weights=[1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id],
                              optimizer=optimizer)

    def build_generator(self):

        k = 5
        s = 2

        # for convolution kernel
        conv_init = RandomNormal(0, 0.02)
        # for batch normalization
        gamma_init = RandomNormal(1., 0.02)

        model = Sequential()

        #########################
        # ENCODING LAYERS
        #########################

        model.add(
            Conv2D(
                filters=self.ngf,
                kernel_size=k,
                strides=s,
                padding='same',
                use_bias=False,
                kernel_initializer=conv_init,
                input_shape=self.img_shape,
            )
        )
        model.add(InstanceNormalization(
            gamma_initializer=gamma_init,
        )
        )
        model.add(Activation('relu'))

        model.add(
            Conv2D(
                filters=self.ngf*2,
                kernel_size=k,
                strides=s,
                padding='same',
                use_bias=False,
                kernel_initializer=conv_init,
            )
        )
        model.add(InstanceNormalization(
            gamma_initializer=gamma_init,
        )
        )
        model.add(Activation('relu'))

        model.add(
            Conv2D(
                filters=self.ngf*4,
                kernel_size=k,
                strides=s,
                padding='same',
                use_bias=False,
                kernel_initializer=conv_init,
            )
        )
        model.add(InstanceNormalization(

            gamma_initializer=gamma_init,
        )
        )
        model.add(Activation('relu'))

        
        #########################
        # TRANSITION LAYERS
        #########################

        model.add(
            Conv2D(
                filters=self.ngf*4,
                kernel_size=k,
                strides=1,
                padding='same',
                use_bias=False,
                kernel_initializer=conv_init,
            )
        )
        model.add(InstanceNormalization(
            gamma_initializer=gamma_init,
        ))
        model.add(Activation('relu'))

        model.add(
            Conv2D(
                filters=self.ngf*4,
                kernel_size=k,
                strides=1,
                padding='same',
                use_bias=False,
                kernel_initializer=conv_init,
            )
        )
        model.add(InstanceNormalization(
            gamma_initializer=gamma_init,
        ))
        model.add(Activation('relu'))

        model.add(
            Conv2D(
                filters=self.ngf*4,
                kernel_size=k,
                strides=1,
                padding='same',
                use_bias=False,
                kernel_initializer=conv_init,
            )
        )
        model.add(InstanceNormalization(
            gamma_initializer=gamma_init,
        ))
        model.add(Activation('relu'))

        model.add(
            Conv2D(
                filters=self.ngf*4,
                kernel_size=k,
                strides=1,
                padding='same',
                use_bias=False,
                kernel_initializer=conv_init,
            )
        )
        model.add(InstanceNormalization(
            gamma_initializer=gamma_init,
        ))
        model.add(Activation('relu'))

        #########################
        # DECODING LAYERS
        #########################

        model.add(Conv2DTranspose(
            filters=self.ngf*2,
            kernel_size=k,
            strides=s,
            padding='same',
            use_bias=False,
            kernel_initializer=conv_init,
        )
        )
        model.add(
            InstanceNormalization(
                gamma_initializer=gamma_init,
            )
        )
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(
            filters=self.ngf,
            kernel_size=k,
            strides=s,
            padding='same',
            use_bias=False,
            kernel_initializer=conv_init,
        )
        )
        model.add(
            InstanceNormalization(
                gamma_initializer=gamma_init,
            )
        )
        model.add(Activation('relu'))

        # Output Layer
        model.add(
            Conv2DTranspose(
                filters=self.channels,
                kernel_size=k,
                strides=s,
                padding='same',
                use_bias=False,
                kernel_initializer=conv_init,

            )
        )
        model.add(Activation('tanh'))

        model.summary()

        A = Input(shape=self.img_shape)
        B = model(A)

        return Model(A, B)

    def build_discriminator(self):

        k = 4
        s = 2

        model = Sequential()

        # First Layer
        model.add(
            Conv2D(
                filters=self.ndf,
                kernel_size=4,
                strides=s,
                padding='same',
                use_bias=False,
                input_shape=self.img_shape,
            )
        )
        model.add(InstanceNormalization())
        model.add(LeakyReLU(alpha=0.2))

        # Layer 2
        model.add(
            Conv2D(
                filters=self.ndf*2,
                kernel_size=k,
                strides=s,
                padding='same',
                use_bias=False,
            )
        )
        model.add(InstanceNormalization())
        model.add(LeakyReLU(alpha=0.2))

        # Layer 3
        model.add(
            Conv2D(
                filters=self.ndf*4,
                kernel_size=k,
                strides=s,
                padding='same',
                use_bias=False,
            )
        )
        model.add(InstanceNormalization())
        model.add(LeakyReLU(alpha=0.2))

        # Layer 4
        model.add(
            Conv2D(
                filters=self.ndf*8,
                kernel_size=k,
                strides=s,
                padding='same',
                use_bias=False,
            )
        )
        model.add(InstanceNormalization())
        model.add(LeakyReLU(alpha=0.2))

        # Final Layer
        model.add(
            Conv2D(
                filters=1,
                kernel_size=k,
                strides=1,
                padding='same',
                use_bias=False,
                activation='sigmoid',
            )
        )

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, x_training_dir, y_training_dir, epochs,
              batch_size=1, save_interval=100,
              pexels_query=None, wikiart_scrape_url=None):

        # ---------------------
        #  Preprocessing
        # ---------------------

        god = ImageGod()

        # Scrape photos from pexel to as X Train if given query
        if pexels_query:
            pd = PexelDownloader()
            pd.download_pexels_query(pexels_query, x_training_dir)

            god.crop_images(self.img_shape, x_training_dir)
            x_training_dir = x_training_dir + '_crops'

        # Scrape painter's collection from wikiart as Y Train if given painter
        if wikiart_scrape_url:
            ws = WikiartScraper()
            ws.scrape_art(
                wikiart_scrape_url,
                y_training_dir,
            )

            god.transform_images(self.img_shape, y_training_dir, epochs=50)
            y_training_dir = y_training_dir + '_transformations'

        # Load from x and y training_dir
        X_train = god.load_images(
            self.img_shape,
            x_training_dir,
        )

        Y_train = god.load_images(
            self.img_shape,
            y_training_dir,
        )

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size, 8, 8, 1))
        fake = np.zeros((batch_size, 8, 8, 1))

        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs_A = X_train[idx]

            idx = np.random.randint(0, Y_train.shape[0], batch_size)
            imgs_B = Y_train[idx]

            # ----------------------
            #  Train Discriminators
            # ----------------------

            # Translate images to opposite domain
            fake_B = self.gen_AtoB.predict(imgs_A)
            fake_A = self.gen_BtoA.predict(imgs_B)

            # Train the discriminators (original images = real / translated = Fake)
            dA_loss_real = self.dis_A.train_on_batch(imgs_A, valid)
            dA_loss_fake = self.dis_A.train_on_batch(fake_A, fake)
            dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

            dB_loss_real = self.dis_B.train_on_batch(imgs_B, valid)
            dB_loss_fake = self.dis_B.train_on_batch(fake_B, fake)
            dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

            # Total disciminator loss
            d_loss = 0.5 * np.add(dA_loss, dB_loss)

            # ------------------
            #  Train Generators
            # ------------------

            # Train the generators
            g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                  [valid, valid,
                                                   imgs_A, imgs_B,
                                                   imgs_A, imgs_B])

            elapsed_time = datetime.datetime.now() - start_time

            # Plot the progress
            print("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s "
                  % (epoch, epochs,
                     d_loss[0], 100 *
                     d_loss[1],
                     g_loss[0],
                     np.mean(
                         g_loss[1:3]),
                     np.mean(
                         g_loss[3:5]),
                     np.mean(
                         g_loss[5:6]),
                     elapsed_time))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs_A = X_train[idx]

                idx = np.random.randint(0, Y_train.shape[0], batch_size)
                imgs_B = Y_train[idx]
                self.save_imgs(imgs_A, imgs_B, epoch)
        
        self.gen_AtoB.save('cycle_gan_A_to_B_generator.h5')
        self.gen_BtoA.save('cycle_gan_B_to_A_generator.h5')
    
    def save_imgs(self, imgs_A, imgs_B, epoch):
        os.makedirs('samples/cyclegan', exist_ok=True)
        r, c = 2, 3

        # Demo (for GIF)
        #imgs_A = self.data_loader.load_img('datasets/apple2orange/testA/n07740461_1541.jpg')
        #imgs_B = self.data_loader.load_img('datasets/apple2orange/testB/n07749192_4241.jpg')

        # Translate images to the other domain
        fake_B = self.gen_AtoB.predict(imgs_A)
        fake_A = self.gen_BtoA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.gen_BtoA.predict(fake_B)
        reconstr_B = self.gen_AtoB.predict(fake_A)

        gen_imgs = np.concatenate(
            [imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("samples/cyclegan/art_%d.png" %
                    (epoch))
        plt.close()


def parse_command_line_args():
    parser = argparse.ArgumentParser(description='AI Generated Art Bitch')
    parser.add_argument('epochs', type=int,
                        help='number of epochs')
    parser.add_argument('x_training_dir', type=str,
                        help='filepath of images set')
    parser.add_argument('y_training_dir', type=str,
                        help='filepath of style set')
    parser.add_argument('-b', '--batchsize',
                        default=1, type=int, help='size of batches per epoch')
    parser.add_argument('-s', '--saveinterval',
                        type=int, default=100, help='interval to save sample images')
    parser.add_argument('-w', '--wikiart', type=str, default=None,
                        help='painter on wikiart to dowload from')
    parser.add_argument('-p', '--pexels', type=str, default=None,
                        help='pexels query to dowload from')

    return vars(parser.parse_args())
    # print(args)


if __name__ == '__main__':

    args = parse_command_line_args()
    gan = CycleGAN()
    gan.train(
        x_training_dir=args['x_training_dir'],
        y_training_dir=args['y_training_dir'],
        epochs=args['epochs'],
        batch_size=args['batchsize'],
        save_interval=args['saveinterval'],
        wikiart_scrape_url=args['wikiart'],
        pexels_query=args['pexels']
    )
