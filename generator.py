import argparse

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from PIL import Image


def generate(n, model_filepath):
    noise = np.random.normal(0, 1, (n, 100))
    model = load_model(model_filepath)

    gen_art_data = model.predict(noise)
    # Rescale images 0 - 1
    gen_art_data = 0.5 * gen_art_data + 0.5

    # Saves n generated images
    for i, art_data in enumerate(gen_art_data):
        fig, axs = plt.subplots()
        axs.imshow(art_data)
        axs.axis('off')
        fig.savefig('gen_art/art_%d.jpg' % i, 
                    bbox_inches='tight', pad_inches=0)
        plt.close()


def parse_command_line_args():
    parser = argparse.ArgumentParser(description='AI Generated Art Bitch')
    parser.add_argument('N', type=int,
                        help='number of images to generate and save')
    parser.add_argument('path', type=str,
                        help='filepath to pretrained h5 generator')
    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_command_line_args()
    # generate(10, 'art_gan_generator.h5')
    generate(args['N'], args['path'])
