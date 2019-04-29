import argparse

import numpy as np
from keras.models import load_model
from PIL import Image


def generate(n, model_filepath):
    noise_shape = 100
    noise = np.random.normal(0, 1, (n, noise_shape))
    model = load_model(model_filepath)
    gen_art_data = model.predict(noise)

    for i, art_data in enumerate(gen_art_data):
        art_data = ((art_data + 127.5) * 127.5).astype(np.int16)
        Image.fromarray(art_data, mode='RGB').save('./gen_art/art_%d.jpg' % i)


def parse_command_line_args():
    parser = argparse.ArgumentParser(description='AI Generated Art Bitch')
    parser.add_argument('N', type=int,
                        help='number of images to generate and save')
    parser.add_argument('path', type=str,
                        help='filepath to pretrained h5 Keras model')
    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_command_line_args()
    generate(args['N'], args['path'])
