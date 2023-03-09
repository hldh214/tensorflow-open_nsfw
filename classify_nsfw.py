#!/usr/bin/env python
import os
import pathlib
import sys

import tensorflow as tf

from model import OpenNsfwModel, InputType
from image_utils import create_yahoo_image_loader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMAGE_LOADER_TENSORFLOW = "tensorflow"
IMAGE_LOADER_YAHOO = "yahoo"


def prediction(img_path):
    model = OpenNsfwModel()

    with tf.Session(graph=tf.Graph()) as sess:
        input_type = InputType[InputType.TENSOR.name.upper()]
        model.build(weights_path=pathlib.Path(__file__).parent / 'data/open_nsfw-weights.npy', input_type=input_type)

        fn_load_image = create_yahoo_image_loader()

        sess.run(tf.global_variables_initializer())

        image = fn_load_image(img_path)

        predictions = sess.run(model.predictions, feed_dict={model.input: image})

        # [[SFW_score, NSFW_score]]
        # [[0.00922205, 0.99077797]]
        return predictions


if __name__ == "__main__":
    # todo: add mode for leave only highest scored image
    if len(sys.argv) == 2:
        path = sys.argv[1]

        if os.path.isfile(path):
            scores = prediction(path)
            print(scores[0][1])  # NSFW_score
            exit(0)

        if os.path.isdir(path):
            # list images in the folder
            images = [f for f in os.listdir(path) if os.path.isfile(f) and f.lower().endswith(('.jpg', '.jpeg'))]
            for img in images:
                scores = prediction(img)
                print(img, f'SFW_score: {scores[0][0]:.2%}, NSFW_score: {scores[0][1]:.2%}')
            exit(0)

    # list images in the current directory
    images = [f for f in os.listdir('.') if os.path.isfile(f) and f.lower().endswith(('.jpg', '.jpeg'))]
    for img in images:
        scores = prediction(img)
        print(img, f'SFW_score: {scores[0][0]:.2%}, NSFW_score: {scores[0][1]:.2%}')
