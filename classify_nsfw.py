#!/usr/bin/env python
import os
import pathlib

import fire
import tensorflow as tf
import tqdm

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
        return predictions[0][1]


def pred_folder(folder_path, keep_only_highest_scored_image=False):
    images = [str(f.resolve()) for f in pathlib.Path.iterdir(pathlib.Path(folder_path))]
    images = [f for f in images if os.path.isfile(f) and f.lower().endswith(('.jpg', '.jpeg'))]

    result = []
    for img in tqdm.tqdm(images, desc=f'Predicting {len(images)} images'):
        scores = prediction(img)
        result.append((img, scores))

    if keep_only_highest_scored_image:
        survivor = max(result, key=lambda x: x[1])
        # delete others
        for img, score in result:
            if img != survivor[0]:
                os.remove(img)

        result = [survivor]

    return result


if __name__ == "__main__":
    fire.Fire()
