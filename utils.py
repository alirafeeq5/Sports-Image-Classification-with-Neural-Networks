import csv
import os

import cv2
import numpy as np
import tensorflow as tf


SUBMISSION_ENCODING = [
    'basketball',
    'football',
    'rowing',
    'swimming',
    'tennis',
    'yoga',
]


def one_hot_encoding(name, positions=SUBMISSION_ENCODING):
    return np.array(
        [(1 if name.lower() == pos else 0) for pos in positions]
    )


def load_images(path, new_size=(256, 256), include_labels=True):
    images = []
    labels = []
    file_names = []

    for image_file in os.listdir(path):
        image_path = str(path / image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, new_size)

        image_name = image_file.split('_')[0]
        label = one_hot_encoding(image_name)

        images.append(image)
        labels.append(label)
        file_names.append(image_file)

    if include_labels:
        result = (np.array(images), np.array(labels))
    else:
        result = np.array(images)

    return file_names, tf.data.Dataset.from_tensor_slices(result)


def write_predictions(path, file_names, predictions):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['image_name', 'label'])

        for fname, preds in zip(file_names, predictions):
            pred = preds.argmax()
            writer.writerow([fname, pred])

