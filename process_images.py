import numpy as np
import cv2
import os
from random import shuffle
# for catching errors
import traceback
import logging

dataset_path = r"C:\Users\jadwi\training_fish"
img_size = 200
file_name = "dataset_fish.npy"


def label_image(img):
    img_name = img.split(".")[0]
    if 'gf' in img_name:
        return np.array([1, 0])
    elif 'perch' in img_name:
        return np.array([0, 1])


def prepare_dataset():
    """Prepare dataset (numpy array) with resized images and labels."""
    dataset = []
    for img in os.listdir(dataset_path):
        label = label_image(img)
        path = os.path.join(dataset_path, img)
        print(path)

        try:
            # load image from the path
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            # resize images
            img = cv2.resize(img, (img_size, img_size))

            # append img and its label to dataset
            dataset.append([np.array(img), label])

        except Exception as e:
            logging.error(traceback.format_exc())

    shuffle(dataset)
    return dataset


def save_dataset(dataset, filename):
    return np.save(filename, dataset)


def load_dataset(filename, pickle_allowed=True):
    return np.load(filename, allow_pickle=pickle_allowed)


# dataset_fish = prepare_dataset()
# save_dataset(dataset_fish, file_name)
# my_fish = load_dataset(file_name)
