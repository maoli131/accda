import tensorflow as tf

import numpy as np

import imageio
from PIL import Image


def read_image(img_path):
    # Reads an image in .npy format, and normalizes it via tf.keras.applications.imagenet_utils.preprocess_input

    assert '.npy' in img_path

    img = np.load(img_path)

    # Tensorflow normalization
    # Each image is divided by 127.5, and decreased by 1
    # Source: https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/applications/imagenet_utils.py#L103-L119
    img = tf.keras.applications.imagenet_utils.preprocess_input(img, mode='tf')

    return img

def read_label(label_path):
    # Reads an array of labels in .npy format

    assert '.npy' in label_path

    label = np.load(label_path)

    return label


def read_and_reshape(img_path, labels_path, img_shape=(1024,512), dataset='CITYSCAPES'):
    # Reads an image/label pair and reshapes it to the desired shape
    assert dataset in ['CITYSCAPES', 'SYNTHIA', 'GTA5']
    try:
        # Read Images
        img = np.asarray(imageio.imread(img_path, format='PNG'))
        labels = np.asarray(imageio.imread(labels_path, format='PNG'))
        
        if dataset == 'SYNTHIA':
            # Synthia labels come in a 2-channel format
            labels = labels[:,:,0]

        if dataset == 'GTA5':
            labels = np.asarray(Image.open(labels_path))

        # Resize Images
        img = Image.fromarray(img).resize((img_shape[0], img_shape[1]), Image.LANCZOS)
        img = np.asarray(img)

        labels = Image.fromarray(labels).resize((img_shape[0], img_shape[1]), Image.NEAREST)
        labels = np.asarray(labels)
        
        return img, labels
    except Exception as e:
        print(e, img_path, labels_path)
        # print("Failed on", labels_path)   
        return None, None