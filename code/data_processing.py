import gzip
import os
import pdb

import numpy as np
from six.moves import urllib
from sklearn.utils import shuffle
import tensorflow as tf
from PIL import Image

## Initialize seeds
np.random.seed(0)
tf.set_random_seed(0)

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = '../../data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
BATCH_SIZE = 2

######################################## Data processing ########################################
def maybe_download(filename):
    """Download the data from Yann's website, unless it's already here."""
    if not tf.gfile.Exists(WORK_DIRECTORY):
        tf.gfile.MakeDirs(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath

def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [0.0, 1.0].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        return data

def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

def get_data():
    # download the data id needed
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    #train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    #test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')
    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, 60000)
    #train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)
    #test_labels = extract_labels(test_labels_filename, 10000)
    # Merge train and test_data
    data = np.concatenate((train_data,test_data))
    #labels = np.concatenate(train_labels,test_labels)
    return data

def get_batches(images, batch_size=BATCH_SIZE):
    batches = []
    #X = shuffle(images)
    X = images
    for i in range(int(X.shape[0]/batch_size)):
        X_batch = X[i * batch_size: (i + 1) * batch_size]
        """
        if i<=int(X.shape[0]/batch_size):
            X_batch = X[i * batch_size: (i + 1) * batch_size]
        else:
            X_batch = X[-batch_size:]
        """
        batches.append(X_batch)
    return batches

def binarize(images, threshold=0.1):
    return (threshold < images).astype("float32")
