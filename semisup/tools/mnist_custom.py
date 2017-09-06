from __future__ import division
from __future__ import print_function

import numpy as np
import data_dirs
import os

DATADIR = data_dirs.mnist_custom


NUM_LABELS = 10
IMAGE_SHAPE = [32, 32, 3]

SPLITS_TO_SIZES = {'train': 54000, 'val': 6000, 'test': 10000}
FILE_PATTERN = 'target_mnist_%s.tfrecords'
ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [32 x 32 x 3] RGB image. (actually gray scale)',
    'label': 'A single integer between 0 and 9',
}

def get_data(name):
  """Utility for convenient data loading."""

  if name == 'train' or name == 'unlabeled':
    return np.load(os.path.join(DATADIR, 'target_mnist_train_X.npy')), \
           np.argmax(np.load(os.path.join(DATADIR, 'target_mnist_train_y.npy')), axis=1)
  elif name == 'validation':
    return np.load(os.path.join(DATADIR, 'target_mnist_val_X.npy')), \
           np.argmax(np.load(os.path.join(DATADIR, 'target_mnist_val_y.npy')), axis=1)
  elif name == 'test':
    return np.load(os.path.join(DATADIR, 'target_mnist_test_X.npy')), \
           np.argmax(np.load(os.path.join(DATADIR, 'target_mnist_test_y.npy')), axis=1)

