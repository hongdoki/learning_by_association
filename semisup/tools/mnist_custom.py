from __future__ import division
from __future__ import print_function

import numpy as np
import data_dirs
import os

DATADIR = data_dirs.mnist_custom


NUM_LABELS = 10
IMAGE_SHAPE = [32, 32, 3]


def get_data(name):
  """Utility for convenient data loading."""

  if name == 'train' or name == 'unlabeled':
    return np.load(os.path.join(DATADIR, 'target_mnist_X_train.npy')), \
           np.argmax(np.load(os.path.join(DATADIR, 'target_mnist_y_train.npy')), axis=1)
  elif name == 'validation':
    return np.load(os.path.join(DATADIR, 'target_mnist_X_val.npy')), \
           np.argmax(np.load(os.path.join(DATADIR, 'target_mnist_y_val.npy')), axis=1)
  elif name == 'test':
    return np.load(os.path.join(DATADIR, 'target_mnist_X_test.npy')), \
           np.argmax(np.load(os.path.join(DATADIR, 'target_mnist_y_test.npy')), axis=1)

