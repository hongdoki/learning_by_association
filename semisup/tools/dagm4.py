from __future__ import division
from __future__ import print_function

import numpy as np
import data_dirs
import os

DATADIR = data_dirs.dagm


NUM_LABELS = 2
IMAGE_SHAPE = [256, 256, 1]


def get_data(name):
  """Utility for convenient data loading."""

  if name == 'train' or name == 'unlabeled':
    return np.expand_dims(np.load(os.path.join(DATADIR, 'target_d4_X_train.npy')), axis=-1), \
           np.argmax(np.load(os.path.join(DATADIR, 'target_d4_y_train.npy')), axis=1)
  # custom svhn has only training/validation set
  elif name == 'test':
    return np.expand_dims(np.load(os.path.join(DATADIR, 'target_d4_X_test.npy')), axis=-1), \
           np.argmax(np.load(os.path.join(DATADIR, 'target_d4_y_test.npy')), axis=1)

