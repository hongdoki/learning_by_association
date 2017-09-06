from __future__ import division
from __future__ import print_function

import numpy as np
import data_dirs
import os

DATADIR = data_dirs.svhn_custom


NUM_LABELS = 10
IMAGE_SHAPE = [32, 32, 3]


def get_data(name):
  """Utility for convenient data loading."""

  if name == 'train' or name == 'unlabeled':
    return np.load(os.path.join(DATADIR, 'source_svhn_train_X.npy')), \
           np.argmax(np.load(os.path.join(DATADIR, 'source_svhn_train_y.npy')), axis=1)
  # custom svhn has only training/validation set
  elif name == 'test':
    return np.load(os.path.join(DATADIR, 'source_svhn_val_X.npy')), \
           np.argmax(np.load(os.path.join(DATADIR, 'source_svhn_val_y.npy')), axis=1)
