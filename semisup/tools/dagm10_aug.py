from __future__ import division
from __future__ import print_function

import numpy as np
import data_dirs
import os

DATADIR = os.path.join(data_dirs.dagm, 'augmented')


NUM_LABELS = 2
IMAGE_SHAPE = [256, 256, 1]

SPLITS_TO_SIZES = {'train': 72288, 'val': 2007,
                   'train_cls0': 69288, 'train_cls1': 3000,
                   'val_cls0': 1937, 'val_cls1': 70}
FILE_PATTERN = 'source_d10_%s.tfrecords'
ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [256 x 256 x 1] image.',
    'label': 'A single integer between 0 and 1',
}


def get_data(name):
  """Utility for convenient data loading."""

  if name == 'train' or name == 'unlabeled':
    return np.expand_dims(np.load(os.path.join(DATADIR, 'source_d10_train_X.npy')), axis=-1), \
           np.argmax(np.load(os.path.join(DATADIR, 'source_d10_train_y.npy')), axis=1)
  # custom svhn has only training/validation set
  elif name == 'validation' or name == 'val':
    return np.expand_dims(np.load(os.path.join(DATADIR, 'source_d10_val_X.npy')), axis=-1), \
           np.argmax(np.load(os.path.join(DATADIR, 'source_d10_val_y.npy')), axis=1)

