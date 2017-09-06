from __future__ import division
from __future__ import print_function

import numpy as np
import data_dirs
import os

DATADIR = data_dirs.dagm


NUM_LABELS = 2
IMAGE_SHAPE = [256, 256, 1]

SPLITS_TO_SIZES = {'train': 4559, 'val': 506, 'test': 5095,
                   'train_cls0': 4412, 'train_cls1': 147,
                   'val_cls0': 488, 'val_cls1': 18,
                   'test_cls0': 4953, 'test_cls1':142}
FILE_PATTERN = 'target_d4_%s.tfrecords'
ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [256 x 256 x 1] image.',
    'label': 'A single integer between 0 and 1',
}

def get_data(name):
  """Utility for convenient data loading."""

  if name == 'train' or name == 'unlabeled':
    return np.expand_dims(np.load(os.path.join(DATADIR, 'target_d4_train_X.npy')), axis=-1), \
           np.argmax(np.load(os.path.join(DATADIR, 'target_d4_train_y.npy')), axis=1)
  elif name == 'validation':
    return np.expand_dims(np.load(os.path.join(DATADIR, 'target_d4_val_X.npy')), axis=-1), \
           np.argmax(np.load(os.path.join(DATADIR, 'target_d4_val_y.npy')), axis=1)
  elif name == 'test':
    return np.expand_dims(np.load(os.path.join(DATADIR, 'target_d4_test_X.npy')), axis=-1), \
           np.argmax(np.load(os.path.join(DATADIR, 'target_d4_test_y.npy')), axis=1)

