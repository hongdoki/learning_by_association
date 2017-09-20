import os
import tensorflow as tf
slim = tf.contrib.slim
LABELS_FILENAME = 'labels.txt'


def has_labels(dataset_dir, filename=LABELS_FILENAME):
    """Specifies whether or not the dataset directory contains a label map file.

    Args:
      dataset_dir: The directory in which the labels file is found.
      filename: The filename where the class names are written.

    Returns:
      `True` if the labels file exists and `False` otherwise.
    """
    return tf.gfile.Exists(os.path.join(dataset_dir, filename))


def read_label_file(dataset_dir, filename=LABELS_FILENAME):
    """Reads the labels file and returns a mapping from ID to class name.

    Args:
      dataset_dir: The directory in which the labels file is found.
      filename: The filename where the class names are written.

    Returns:
      A map from a label (integer) to class name.
    """
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'rb') as f:
        lines = f.read().decode()
    lines = lines.split('\n')
    lines = filter(None, lines)

    labels_to_class_names = {}
    for line in lines:
        index = line.index(':')
        labels_to_class_names[int(line[:index])] = line[index+1:]
    return labels_to_class_names
    batch_labels.append(labels_each)


def get_slim_dataset(dataset_tools, split_name, cls=None, reader=None):
  """load dataset with tf records into slim.dataset.Dataset
  Args:
    split_name: A train/test split name.
    dataset_dir: The base directory of the dataset sources.
    cls: integer for class

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/val split.
  """
  dataset_dir = dataset_tools.DATADIR
  file_pattern = dataset_tools.FILE_PATTERN
  image_shape = dataset_tools.IMAGE_SHAPE

  # use unlabeled dataset as training dataset
  if split_name == 'unlabeled':
      split_name = 'train'

  if cls is not None:
      file_name_postfix = split_name + ('_cls%d' % cls)
  else:
      file_name_postfix = split_name

  file_pattern = os.path.join(dataset_dir, file_pattern % file_name_postfix)

  if reader is None:
      reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded':
          tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format':
          tf.FixedLenFeature((), tf.string, default_value='raw'),
      'image/class/label':
          tf.FixedLenFeature(
              [1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(shape=image_shape, channels=image_shape[-1]),
      'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[]),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if has_labels(dataset_dir):
      labels_to_names = read_label_file(dataset_dir)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=dataset_tools.SPLITS_TO_SIZES[split_name],
      num_classes=dataset_tools.NUM_LABELS,
      items_to_descriptions=dataset_tools.ITEMS_TO_DESCRIPTIONS,
      labels_to_names=labels_to_names
  )


def dataset_to_batch(slim_dataset, batch_size, num_readers=4, num_preprocessing_threads=4):
    provider = slim.dataset_data_provider.DatasetDataProvider(
        slim_dataset,
        num_readers=num_readers,
        common_queue_capacity=20 * batch_size,
        common_queue_min=10 * batch_size)
    [images, labels] = provider.get(['image', 'label'])

    # Load the data.
    images, labels = tf.train.batch(
        [images, labels],
        batch_size=batch_size,
        num_threads=num_preprocessing_threads,
        capacity=5 * batch_size)
    return images, labels


def generate_class_balanced_batch_with_slim_dataset(dataset_tools, batch_size_per_class):
  num_classes = dataset_tools.NUM_LABELS
  batch_images, batch_labels = [], []

  if type(batch_size_per_class) == str:
      batch_size_per_class_list = [int(s) for s in batch_size_per_class.split(',')]
      assert len(batch_size_per_class_list) == num_classes, \
          'number of classes not matched: %d classes in batch size specified but actually %d classes' % \
          (len(batch_size_per_class_list), num_classes)
  else:
      batch_size_per_class_list = [batch_size_per_class for i in range(num_classes)]

  for i in xrange(num_classes):
      sup_dataset = get_slim_dataset(dataset_tools, 'train', cls=i)
      images_each, labels_each = dataset_to_batch(sup_dataset, batch_size_per_class_list[i])
      batch_images.append(images_each)
      batch_labels.append(labels_each)

  return tf.concat(batch_images, 0), tf.concat(batch_labels, 0)


