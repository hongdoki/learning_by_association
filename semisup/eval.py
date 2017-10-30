#! /usr/bin/env python
"""
Copyright 2016 Google Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Association-based semi-supervised eval module.

This script defines the evaluation loop that works with the training loop
from train.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from functools import partial
from importlib import import_module

import semisup
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
import os
import tools.data_util as data_util

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'svhn', 'Which dataset to work on.')

flags.DEFINE_string('architecture', 'svhn_model', 'Which dataset to work on.')

flags.DEFINE_integer('eval_batch_size', 500, 'Batch size for eval loop.')

flags.DEFINE_integer('new_size', 0, 'If > 0, resize image to this width/height.'
                                    'Needs to match size used for training.')

flags.DEFINE_integer('emb_size', 128,
                     'Size of the embeddings to learn.')

flags.DEFINE_integer('eval_interval_secs', 300,
                     'How many seconds between executions of the eval loop.')

flags.DEFINE_string('logdir', '/tmp/semisup',
                    'Where the checkpoints are stored '
                    'and eval events will be written to.')

flags.DEFINE_string('master', '',
                    'BNS name of the TensorFlow master to use.')

flags.DEFINE_integer('timeout', 1200,
                     'The maximum amount of time to wait between checkpoints. '
                     'If left as `None`, then the process will wait '
                     'indefinitely.')

flags.DEFINE_integer('max_num_of_eval', None,
                     'The maximum number of evaluations '
                     'If left as `None`, then the process will perform infinitely.')

flags.DEFINE_boolean('extract_image', False,
                     'extract images as file on evaluation')

flags.DEFINE_string('dataset_name', 'test',
                    'string for name of dataset using in evaluation')

flags.DEFINE_boolean('extract_emb', False,
                     'extract embeddings as file on evaluation')

flags.DEFINE_boolean('load_tfrecords', False,
                     'whether load dataset using tfrecords or not')

def main(_):
    # Get dataset-related toolbox.
    dataset_tools = import_module('tools.' + FLAGS.dataset)
    architecture = getattr(semisup.architectures, FLAGS.architecture)

    num_labels = dataset_tools.NUM_LABELS
    image_shape = dataset_tools.IMAGE_SHAPE

    num_examples = dataset_tools.SPLITS_TO_SIZES[FLAGS.dataset_name]

    graph = tf.Graph()
    with graph.as_default():

        if FLAGS.load_tfrecords:
            images, labels = data_util.dataset_to_batch(
                data_util.get_slim_dataset(dataset_tools, FLAGS.dataset_name),
                FLAGS.eval_batch_size)
        else:
            test_images, test_labels = dataset_tools.get_data(FLAGS.dataset_name)
            # Set up input pipeline.
            image, label = tf.train.slice_input_producer([test_images, test_labels])
            images, labels = tf.train.batch(
                [image, label], batch_size=FLAGS.eval_batch_size)
            images = tf.cast(images, tf.float32)
            labels = tf.cast(labels, tf.int64)

        # Reshape if necessary.
        if FLAGS.new_size > 0:
            new_shape = [FLAGS.new_size, FLAGS.new_size, 3]
        else:
            new_shape = None

        # Create function that defines the network.
        model_function = partial(
            architecture,
            is_training=False,
            new_shape=new_shape,
            img_shape=image_shape,
            augmentation_function=None,
            image_summary=False,
            emb_size=FLAGS.emb_size)

        # Set up semisup model.
        model = semisup.SemisupModel(
            model_function,
            num_labels,
            image_shape,
            test_in=images)

        # Add moving average variables.
        for var in tf.get_collection('moving_vars'):
            tf.add_to_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, var)
        for var in slim.get_model_variables():
            tf.add_to_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, var)


        # Get prediction tensor from semisup model.
        predictions = tf.argmax(model.test_logit, 1)
        proba = tf.nn.softmax(model.test_logit)[:, 1]
        # write embeddings to file and extract image as file
        extra_eval_ops = []
        idx_batch = tf.random_uniform((), minval=0, maxval=tf.int64.max/FLAGS.eval_batch_size)

        class ResultFilePathGenerator(object):
            def __init__(self, logdir, dataset, dataset_name):
                self.dataset_name = dataset_name
                self.dataset = dataset
                self.logdir = logdir

            def gen(self, result_type, idx):
                dir = os.path.join(self.logdir, 'eval', self.dataset, self.dataset_name, result_type)
                if not os.path.exists(dir):
                    os.makedirs(dir)
                return tf.string_join((os.path.join(dir, ''), tf.as_string(idx), '.', result_type))

        path_generator = ResultFilePathGenerator(FLAGS.logdir, FLAGS.dataset, FLAGS.dataset_name)

        if FLAGS.extract_emb:
            embeddings = model.test_emb
            extra_eval_ops.append(tf.write_file(path_generator.gen('emb', idx_batch),
                                  tf.reduce_join(tf.as_string(embeddings), [1, 0], separator=',')))

        if FLAGS.extract_image:
            for i in range(FLAGS.eval_batch_size):
                extra_eval_ops.append(tf.write_file(
                    path_generator.gen('png', tf.add(tf.multiply(idx_batch, FLAGS.eval_batch_size), i)),
                                                    tf.image.encode_png(tf.cast(images[i], tf.uint8))))
        if FLAGS.extract_emb or FLAGS.extract_image:
            extra_eval_ops.append(tf.write_file(path_generator.gen('lb', idx_batch),
                                                tf.reduce_join(tf.as_string(labels), separator=',')))
            extra_eval_ops.append(tf.write_file(path_generator.gen('proba', idx_batch),
                                                tf.reduce_join(tf.as_string(proba), separator=',')))
            extra_eval_ops.append(tf.write_file(path_generator.gen('pred', idx_batch),
                                                tf.reduce_join(tf.as_string(predictions), separator=',')))

        # Accuracy metric for summaries.
        metric_dict = {'Accuracy_%s' % FLAGS.dataset_name: slim.metrics.streaming_accuracy(predictions, labels)}
        if num_labels == 2:
            metric_dict['AUC_%s' % FLAGS.dataset_name] = slim.metrics.streaming_auc(tf.nn.softmax(model.test_logit)[:, 1], labels)
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(metric_dict)
        for name, value in names_to_values.iteritems():
            tf.summary.scalar(name, value)

        # Run the actual evaluation loop.
        num_batches = math.ceil(num_examples / float(FLAGS.eval_batch_size))
        # num_batches = 1

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        slim.evaluation.evaluation_loop(
            master=FLAGS.master,
            checkpoint_dir=FLAGS.logdir + '/train',
            logdir=FLAGS.logdir + '/eval',
            num_evals=num_batches,
            eval_op=names_to_updates.values() + extra_eval_ops,
            eval_interval_secs=FLAGS.eval_interval_secs,
            session_config=config,
            max_number_of_evaluations=FLAGS.max_num_of_eval,
            timeout=FLAGS.timeout,
        )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run()
