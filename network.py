from consts import *
import os
import tensorflow as tf
import util


class Network(object):
  def __init__(self, scope):
    self.scope = scope

    with tf.variable_scope(scope):
      self.turn = tf.placeholder(tf.float32, shape=[None], name='turn')
      tiled_turn = tf.tile(
          tf.reshape(util.turn_win(self.turn), [-1, 1, 1, 1]),
          [1, 2, HEIGHT, WIDTH])

      self.disks = tf.placeholder(
          tf.float32, shape=[None, 2, HEIGHT, WIDTH], name='disks')

      self.empty = tf.placeholder(
          tf.float32, shape=[None, HEIGHT, WIDTH], name='empty')
      empty = tf.expand_dims(self.empty, axis=1)

      self.legal_moves = tf.placeholder(
          tf.float32, shape=[None, HEIGHT, WIDTH], name='legal_moves')
      legal_moves = tf.expand_dims(self.legal_moves, axis=1)

      self.threats = tf.placeholder(
          tf.float32, shape=[None, 2, HEIGHT, WIDTH], name='threats')

      constant_features = np.array(
          [TILED_ROWS, ODDS, ROW_EDGE_DISTANCE, COLUMN_EDGE_DISTANCE],
          dtype=np.float32).reshape([1, 4, HEIGHT, WIDTH])
      batch_size = tf.shape(self.turn)[0]
      tiled_constant_features = tf.tile(constant_features,
                                        [batch_size, 1, 1, 1])

      feature_planes = tf.concat(
          [
              tiled_turn, self.disks, empty, legal_moves, self.threats,
              tiled_constant_features
          ],
          axis=1)

      conv1 = tf.layers.conv2d(
          feature_planes,
          filters=32,
          kernel_size=[4, 5],
          padding='same',
          data_format='channels_first',
          activation=tf.nn.relu,
          name='conv1')
      conv2 = tf.layers.conv2d(
          conv1,
          filters=32,
          kernel_size=[4, 5],
          padding='same',
          data_format='channels_first',
          activation=tf.nn.relu,
          name='conv2')
      conv3 = tf.layers.conv2d(
          conv2,
          filters=32,
          kernel_size=[4, 5],
          padding='same',
          data_format='channels_first',
          activation=tf.nn.relu,
          name='conv3')
      final_conv = tf.layers.conv2d(
          conv3,
          filters=1,
          kernel_size=[1, 1],
          data_format='channels_first',
          name='final_conv')

      conv_layers = [conv1, conv2, conv3, final_conv]

      with tf.name_scope('policy'):
        disk_bias = tf.get_variable('disk_bias', shape=[HEIGHT, WIDTH])
        disk_logits = tf.add(final_conv, disk_bias, name='disk_logits')

        # Make illegal moves impossible
        legal_disk_logits = tf.nn.relu(disk_logits) * legal_moves
        illegal_penalty = (legal_moves - 1) * ILLEGAL_PENALTY
        legal_disk_logits = tf.contrib.layers.flatten(
            tf.add(
                legal_disk_logits, illegal_penalty, name='legal_disk_logits'))

        self.policy = tf.nn.softmax(legal_disk_logits, name='policy')

        self.entropy = tf.reduce_sum(
            self.policy * -tf.log(self.policy + EPSILON),  # Avoid Nans
            axis=1,
            name='entropy')

        self.policy_layers = conv_layers + [
            disk_logits, self.policy, self.entropy
        ]

      with tf.name_scope('value'):
        fully_connected = tf.layers.dense(
            final_conv, 256, name='fully_connected')
        self.value = tf.layers.dense(fully_connected, 1, tf.tanh, name='value')

        self.value_layers = conv_layers + [fully_connected, self.value]

  @property
  def variables(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

  def assign(self, other):
    copy_ops = []
    for self_var, other_var in zip(self.variables, other.variables):
      copy_ops.append(tf.assign(other_var, self_var))
    return copy_ops
