import collections
from network import ValueNetwork
import numpy as np
import os
from position import Position
import tensorflow as tf
import util

flags = tf.app.flags
flags.DEFINE_integer('epochs', 1000, 'Number of batches')
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_string('run_dir', 'latest', 'Run directory')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
config = flags.FLAGS


class ValueTraining(object):
  def __init__(self, config):
    self.config = config
    self.run_dir = util.run_directory(config)
    self.position_results = PositionResults(config, self.run_dir)

    self.session = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(
        allow_growth=True)))

    self.value_network = ValueNetwork('value')

    # Train ops
    self.create_train_op(self.value_network)
    self.writer = tf.summary.FileWriter(self.run_dir)

    if not util.restore(self.session, self.run_dir, self.value_network):
      self.session.run(tf.global_variables_initializer())

  def create_train_op(self, value_network):
    self.result = tf.placeholder(tf.float32, shape=[None], name='result')
    loss = tf.reduce_mean(
        tf.squared_difference(self.value_network.value, self.result))
    optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
    self.global_step = tf.contrib.framework.get_or_create_global_step()
    self.train_op = optimizer.minimize(loss, self.global_step)

    # Summary
    tf.summary.scalar('value_loss', loss)
    for var in value_network.variables + value_network.value_layers:
      tf.summary.histogram(var.name, var)
      self.summary = tf.summary.merge_all()

  def train(self):
    for _ in range(self.config.epochs):
      for batch in self.position_results.batches():
        _, step, summary = self.session.run(
            [self.train_op, self.global_step, self.summary], {
                self.value_network.turn: batch.turn,
                self.value_network.disks: batch.disks,
                self.value_network.empty: batch.empty,
                self.value_network.legal_moves: batch.legal_moves,
                self.value_network.threats: batch.threats,
                self.result: batch.results
            })
        self.writer.add_summary(summary, step)

      self.save()

  def save(self):
    util.save(self.session, self.run_dir, self.value_network)


class PositionResults(object):
  def __init__(self, config, run_dir):
    self.config = config

    turn, disks, empty, legal_moves, threats, results = [], [], [], [], [], []
    with open(os.path.join(run_dir, 'examples_positions.txt')) as f:
      print('Loading positions from %s' % f.name)
      for line in f.readlines():
        position_repr, result = line.strip().split(' ')
        position = Position(position_repr)
        result = int(result)

        turn.append(position.turn)
        disks.append(position.disks)
        empty.append(position.empty)
        legal_moves.append(position.legal_moves)
        threats.append(position.threats)
        results.append(result)

    self.turn = np.array(turn)
    self.disks = np.array(disks)
    self.empty = np.array(empty)
    self.legal_moves = np.array(legal_moves)
    self.threats = np.array(threats)
    self.results = np.array(results)
    self.total = len(self.results)

  def batches(self):
    indices = np.random.permutation(self.total)
    splits = np.arange(self.config.batch_size, self.total,
                       self.config.batch_size)

    for batch_indices in np.split(indices, splits):
      yield Batch(self.turn[batch_indices], self.disks[batch_indices],
                  self.empty[batch_indices], self.legal_moves[batch_indices],
                  self.threats[batch_indices], self.results[batch_indices])


Batch = collections.namedtuple(
    'Batch', ['turn', 'disks', 'empty', 'legal_moves', 'threats', 'results'])


def main(_):
  training = ValueTraining(config)
  training.train()


if __name__ == '__main__':
  tf.app.run()
