from consts import *
import collections
from network import ValueNetwork
import numpy as np
import os
from position import Position
import tensorflow as tf
import util

flags = tf.app.flags
flags.DEFINE_integer('num_games', 1000, 'Number of games')
flags.DEFINE_integer('epochs', 1000, 'Number of batches')
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_string('run_dir', 'latest', 'Run directory')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
flags.DEFINE_float('lambda_discount', 0.95, 'Lambda return discount')
config = flags.FLAGS


class TDLambdaTraining(object):
  def __init__(self, config):
    self.config = config
    self.run_dir = util.run_directory(config)

    self.session = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(
        allow_growth=True)))

    self.value_network = ValueNetwork('value')
    util.restore_or_initialize_network(self.session, self.run_dir,
                                       self.value_network)

    # Train ops
    self.create_train_op(self.value_network)
    self.writer = tf.summary.FileWriter(self.run_dir)
    util.restore_or_initialize_scope(self.session, self.run_dir,
                                     self.training_scope.name)

  def create_train_op(self, value_network):
    with tf.variable_scope('td_lambda_training') as self.training_scope:
      self.target = tf.placeholder(tf.float32, shape=[None], name='target')
      loss = tf.reduce_mean(
          tf.squared_difference(self.value_network.value, self.target))
      optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate)
      self.global_step = tf.contrib.framework.get_or_create_global_step()
      self.train_op = optimizer.minimize(loss, self.global_step)

      # Summary
      tf.summary.scalar('value_loss', loss)
      tf.summary.histogram('target', self.target)
      for var in value_network.variables + value_network.value_layers:
        tf.summary.histogram(var.name, var)
        self.summary = tf.summary.merge_all()

  def train(self):
    for _ in range(self.config.num_games):
      positions, lambda_returns = [], []
      for _ in range(32):
        pos, values = self.play_game()
        positions += pos
        lambda_returns += self.lambda_returns(values)

      feed_dict = {
          self.value_network.turn: [p.turn for p in positions],
          self.value_network.disks: [p.disks for p in positions],
          self.value_network.empty: [p.empty for p in positions],
          self.value_network.legal_moves: [p.legal_moves for p in positions],
          self.value_network.threats: [p.threats for p in positions],
          self.target: lambda_returns
      }
      _, step, summary = self.session.run(
          [self.train_op, self.global_step, self.summary], feed_dict)

      if step % 100 == 0:
        self.writer.add_summary(summary, step)
        self.save()

  def play_game(self):
    position = Position()
    while np.random.rand() < 0.85:
      position = position.move(np.random.choice(position.legal_columns()))
      if position.gameover():
        position = Position()
    positions = [position]
    values = []

    while not position.gameover():
      children = position.children()
      child_values = self.session.run(self.value_network.value, {
          self.value_network.turn: [p.turn for p in children],
          self.value_network.disks: [p.disks for p in children],
          self.value_network.empty: [p.empty for p in children],
          self.value_network.legal_moves: [p.legal_moves for p in children],
          self.value_network.threats: [p.threats for p in children]
      })

      # Determine best move
      if position.turn == RED:
        index = np.argmax(child_values)
      else:
        index = np.argmin(child_values)

      position = children[index]

      if not position.gameover():
        positions.append(position)

        value = child_values[index]
        values.append(value)
      else:
        values.append(position.result)

    return positions, values

  def lambda_returns(self, values):
    lambda_ = self.config.lambda_discount
    target = values[-1]
    targets = [target]
    for value in reversed(values[:-1]):
      target = lambda_ * target + (1 - lambda_) * value
      targets.insert(0, target)
    return targets

  def save(self):
    util.save_network(self.session, self.run_dir, self.value_network)
    util.save_scope(self.session, self.run_dir, self.training_scope.name)


def main(_):
  training = TDLambdaTraining(config)
  training.train()


if __name__ == '__main__':
  tf.app.run()
