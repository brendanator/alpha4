import collections
from network import ValueNetwork
import numpy as np
import os
from position import Position
import tensorflow as tf
import util

flags = tf.app.flags
flags.DEFINE_string('run_dir', 'latest', 'Run directory')
flags.DEFINE_integer('epochs', 100, 'Number of batches')
flags.DEFINE_integer('batch_size', 256, 'Batch size')
flags.DEFINE_float('train_proportion', 0.9,
                   'Split between train and validation sets')
flags.DEFINE_float('initial_learning_rate', 0.001, 'Initial learning rate')
flags.DEFINE_float('learning_rate_decay', 0.1,
                   'Learning rate decay on validation loss increase')
flags.DEFINE_float('discount_rate', 0.99, 'Result discount rate')
config = flags.FLAGS


class ValueTraining(object):
  def __init__(self, config):
    self.config = config
    self.run_dir = util.run_directory(config)
    self.position_targets = PositionTargets(config, self.run_dir)

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
    with tf.variable_scope('value_training') as self.training_scope:
      self.learning_rate = tf.get_variable(
          'learning_rate', initializer=self.config.initial_learning_rate)
      self.decay_learning_rate = tf.assign(
          self.learning_rate,
          self.learning_rate * self.config.learning_rate_decay)
      optimizer = tf.train.AdamOptimizer(self.learning_rate)

      self.target = tf.placeholder(tf.float32, shape=[None], name='target')
      self.loss = tf.reduce_mean(
          tf.squared_difference(self.value_network.value, self.target))

      self.global_step = tf.contrib.framework.get_or_create_global_step()
      self.train_op = optimizer.minimize(self.loss, self.global_step)

      # Summary
      tf.summary.scalar('loss', self.loss)
      tf.summary.scalar('learning_rate', self.learning_rate)
      for var in value_network.variables + value_network.value_layers:
        tf.summary.histogram(var.name, var)
        self.summary = tf.summary.merge_all()

  def train(self):
    best_validation = self.calculate_validation_loss()
    for epoch in range(1, self.config.epochs + 1):
      self.train_epoch()

      validation_loss = self.calculate_validation_loss()
      if validation_loss < best_validation:
        best_validation = validation_loss
        self.save()
      else:
        self.session.run(self.decay_learning_rate)

      print('Epoch %d complete, validation loss %f' % (epoch, validation_loss))

  def train_epoch(self):
    for batch in self.position_targets.train_batches():
      if self.step() % 100 == 0:
        _, summary = self.session.run([self.train_op, self.summary],
                                      self.feed_dict(batch))
        self.writer.add_summary(summary, self.step())
      else:
        self.session.run([self.train_op], self.feed_dict(batch))

  def calculate_validation_loss(self):
    validation_losses = []
    for batch in self.position_targets.validation_batches():
      loss = self.session.run([self.loss], self.feed_dict(batch))
      validation_losses.append(loss)
    validation_loss = np.mean(validation_losses)

    validation_summary = tf.Summary()
    validation_summary.value.add(
        tag=self.training_scope.name + '/validation_loss',
        simple_value=validation_loss)
    self.writer.add_summary(validation_summary, self.step())

    return validation_loss

  def step(self):
    return self.global_step.eval(self.session)

  def feed_dict(self, batch):
    return {
        self.value_network.turn: batch.turn,
        self.value_network.disks: batch.disks,
        self.value_network.empty: batch.empty,
        self.value_network.legal_moves: batch.legal_moves,
        self.value_network.threats: batch.threats,
        self.target: batch.targets
    }

  def save(self):
    util.save_network(self.session, self.run_dir, self.value_network)
    util.save_scope(self.session, self.run_dir, self.training_scope.name)


class PositionTargets(object):
  def __init__(self, config, run_dir):
    self.config = config

    turn, disks, empty, legal_moves, threats, targets = [], [], [], [], [], []
    with open(os.path.join(run_dir, 'rollout_positions.txt')) as f:
      print('Loading positions from %s' % f.name)
      for line in f.readlines():
        position, sample_move, num_moves, result = line.strip().split(' ')
        position = Position(position)
        sample_move = int(sample_move)
        num_moves = int(num_moves)
        result = int(result)

        turn.append(position.turn)
        disks.append(position.disks)
        empty.append(position.empty)
        legal_moves.append(position.legal_moves)
        threats.append(position.threats)
        targets.append(result * 0.95**(num_moves - sample_move))

    turn = np.array(turn)
    disks = np.array(disks)
    empty = np.array(empty)
    legal_moves = np.array(legal_moves)
    threats = np.array(threats)
    targets = np.array(targets)
    count = len(targets)

    # Permute examples
    indices = np.random.permutation(count)
    turn = turn[indices]
    disks = disks[indices]
    empty = empty[indices]
    legal_moves = legal_moves[indices]
    threats = threats[indices]
    targets = targets[indices]

    # Split into train and validation sets
    train_count = int(count * config.train_proportion)
    train_turn, validation_turn = np.split(turn, [train_count])
    train_disks, validation_disks = np.split(disks, [train_count])
    train_empty, validation_empty = np.split(empty, [train_count])
    train_legal_moves, validation_legal_moves = np.split(
        legal_moves, [train_count])
    train_threats, validation_threats = np.split(threats, [train_count])
    train_targets, validation_targets = np.split(targets, [train_count])

    self.train_inputs = Inputs(train_turn, train_disks, train_empty,
                               train_legal_moves, train_threats, train_targets,
                               train_count)

    self.validation_inputs = Inputs(validation_turn, validation_disks,
                                    validation_empty, validation_legal_moves,
                                    validation_threats, validation_targets,
                                    count - train_count)

  def train_batches(self):
    return self.batches(self.train_inputs)

  def validation_batches(self):
    return self.batches(self.validation_inputs)

  def batches(self, inputs):
    indices = np.random.permutation(inputs.count)
    splits = np.arange(self.config.batch_size, inputs.count,
                       self.config.batch_size)

    for batch_indices in np.split(indices, splits):
      yield Inputs(
          inputs.turn[batch_indices], inputs.disks[batch_indices],
          inputs.empty[batch_indices], inputs.legal_moves[batch_indices],
          inputs.threats[batch_indices], inputs.targets[batch_indices],
          len(batch_indices))


Inputs = collections.namedtuple('Inputs', [
    'turn', 'disks', 'empty', 'legal_moves', 'threats', 'targets', 'count'
])


def main(_):
  training = ValueTraining(config)
  training.train()


if __name__ == '__main__':
  tf.app.run()
