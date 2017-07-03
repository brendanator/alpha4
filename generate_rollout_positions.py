from network import PolicyNetwork
import numpy as np
import os
from players import PolicyPlayer, RandomPlayer
from position import Position
import tensorflow as tf
import util

flags = tf.app.flags
flags.DEFINE_string('run_dir', 'latest', 'Run directory')
flags.DEFINE_string('exploratory_network', 'policy',
                    'Name of exploratory player')
flags.DEFINE_string('playout_network', 'policy', 'Name of playout player')
flags.DEFINE_float('exploratory_temperature', 20.0,
                   'Softmax temperature in exploratory network')
flags.DEFINE_float('playout_temperature', 1.0,
                   'Softmax temperature in playout network')
flags.DEFINE_integer('max_sample_move', 30, '')
flags.DEFINE_integer('num_games', 100000, 'Number of games per sample move')
flags.DEFINE_integer('max_random_moves', 10,
                     'Max random moves before exploratory player')
config = flags.FLAGS


class PositionGenerator(object):
  def __init__(self, config):
    self.config = config

    session = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(
        allow_growth=True)))
    self.random_player = RandomPlayer()
    self.exploratory_network = PolicyNetwork(config.exploratory_network)
    self.exploratory_player = PolicyPlayer(self.exploratory_network, session)

    self.playout_network = PolicyNetwork(
        config.playout_network,
        reuse=config.exploratory_network == config.playout_network)
    self.playout_player = PolicyPlayer(self.playout_network, session)

    self.run_dir = util.run_directory(config)
    util.restore_network_or_fail(session, self.run_dir,
                                 self.exploratory_network)
    util.restore_network_or_fail(session, self.run_dir, self.playout_network)

  def generate_positions(self):
    with open(os.path.join(self.run_dir, 'rollout_positions.txt'), 'w') as f:
      for sample_move in range(3, self.config.max_sample_move):
        print('Generating rollouts after %d exploratory moves' % sample_move)
        examples = self.play_games(config.num_games, sample_move)
        for position, sample_move, num_moves, result in examples:
          f.write('%r %d %d %d\n' % (position, sample_move, num_moves, result))

  def play_games(self, num_games, sample_move):
    # Create games
    games = [Game() for _ in range(num_games)]

    # Initialize with random setup
    random_moves = min(self.config.max_random_moves, sample_move - 1)
    for _ in range(random_moves):
      self.play_move(games, self.random_player)

    # Play exploratory moves
    random_setup_games = games
    for _ in range(random_moves, sample_move - 1):
      self.play_move(games, self.exploratory_player)

    # Remove finished games
    games = [game for game in games if not game.position.gameover()]

    # Play random move
    self.play_move(games, self.random_player)
    # Save positions
    sample_positions = [game.position for game in games]

    # Playout game
    incomplete_games = games
    while incomplete_games:
      self.play_move(incomplete_games, self.playout_player)
      incomplete_games = [
          game for game in incomplete_games if not game.position.gameover()
      ]

    results = [game.result for game in games]
    sample_moves = [
        np.count_nonzero(position.disks) for position in sample_positions
    ]
    num_moves = [game.num_moves for game in games]
    position_results = list(
        zip(sample_positions, sample_moves, num_moves, results))

    if len(position_results) < num_games:
      return position_results + self.play_games(
          num_games - len(position_results), sample_move)
    else:
      return position_results

  def play_move(self, games, player):
    positions = [game.position for game in games]
    moves = player.play(positions)

    for game, move in zip(games, moves):
      game.move(move)


class Game(object):
  def __init__(self):
    self.position = Position()
    self.num_moves = 0
    self.result = None

  def move(self, move):
    if not self.position.gameover():
      self.position = self.position.move(move)
      self.num_moves += 1

      if self.position.gameover():
        self.result = self.position.result


def main(_):
  position_generator = PositionGenerator(config)
  position_generator.generate_positions()


if __name__ == '__main__':
  tf.app.run()
