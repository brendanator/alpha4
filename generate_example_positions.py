from network import Network
import numpy as np
import os
from players import NetworkPlayer, RandomPlayer
from position import Position
import tensorflow as tf
import util

flags = tf.app.flags
flags.DEFINE_string('run_dir', 'latest', 'Run directory')
flags.DEFINE_string('exploratory_player', 'network-1',
                    'Name of exploratory player')
flags.DEFINE_string('playout_player', 'alpha4', 'Name of playout player')
flags.DEFINE_integer('max_sample_move', 30, '')
flags.DEFINE_integer('num_games', 1000, 'Number of games per sample move')
flags.DEFINE_integer('max_random_moves', 10,
                     'Max random moves before exploratory player')
config = flags.FLAGS


class PositionGenerator(object):
  def __init__(self, config):
    self.config = config

    session = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(
        allow_growth=True)))
    self.random_player = RandomPlayer()
    self.exploratory_player = NetworkPlayer(
        Network(config.exploratory_player), session)
    self.playout_player = NetworkPlayer(
        Network(config.playout_player), session)

    self.run_dir = util.run_directory(config)
    if not util.restore(session, self.run_dir, 'policy'):
      raise Exception('Checkpoint not found in %s' % self.run_dir)

  def generate_positions(self):
    examples = []
    for sample_move in range(1, self.config.max_sample_move):
      examples += self.play_games(config.num_games, sample_move)

    with open(os.path.join(self.run_dir, 'examples_positions.txt'), 'w') as f:
      for position, result in examples:
        f.write(repr(position) + ' ' + repr(result) + '\n')

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

    # Play random move
    self.play_move(games, self.random_player)
    # Remove complete games
    games = [game for game in games if not game.position.gameover()]
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
    position_results = list(zip(sample_positions, results))

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
    self.result = None

  def move(self, move):
    self.position = self.position.move(move)
    if self.position.gameover():
      self.result = self.position.result


def main(_):
  position_generator = PositionGenerator(config)
  position_generator.generate_positions()


if __name__ == '__main__':
  tf.app.run()
