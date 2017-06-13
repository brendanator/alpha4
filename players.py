from consts import *
import numpy as np
from position import Position


class Player(object):
  def play(self, positions):
    return [self.play_single(position) for position in positions]

  def play_single(self, position):
    raise NotImplementedError()


class RandomPlayer(Player):
  def __init__(self):
    self.name = 'random'

  def random_move(self, position):
    # Play move proportional to the number of fours it belongs too
    ratios = (DISK_FOUR_COUNTS * [position.legal_moves]).sum(axis=(0, 1))
    probabilities = ratios / ratios.sum()
    return np.random.choice(WIDTH, p=probabilities)

  def play_single(self, position):
    return self.random_move(position)


class RandomThreatPlayer(RandomPlayer):
  def __init__(self):
    self.name = 'random-threat'

  def play_single_threat(self, position):
    if position.turn == RED:
      our_threats, their_threats = position.threats & position.legal_moves
    else:
      their_threats, our_threats = position.threats & position.legal_moves

    if np.any(our_threats):
      return np.argmax(our_threats) % WIDTH
    elif np.any(their_threats):
      return np.argmax(their_threats) % WIDTH
    else:
      return None

  def play_single(self, position):
    move = self.play_single_threat(position)
    if move is not None:
      return move
    else:
      return self.random_move(position)


class MaxThreatPlayer(RandomThreatPlayer):
  def __init__(self):
    self.name = 'max-threat'

  def play_single(self, position):
    move = self.play_single_threat(position)
    if move is not None:
      return move

    moves = position.legal_columns()
    children = [position.move(move) for move in moves]

    max_threats = -TOTAL_DISKS
    best_moves = set()
    for move, child in zip(moves, children):
      if position.turn == RED:
        our_threats, their_threats = child.threats & child.legal_moves
      else:
        their_threats, our_threats = child.threats & child.legal_moves

      threats = np.count_nonzero(our_threats) - np.count_nonzero(their_threats)
      if threats > max_threats:
        max_threats = threats
        best_moves = {move}
      elif threats == max_threats:
        best_moves |= {move}

    # Play best move proportional to the number of fours it belongs too
    ratios = (DISK_FOUR_COUNTS * [position.legal_moves]).sum(axis=(0, 1))
    for mediocre_move in set(range(WIDTH)) - best_moves:
      ratios[mediocre_move] = 0
    probabilities = ratios / ratios.sum()
    return np.random.choice(WIDTH, p=probabilities)


class ParityThreatPlayer(Player):
  def __init__(self):
    self.name = 'parity-threat'

  def play_single(self, position):
    move = self.play_single_threat(position)
    if move is not None:
      return move

    moves = position.legal_moves()
    children = [self.move(move) for move in moves]

    max_threats = -TOTAL_DISKS
    best_moves = []
    for move, child in zip(moves, children):
      if position.turn == RED:
        our_threats, their_threats = position.threats & position.legal_moves
      else:
        their_threats, our_threats = position.threats & position.legal_moves

      threats = np.count_nonzero(our_threats) - np.count_nonzero(their_threats)
      if threats > max_threats:
        max_threats = threats
        best_moves = [move]
      elif threats == max_threats:
        best_moves.append(move)

    return np.random.choice(best_moves)


class NetworkPlayer(Player):
  def __init__(self, network, session):
    self.name = network.scope
    self.network = network
    self.session = session

  def play(self, positions):
    turns = [position.turn for position in positions]
    disks = [position.disks for position in positions]
    empty = [position.empty for position in positions]
    legal_moves = [position.legal_moves for position in positions]
    threats = [position.threats for position in positions]

    policies = self.session.run(self.network.policy, {
        self.network.turn: turns,
        self.network.disks: disks,
        self.network.empty: empty,
        self.network.legal_moves: legal_moves,
        self.network.threats: threats
    })

    moves = []
    for policy, position in zip(policies, positions):
      column = np.random.choice(TILED_COLUMNS, p=policy)
      if position.legal_column(column):
        moves.append(column)
      else:
        print(self.name)
        print(position)
        print(policy.reshape(HEIGHT, WIDTH))
        print(column)
        raise Exception('Illegal column chosen: %d' % column)
    return moves
