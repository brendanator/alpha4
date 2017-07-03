from consts import *
import numpy as np
from position import Position


class Player(object):
  def play(self, positions):
    return [self.play_single(position) for position in positions]

  def play_single(self, position):
    return self.play([position])[0]


class RandomPlayer(Player):
  def __init__(self):
    self.name = 'random'

  def random_move(self, position):
    # Play move proportional to the number of fours it belongs to
    ratios = (DISK_FOUR_COUNTS * [position.legal_moves]).sum(axis=(0, 1))
    probabilities = ratios / ratios.sum()
    return np.random.choice(WIDTH, p=probabilities)

  def play_single(self, position):
    return self.random_move(position)


class RandomThreatPlayer(RandomPlayer):
  def __init__(self):
    self.name = 'random-threat'

  def play_single(self, position):
    if position.counter_move is not None:
      return position.counter_move
    else:
      return self.random_move(position)


class MaxThreatPlayer(RandomThreatPlayer):
  def __init__(self):
    self.name = 'max-threat'

  def play_single(self, position):
    if position.counter_move is not None:
      return position.counter_move

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


class PolicyPlayer(Player):
  def __init__(self, policy_network, session):
    self.name = policy_network.scope
    self.policy_network = policy_network
    self.session = session

  def play(self, positions):
    turns = [position.turn for position in positions]
    disks = [position.disks for position in positions]
    empty = [position.empty for position in positions]
    legal_moves = [position.legal_moves for position in positions]
    threats = [position.threats for position in positions]

    policies = self.session.run(self.policy_network.policy, {
        self.policy_network.turn: turns,
        self.policy_network.disks: disks,
        self.policy_network.empty: empty,
        self.policy_network.legal_moves: legal_moves,
        self.policy_network.threats: threats
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


class ValuePlayer(Player):
  def __init__(self, value_network, session):
    self.name = value_network.scope
    self.value_network = value_network
    self.session = session

  def play_single(self, position):
    children = position.children()
    turns = [child.turn for child in children]
    disks = [child.disks for child in children]
    empty = [child.empty for child in children]
    legal_moves = [child.legal_moves for child in children]
    threats = [child.threats for child in children]

    child_values = self.session.run(self.value_network.value, {
        self.value_network.turn: turns,
        self.value_network.disks: disks,
        self.value_network.empty: empty,
        self.value_network.legal_moves: legal_moves,
        self.value_network.threats: threats
    })

    if position.turn == RED:
      index = np.argmax(child_values)
    else:
      index = np.argmin(child_values)

    return position.legal_columns()[index]
