import numpy as np
from consts import *


class Position(object):
  __slots__ = [
      'turn', 'disks', 'empty', 'legal_moves', 'threats', 'result', 'win'
  ]

  def __init__(self, position=None, column=None, disks=None, turn=RED):
    if position is None:
      # New game
      self.turn = RED
      self.disks = np.zeros([2, HEIGHT, WIDTH], dtype=np.byte)
      self.empty = np.ones([HEIGHT, WIDTH], dtype=np.byte)
      self.legal_moves = np.zeros([HEIGHT, WIDTH], dtype=np.byte)
      self.legal_moves[-1] = True
      # Threats are a single missing disk from a four
      self.threats = np.zeros([2, HEIGHT, WIDTH], dtype=np.byte)
      self.result = None
    elif type(position) == str:
      # Construct from __repr__
      self.disks = np.zeros([2, HEIGHT, WIDTH], dtype=np.byte)
      self.empty = np.ones([HEIGHT, WIDTH], dtype=np.byte)
      self.legal_moves = np.zeros([HEIGHT, WIDTH], dtype=np.byte)

      for index, disk in enumerate(position):
        row = index // WIDTH
        column = index % WIDTH
        if disk == '.':
          if row:
            self.legal_moves[row - 1, column] = False
          self.legal_moves[row, column] = True
          self.empty[row, column] = True
        else:
          colour = RED if disk == 'r' else YELLOW
          self.disks[colour, row, column] = True
          self.legal_moves[row, column] = False
          self.empty[row, column] = False

      self.turn = np.count_nonzero(self.empty) % 2
      self.threats = np.zeros([2, HEIGHT, WIDTH], dtype=np.byte)
      self.result = None
      for index in range(TOTAL_DISKS):
        row = index // WIDTH
        column = index % WIDTH
        self.check_result_and_threats(row, column)
    else:
      # Make move from old position
      self.turn = 1 - position.turn
      self.disks = np.copy(position.disks)
      self.empty = np.copy(position.empty)
      self.legal_moves = np.copy(position.legal_moves)
      self.threats = np.copy(position.threats)
      self.result = None

      row = np.count_nonzero(self.empty[:, column]) - 1
      self.disks[position.turn, row, column] = True
      self.empty[row, column] = False
      self.legal_moves[row, column] = False
      if row:
        self.legal_moves[row - 1, column] = True

      self.check_result_and_threats(row, column)

  def check_result_and_threats(self, row, column):
    if not self.result and not np.any(self.empty):
      self.result = DRAW

    for four in DISK_FOURS[row, column]:
      empty = self.empty & four
      empty_count = np.count_nonzero(empty)
      if empty_count == 0:
        self.threats &= 1 - four
        red_four_count = np.count_nonzero(self.disks[0] & four)
        if red_four_count == 4:
          self.result = RED_WIN
          self.win = four
        elif red_four_count == 0:
          self.result = YELLOW_WIN
          self.win = four
      elif empty_count == 1:
        red_four_count = np.count_nonzero(self.disks[0] & four)
        if red_four_count == 3:
          self.threats[RED] |= empty
        elif red_four_count == 0:
          self.threats[YELLOW] |= empty

  def gameover(self):
    return self.result is not None

  def legal_columns(self):
    return COLUMNS[np.count_nonzero(self.empty, axis=0) > 0]

  def legal_column(self, column):
    return np.any(self.legal_moves[:, column])

  def move(self, column):
    if not self.gameover():
      return Position(self, column)
    else:
      return self

  def children(self):
    return [self.move(move) for move in self.legal_columns()]

  def __str__(self):
    result = ''
    for row in range(HEIGHT):
      for column in range(WIDTH):
        red, yellow = self.disks[:, row, column]
        if red:
          result += 'r'
        elif yellow:
          result += 'y'
        else:
          result += '.'
      result += '\n'
    return result.strip()

  def __repr__(self):
    return self.__str__().replace('\n', '')


if __name__ == '__main__':
  # Test start position
  pos = Position()
  assert (pos.turn == RED)
  assert (np.count_nonzero(pos.disks) == 0)
  assert (np.count_nonzero(pos.empty) == TOTAL_DISKS)
  assert (np.count_nonzero(pos.legal_moves) == WIDTH)
  assert (np.count_nonzero(pos.threats) == 0)
  assert (pos.result is None)
  assert (np.all(pos.legal_columns() == np.arange(WIDTH)))
  for column in range(WIDTH):
    assert (pos.legal_column(column))
  assert (len(pos.children()) == WIDTH)

  # Play single move
  pos = pos.move(2)
  assert (pos.turn == YELLOW)
  assert (np.count_nonzero(pos.disks) == 1)
  assert (np.count_nonzero(pos.empty) == TOTAL_DISKS - 1)
  assert (np.count_nonzero(pos.legal_moves) == WIDTH)
  assert (np.count_nonzero(pos.threats) == 0)
  assert (pos.result is None)

  # Each colour has 3 vertical disks
  pos = pos.move(3)
  pos = pos.move(2)
  pos = pos.move(3)
  pos = pos.move(2)
  pos = pos.move(3)
  assert (pos.turn == RED)
  assert (np.count_nonzero(pos.disks) == 6)
  assert (np.count_nonzero(pos.empty) == TOTAL_DISKS - 6)
  assert (np.count_nonzero(pos.legal_moves) == WIDTH)
  assert (np.count_nonzero(pos.threats[RED]) == 1)
  assert (np.count_nonzero(pos.threats[YELLOW]) == 1)
  assert (pos.result is None)

  # Fill a column to check legal moves
  pos = pos.move(3)
  pos = pos.move(3)
  pos = pos.move(3)
  assert (pos.turn == YELLOW)
  assert (np.count_nonzero(pos.disks) == 9)
  assert (np.count_nonzero(pos.empty) == TOTAL_DISKS - 9)
  assert (np.count_nonzero(pos.legal_moves) == WIDTH - 1)
  assert (np.count_nonzero(pos.threats[RED]) == 1)
  assert (np.count_nonzero(pos.threats[YELLOW]) == 0)
  assert (pos.result is None)
  for column in range(WIDTH):
    assert (pos.legal_column(column) == (column != 3))
  assert (len(pos.children()) == WIDTH - 1)

  # Red wins the game
  pos = pos.move(4)
  pos = pos.move(2)
  assert (pos.turn == YELLOW)
  assert (np.count_nonzero(pos.disks) == 11)
  assert (np.count_nonzero(pos.empty) == TOTAL_DISKS - 11)
  assert (np.count_nonzero(pos.legal_moves) == WIDTH - 1)
  assert (pos.result is RED_WIN)

  # Test __str__
  assert (pos.__str__() == ('...r...\n'
                            '...y...\n'
                            '..rr...\n'
                            '..ry...\n'
                            '..ry...\n'
                            '..ryy..'))

  # Test __repr__
  pos_repr = Position(repr(pos))
  assert (pos.turn == pos_repr.turn)
  assert (np.all(pos.disks == pos_repr.disks))
  assert (np.all(pos.empty == pos_repr.empty))
  assert (np.all(pos.legal_moves == pos_repr.legal_moves))
  assert (np.all(pos.threats == pos_repr.threats))
  assert (pos.result == pos_repr.result)
  assert (np.all(pos.win == pos_repr.win))
