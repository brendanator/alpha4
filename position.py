import numpy as np
from consts import RED, YELLOW, HEIGHT, WIDTH, COLUMNS, FOURS


class Position(object):
  def __init__(self):
    self.disks = np.zeros([2, WIDTH, HEIGHT], dtype=np.bool)
    self.turn = RED
    self.gameover = False
    self.win = None

  def move(self, column):
    top = np.count_nonzero(self.disks[:, column])
    self.disks[self.turn, column, top] = True

    for four in FOURS:
      if np.count_nonzero((self.disks[self.turn] == four) & four) == 4:
        self.gameover = True
        self.win = four
        break

    self.turn = 1 - self.turn

  def legal_moves(self):
    counts = np.count_nonzero(self.disks.max(axis=0), axis=1)
    return COLUMNS[counts < HEIGHT]
