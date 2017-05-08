import numpy as np
from consts import HEIGHT, WIDTH


class Alpha4(object):
  def __init__(self, position):
    self.position = position

  def play(self):
    return np.random.choice(self.position.legal_moves())
