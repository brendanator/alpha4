import numpy as np

RED = 0
YELLOW = 1
HEIGHT = 6
WIDTH = 7
COLUMNS = np.arange(WIDTH)

FOURS = []
# Horizontal
for x in range(WIDTH - 3):
  for y in range(HEIGHT):
    four = np.zeros([WIDTH, HEIGHT], np.bool)
    for i in range(4):
      four[x + i, y] = True
    FOURS.append(four)
# Vertical
for y in range(HEIGHT - 3):
  for x in range(WIDTH):
    four = np.zeros([WIDTH, HEIGHT], np.bool)
    for i in range(4):
      four[x, y + i] = True
    FOURS.append(four)
# Diagonal
for x in range(WIDTH - 3):
  for y in range(HEIGHT - 3):
    four1 = np.zeros([WIDTH, HEIGHT], np.bool)
    four2 = np.zeros([WIDTH, HEIGHT], np.bool)
    for i in range(4):
      four1[x + i, y + i] = True
      four2[x + i, y + 3 - i] = True
    FOURS.append(four1)
    FOURS.append(four2)
FOURS = np.array(FOURS)
