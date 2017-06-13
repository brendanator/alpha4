import numpy as np

RED = 0
YELLOW = 1
HEIGHT = 6
WIDTH = 7
TOTAL_DISKS = HEIGHT * WIDTH
ROWS = np.arange(HEIGHT)
COLUMNS = np.arange(WIDTH)
EPSILON = 1e-20
ILLEGAL_PENALTY = 100

TILED_ROWS = np.arange(TOTAL_DISKS - 1, -1, -1) // WIDTH
TILED_COLUMNS = np.arange(TOTAL_DISKS) % WIDTH

ROW_EDGE_DISTANCE = np.min([TILED_ROWS, np.flip(TILED_ROWS, axis=0)], axis=0)
COLUMN_EDGE_DISTANCE = np.min(
    [TILED_COLUMNS, np.flip(TILED_COLUMNS, axis=0)], axis=0)
ODDS = TILED_ROWS % 2

FOURS = []
# Horizontal
for row in range(HEIGHT):
  for column in range(WIDTH - 3):
    four = np.zeros([HEIGHT, WIDTH], np.byte)
    for i in range(4):
      four[row, column + i] = True
    FOURS.append(four)
# Vertical
for row in range(HEIGHT - 3):
  for column in range(WIDTH):
    four = np.zeros([HEIGHT, WIDTH], np.byte)
    for i in range(4):
      four[row + i, column] = True
    FOURS.append(four)
# Diagonal
for row in range(HEIGHT - 3):
  for column in range(WIDTH - 3):
    four1 = np.zeros([HEIGHT, WIDTH], np.byte)
    four2 = np.zeros([HEIGHT, WIDTH], np.byte)
    for i in range(4):
      four1[row + i, column + i] = True
      four2[row + 3 - i, column + i] = True
    FOURS.append(four1)
    FOURS.append(four2)
FOURS = np.array(FOURS)

DISK_FOURS = {}
DISK_FOUR_COUNTS = np.zeros([HEIGHT, WIDTH], np.byte)
for row in range(HEIGHT):
  for column in range(WIDTH):
    disk_fours = []
    for four in FOURS:
      if four[row, column]:
        disk_fours.append(four)
    DISK_FOURS[row, column] = disk_fours
    DISK_FOUR_COUNTS[row, column] = len(disk_fours)

# Results
RED_WIN = 1
DRAW = 0
YELLOW_WIN = -1

if __name__ == '__main__':
  print(FOURS[0])
  print(DISK_FOURS[0, 0])
  print(DISK_FOUR_COUNTS)
  print(TILED_COLUMNS.reshape([HEIGHT, WIDTH]))
  print(TILED_ROWS.reshape([HEIGHT, WIDTH]))
  print(ROW_EDGE_DISTANCE.reshape([HEIGHT, WIDTH]))
  print(COLUMN_EDGE_DISTANCE.reshape([HEIGHT, WIDTH]))
  print(ODDS.reshape([HEIGHT, WIDTH]))
