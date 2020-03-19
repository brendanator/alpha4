import numpy as np

RED = 0
YELLOW = 1
COLOURS = 2
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
        four = np.zeros([HEIGHT, WIDTH], bool)
        for i in range(4):
            four[row, column + i] = True
        FOURS.append(four)
# Vertical
for row in range(HEIGHT - 3):
    for column in range(WIDTH):
        four = np.zeros([HEIGHT, WIDTH], bool)
        for i in range(4):
            four[row + i, column] = True
        FOURS.append(four)
# Diagonal
for row in range(HEIGHT - 3):
    for column in range(WIDTH - 3):
        four1 = np.zeros([HEIGHT, WIDTH], bool)
        four2 = np.zeros([HEIGHT, WIDTH], bool)
        for i in range(4):
            four1[row + i, column + i] = True
            four2[row + 3 - i, column + i] = True
        FOURS.append(four1)
        FOURS.append(four2)
FOURS = np.array(FOURS)

DISK_FOURS = {}
DISK_FOUR_COUNTS = np.zeros([HEIGHT, WIDTH], int)
for row in range(HEIGHT):
    for column in range(WIDTH):
        disk_fours = [four for four in FOURS if four[row, column]]
        DISK_FOURS[row, column] = disk_fours
        DISK_FOUR_COUNTS[row, column] = len(disk_fours)

# Results
RED_WIN = 1
DRAW = 0
YELLOW_WIN = -1

# The position hashing in implemented similarly to Zobrist hashing by xoring
# played moves with the previous hash. The difference is that the disk hashes
# are not random, they are chosen to guarantee each position hash is unique.
# Each position is hashed with 63 bits, with each column represented by 9 bits
# The first 3 bits indicate how many disks have been played in the column
# The remaining 6 bits are the rows in the column that contain a yellow disk
NEW_POSITION_HASH = np.uint64(0)
DISK_HASHES = np.zeros([COLOURS, HEIGHT, WIDTH], np.uint64)
for colour in range(COLOURS):
    for row in range(HEIGHT):
        disks_in_column = row ^ (row + 1)
        yellow_disks = 2**(row + 3) if colour == YELLOW else 0
        row_hash = disks_in_column | yellow_disks
        for column in range(WIDTH):
            row_column_hash = row_hash << (9 * column)
            DISK_HASHES[colour, HEIGHT - row - 1, column] = row_column_hash

if __name__ == '__main__':
    print(FOURS[0])
    print(DISK_FOURS[0, 0])
    print(DISK_FOUR_COUNTS)
    print(TILED_COLUMNS.reshape([HEIGHT, WIDTH]))
    print(TILED_ROWS.reshape([HEIGHT, WIDTH]))
    print(ROW_EDGE_DISTANCE.reshape([HEIGHT, WIDTH]))
    print(COLUMN_EDGE_DISTANCE.reshape([HEIGHT, WIDTH]))
    print(ODDS.reshape([HEIGHT, WIDTH]))
    print(np.array(map(bin, DISK_HASHES.flatten())).reshape(DISK_HASHES.shape))
