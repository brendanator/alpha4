from alpha4 import Alpha4
from consts import RED, YELLOW, HEIGHT, WIDTH
from position import Position
import tensorflow as tf
import tkinter as tk
import sys
import util

flags = tf.app.flags
flags.DEFINE_string('run_dir', 'latest', 'Run directory')
flags.DEFINE_string('policy_network', 'policy', 'Name of policy network')
flags.DEFINE_boolean('threats', False, 'Show threats on board')
config = flags.FLAGS


class GUI(object):
  def __init__(self, config):
    self.threats = config.threats

    # Main window
    self.app = tk.Tk()
    self.app.title('Alpha4')
    self.app.resizable(height=False, width=False)

    # New game buttons
    play_both = tk.Button(
        self.app,
        text='Play 2 player',
        command=lambda: self.new_game([RED, YELLOW]),
        width=11)
    play_both.grid(row=1, column=0)
    play_red = tk.Button(
        self.app,
        text='Play RED',
        command=lambda: self.new_game([RED]),
        width=11)
    play_red.grid(row=1, column=1)
    play_yellow = tk.Button(
        self.app,
        text='Play YELLOW',
        command=lambda: self.new_game([YELLOW]),
        width=11)
    play_yellow.grid(row=1, column=2)
    watch = tk.Button(
        self.app,
        text='Watch Alpha4',
        command=lambda: self.new_game([]),
        width=11)
    watch.grid(row=1, column=3)

    # Board
    board = tk.Frame(self.app)
    board.grid(columnspan=WIDTH)
    self.disks = {}
    for row in range(HEIGHT):
      for column in range(WIDTH):
        disk = tk.Canvas(
            board, width=60, height=50, bg='navy', highlightthickness=0)
        disk.grid(row=row, column=column)
        disk.bind('<Button-1>', lambda e: self.player_move(e.x_root // 60))
        self.disks[row, column] = disk

    # Alpha4
    self.alpha4 = Alpha4(config.policy_network, util.run_directory(config))

    # Start new game
    self.new_game([RED])

  def new_game(self, human_colours):
    self.human_colours = human_colours
    self.position = Position()
    self.display_board()
    self.alpha4_move()

  def player_move(self, column):
    if self.position.gameover() or not self.position.legal_column(column):
      return

    self.position = self.position.move(column)
    self.display_board()
    self.alpha4_move()

  def alpha4_move(self):
    if self.position.gameover() or not self.alpha4_turn():
      return

    self.app.config(cursor='watch')
    self.app.update()
    column = self.alpha4.play(self.position)
    if column != None:
      self.position = self.position.move(column)
      self.display_board()
    self.app.config(cursor='')

    if self.alpha4_turn():
      self.app.after(1000, self.alpha4_move)

  def alpha4_turn(self):
    return self.position.turn not in self.human_colours

  def display_board(self):
    for (row, column), disk in self.disks.items():
      red, yellow = self.position.disks[:, row, column]
      win_line = self.position.result and self.position.win[row, column]

      if win_line and red:
        colour = 'dark violet'
      elif red:
        colour = 'red'
      elif win_line and yellow:
        colour = 'green'
      elif yellow:
        colour = 'yellow'
      else:
        colour = 'black'

      outline = 'blue'
      if self.threats:
        red_threat, yellow_threat = self.position.threats[:, row, column]
        if red_threat and yellow_threat:
          outline = 'orange'
        elif red_threat:
          outline = 'red'
        elif yellow_threat:
          outline = 'yellow'

      disk.create_oval(10, 5, 50, 45, fill=colour, outline=outline, width=1)

  def mainloop(self):
    self.app.mainloop()


def main(_):
  GUI(config).mainloop()


if __name__ == '__main__':
  tf.app.run()
