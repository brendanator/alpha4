from alpha4 import Alpha4
from consts import RED, YELLOW, HEIGHT, WIDTH
from position import Position
import tkinter as tk


class GUI(object):
  def __init__(self):
    # Main window
    self.app = tk.Tk()
    self.app.title('Alpha4')
    self.app.resizable(height=False, width=False)

    # New game buttons
    play_red = tk.Button(
        self.app,
        text='Play as RED',
        command=lambda: self.new_game(RED),
        width=26)
    play_red.grid(row=1, column=0)
    play_yellow = tk.Button(
        self.app,
        text='Play as YELLOW',
        command=lambda: self.new_game(YELLOW),
        width=26)
    play_yellow.grid(row=1, column=1)

    # Board
    board = tk.Frame(self.app)
    board.grid(columnspan=WIDTH)
    self.disks = {}
    for x in range(WIDTH):
      for y in range(HEIGHT):
        disk = tk.Canvas(
            board, width=60, height=50, bg='navy', highlightthickness=0)
        disk.grid(row=HEIGHT - 1 - y, column=x)
        disk.bind('<Button-1>', lambda e: self.player_move(e.x_root // 60))
        self.disks[x, y] = disk

    # Start new game
    self.new_game(RED)

  def new_game(self, play_colour):
    self.position = Position()
    self.alpha4 = Alpha4(self.position)
    self.draw_board()

    if play_colour == YELLOW:
      self.alpha4_move()

  def player_move(self, x):
    if self.position.gameover or x not in self.position.legal_moves():
      return

    self.position.move(x)
    self.draw_board()
    self.alpha4_move()

  def alpha4_move(self):
    if self.position.gameover:
      return

    self.app.config(cursor='watch')
    move = self.alpha4.play()
    if move != None:
      self.position.move(move)
      self.draw_board()
    self.app.config(cursor='')

  def draw_board(self):
    for (x, y), disk in self.disks.items():
      red, yellow = self.position.disks[:, x, y]
      win_line = self.position.gameover and self.position.win[x, y]

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

      disk.create_oval(10, 5, 50, 45, fill=colour, outline='blue', width=1)

  def mainloop(self):
    self.app.mainloop()


if __name__ == '__main__':
  GUI().mainloop()
