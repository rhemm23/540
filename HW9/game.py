import random

class Teeko2Player:
  board = [[' ' for _ in range(5)] for _ in range(5)]
  pieces = ['b', 'r']

  def __init__(self):
    self.my_piece = random.choice(self.pieces)
    self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

  def min_value(self, state, depth):
    gv = self.heuristic_game_value(state)
    if gv == 1 or gv == -1 or depth == 3:
      return (gv, state)
    else:
      succs = self.succ(state, self.opp)
      min_val = float('inf')
      min_succ = None
      for succ in succs:
        val, _ = self.max_value(succ, depth + 1)
        if val < min_val:
          min_succ = succ
          min_val = val
      return min_val, min_succ

  def max_value(self, state, depth):
    gv = self.heuristic_game_value(state)
    if gv == 1 or gv == -1 or depth == 3:
      return gv, state
    else:
      succs = self.succ(state, self.my_piece)
      max_val = float('-inf')
      max_succ = None
      for succ in succs:
        val, _ = self.min_value(succ, depth + 1)
        if val > max_val:
          max_succ = succ
          max_val = val
      return max_val, max_succ

  def make_move(self, state):
    _, new_state = self.max_value(state, 0)
    new_piece = None
    rem_piece = None
    move = []
    cnt = 0

    for row in range(5):
      for col in range(5):
        if state[row][col] == ' ' and new_state[row][col] == self.my_piece:
          new_piece = (row, col)
        elif new_state[row][col] == ' ' and state[row][col] == self.my_piece:
          rem_piece = (row, col)
        cnt += 1 if state[row][col] != ' ' else 0

    if cnt < 8:
      return [new_piece]
    return [new_piece, rem_piece]

  def succ(self, state, piece):
    cnt = 0
    for row in range(5):
      for col in range(5):
        if state[row][col] != ' ':
          cnt += 1
    succs = []
    if cnt < 8:
      for row in range(5):
        for col in range(5):
          if state[row][col] == ' ':
            scopy = []
            for r in range(5):
              rc = []
              for c in range(5):
                rc.append(state[r][c])
              scopy.append(rc)
            scopy[row][col] = piece
            succs.append(scopy)
    else:
      for row in range(5):
        for col in range(5):
          if state[row][col] == piece:
            for drow in range(-1, 2):
              for dcol in range(-1, 2):
                if (drow != 0 or dcol != 0):
                  urow = row + drow
                  ucol = col + dcol
                  if urow >= 0 and urow < 5 and ucol >= 0 and ucol < 5 and state[urow][ucol] == ' ':
                    scopy = []
                    for r in range(5):
                      rc = []
                      for c in range(5):
                        rc.append(state[r][c])
                      scopy.append(rc)
                    scopy[row][col] = ' '
                    scopy[urow][ucol] = piece
                    succs.append(scopy)
    return succs

  def opponent_move(self, move):

    # validate input
    if len(move) > 1:
      source_row = move[1][0]
      source_col = move[1][1]
      if source_row != None and self.board[source_row][source_col] != self.opp:
        self.print_board()
        print(move)
        raise Exception("You don't have a piece there!")
      if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
        self.print_board()
        print(move)
        raise Exception('Illegal move: Can only move to an adjacent space')
    if self.board[move[0][0]][move[0][1]] != ' ':
      raise Exception("Illegal move detected")
    # make move
    self.place_piece(move, self.opp)

  def place_piece(self, move, piece):
    if len(move) > 1:
      self.board[move[1][0]][move[1][1]] = ' '
    self.board[move[0][0]][move[0][1]] = piece

  def print_board(self):
    for row in range(len(self.board)):
      line = str(row) + ": "
      for cell in self.board[row]:
        line += cell + " "
      print(line)
    print("   A B C D E")

  def heuristic_game_value(self, state):
    
    # Initial evaluation
    raw = self.game_value(state)
    if raw != 0:
      return raw

    cnt = 0
    
    # three horizontal
    for row in range(5):
      for col in range(3):
        if state[row][col] != ' ' and state[row][col] == state[row][col + 1] == state[row][col + 2]:
          cnt += 1 if state[row][col] == self.my_piece else -1

    # three vertical
    for row in range(3):
      for col in range(5):
        if state[row][col] != ' ' and state[row][col] == state[row + 1][col] == state[row + 2][col]:
          cnt += 1 if state[row][col] == self.my_piece else -1

    # three diagonal
    for row in range(3):
      for col in range(3):
        if state[row][col] != ' ' and state[row][col] == state[row + 1][col + 1] == state[row + 2][col + 2]:
          cnt += 1 if state[row][col] == self.my_piece else -1

    # three diagonal
    for row in range(2, 5):
      for col in range(3):
        if state[row][col] != ' ' and state[row][col] == state[row - 1][col + 1] == state[row - 2][col + 2]:
          cnt += 1 if state[row][col] == self.my_piece else -1

    return cnt / 25

  def game_value(self, state):

    # check horizontal wins
    for row in state:
      for col in range(2):
        if row[col] != ' ' and row[col] == row[col + 1] == row[col + 2] == row[col + 3]:
          return 1 if row[col] == self.my_piece else -1

    # check vertical wins
    for col in range(5):
      for row in range(2):
        if state[row][col] != ' ' and state[row][col] == state[row + 1][col] == state[row + 2][col] == state[row + 3][col]:
          return 1 if state[row][col] == self.my_piece else -1

    # check \ diagonal wins
    for row in range(2):
      for col in range(2):
        if state[row][col] != ' ' and state[row][col] == state[row + 1][col + 1] == state[row + 2][col + 2] == state[row + 3][col + 3]:
          return 1 if state[row][col] == self.my_piece else -1

    # check / diagonal wins
    for row in range(3, 5):
      for col in range(2):
        if state[row][col] != ' ' and state[row][col] == state[row - 1][col + 1] == state[row - 2][col + 2] == state[row - 3][col + 3]:
          return 1 if state[row][col] == self.my_piece else -1

    # check 3x3 square corners wins
    for row in range(3):
      for col in range(3):
        if state[row][col] != ' ' and state[row][col] == state[row + 2][col] == state[row][col + 2] == state[row + 2][col + 2]:
          return 1 if state[row][col] == self.my_piece else -1

    return 0

def main():
  print('Hello, this is Samaritan')
  ai = Teeko2Player()
  piece_count = 0
  turn = 0

  # drop phase
  while piece_count < 8 and ai.game_value(ai.board) == 0:

    # get the player or AI's move
    if ai.my_piece == ai.pieces[turn]:
      ai.print_board()
      move = ai.make_move(ai.board)
      ai.place_piece(move, ai.my_piece)
      print(ai.my_piece + " moved at " + chr(move[0][1] + ord("A")) + str(move[0][0]))
    else:
      move_made = False
      ai.print_board()
      print(ai.opp + "'s turn")
      while not move_made:
        player_move = input("Move (e.g. B3): ")
        while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
          player_move = input("Move (e.g. B3): ")
        try:
          ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
          move_made = True
        except Exception as e:
          print(e)

    # update the game variables
    piece_count += 1
    turn += 1
    turn %= 2

  # move phase - can't have a winner until all 8 pieces are on the board
  while ai.game_value(ai.board) == 0:

    # get the player or AI's move
    if ai.my_piece == ai.pieces[turn]:
      ai.print_board()
      move = ai.make_move(ai.board)
      ai.place_piece(move, ai.my_piece)
      print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
      print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
    else:
      move_made = False
      ai.print_board()
      print(ai.opp+"'s turn")
      while not move_made:
        move_from = input("Move from (e.g. B3): ")
        while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
          move_from = input("Move from (e.g. B3): ")
        move_to = input("Move to (e.g. B3): ")
        while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
          move_to = input("Move to (e.g. B3): ")
        try:
          ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                            (int(move_from[1]), ord(move_from[0])-ord("A"))])
          move_made = True
        except Exception as e:
          print(e)

    # update the game variables
    turn += 1
    turn %= 2

  ai.print_board()
  if ai.game_value(ai.board) == 1:
    print("AI wins! Game over.")
  else:
    print("You win! Game over.")

if __name__ == "__main__":
  main()
