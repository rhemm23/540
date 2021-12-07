import random

class Teeko2Player:
  board = [[' ' for _ in range(5)] for _ in range(5)]
  pieces = ['b', 'r']

  def __init__(self):
    self.my_piece = random.choice(self.pieces)
    self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

  def max_value(self, state, piece, depth):
    succs = self.succ(state, piece)
    if depth == 2:
      

  def make_move(self, state):
    """ Selects a (row, col) space for the next move. You may assume that whenever
    this function is called, it is this player's turn to move.

    Args:
        state (list of lists): should be the current state of the game as saved in
            this Teeko2Player object. Note that this is NOT assumed to be a copy of
            the game state and should NOT be modified within this method (use
            place_piece() instead). Any modifications (e.g. to generate successors)
            should be done on a deep copy of the state.

            In the "drop phase", the state will contain less than 8 elements which
            are not ' ' (a single space character).

    Return:
        move (list): a list of move tuples such that its format is
                [(row, col), (source_row, source_col)]
            where the (row, col) tuple is the location to place a piece and the
            optional (source_row, source_col) tuple contains the location of the
            piece the AI plans to relocate (for moves after the drop phase). In
            the drop phase, this list should contain ONLY THE FIRST tuple.

    Note that without drop phase behavior, the AI will just keep placing new markers
        and will eventually take over the board. This is not a valid strategy and
        will earn you no points.
    """

    drop_phase = True   # TODO: detect drop phase

    if not drop_phase:
      # TODO: choose a piece to move and remove it from the board
      # (You may move this condition anywhere, just be sure to handle it)
      #
      # Until this part is implemented and the move list is updated
      # accordingly, the AI will not follow the rules after the drop phase!
      pass

    # select an unoccupied space randomly
    # TODO: implement a minimax algorithm to play better
    move = []
    (row, col) = (random.randint(0,4), random.randint(0,4))
    while not state[row][col] == ' ':
      (row, col) = (random.randint(0,4), random.randint(0,4))

    # ensure the destination (row,col) tuple is at the beginning of the move list
    move.insert(0, (row, col))
    return move

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
    """ Validates the opponent's next move against the internal board representation.
    You don't need to touch this code.

    Args:
        move (list): a list of move tuples such that its format is
                [(row, col), (source_row, source_col)]
            where the (row, col) tuple is the location to place a piece and the
            optional (source_row, source_col) tuple contains the location of the
            piece the AI plans to relocate (for moves after the drop phase). In
            the drop phase, this list should contain ONLY THE FIRST tuple.
    """
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
  ai = Teeko2Player()
  state = [['r', 'b', 'r', 'b', 'r'],['r', 'b', 'b', ' ', ' '],[' ',' ',' ',' ',' '],[' ',' ',' ',' ',' '],[' ',' ',' ',' ',' ']]
  succs = ai.succ(state, ai.my_piece)
  for succ in succs:
    print('\n')
    for row in succ:
      print(row)
  exit()
  main()
