"""ref: https://github.com/jbradberry/ultimate_tictactoe/tree/master"""
class Board:

  def __init__(self):
    pos = dict(((r, c), 1<<(3*r + c)) for r in range(3) for c in range(3))
    wins = [pos[(r, 0)] | pos[(r, 1)] | pos[(r, 2)] for r in range(3) ] + \
        [pos[(0, c)] | pos[(1, c)] | pos[(2, c)] for c in range(3)] + \
        [pos[(0, 0)] | pos[(1, 1)] | pos[(2, 2)], pos[(0, 2)] | pos[(1, 1)] | pos[(2, 0)]]
    self.positions = pos
    self.wins = wins

  def starting_state(self):
    # Each of the 9 pairs of player 1 and player 2 board bitmasks
    # plus the win/tie state of the big board for p1 and p2 plus
    # the row and column of the required board for the next action
    # and finally the player number to move.
    return (0, 0) * 10 + (None, None, 1)

  def display(self, state, action="", _unicode=True):
    actions = dict(
      ((R, C, r, c), p)
      for R in range(3)
      for C in range(3)
      for r in range(3)
      for c in range(3)
      for i, p in enumerate("XO")
      if state[2*(3*R + C) + i] & self.positions[(r, c)]
    )

    player = state[-1]

    sub = "\u2564".join("\u2550" for x in range(3))
    top = "\u2554" + "\u2566".join(sub for x in range(3)) + "\u2557\n"

    sub = "\u256a".join("\u2550" for x in range(3))
    div = "\u2560" + "\u256c".join(sub for x in range(3)) + "\u2563\n"

    sub = "\u253c".join("\u2500" for x in range(3))
    sep = "\u255f" + "\u256b".join(sub for x in range(3)) + "\u2562\n"

    sub = "\u2567".join("\u2550" for x in range(3))
    bot = "\u255a" + "\u2569".join(sub for x in range(3)) + "\u255d\n"
    if action:
      bot += "Last played: {0}\n\n".format(action)
    bot += "Player{0}'s turn".format(player)

    print(
      top +
      div.join(
        sep.join(
          "\u2551" +
          "\u2551".join(
            "\u2502".join(
              actions.get((R, C, r, c), " ") for c in range(3)
            )
            for C in range(3)
          ) +
          "\u2551\n"
          for r in range(3)
        )
        for R in range(3)
      ) +
      bot
    )

  def next_state(self, state, action):
    R, C, r, c = action
    player = state[-1]
    board_index = 2*(3*R + C)
    player_index = player - 1

    state = list(state)
    state[-1] = 3 - player
    state[board_index + player_index] |= self.positions[(r, c)]
    updated_board = state[board_index + player_index]

    full = (state[board_index] | state[board_index+1] == 0x1ff)
    if any(updated_board & w == w for w in self.wins):
      state[18 + player_index] |= self.positions[(R, C)]
    elif full:
      state[18] |= self.positions[(R, C)]
      state[19] |= self.positions[(R, C)]

    if (state[18] | state[19]) & self.positions[(r, c)]:
      state[20], state[21] = None, None
    else:
      state[20], state[21] = r, c
    return tuple(state)

  def is_legal(self, history, action):
    state = history[-1]
    R, C, r, c = action

    # Is action out of bounds?
    if (R, C) not in self.positions:
      return False
    if (r, c) not in self.positions:
      return False

    board_index = 2*(3*R + C)

    # Is the square within the sub-board already taken?
    occupied = state[board_index] | state[board_index+1]
    if self.positions[(r, c)] & occupied:
      return False

    # Is our action unconstrained by the previous action?
    if state[20] is None:
      return True

    # Otherwise, we must play in the proper sub-board (determined by previous action of the opponent)
    print(R, C, state[20], state[21])
    return (R, C) == (state[20], state[21])

  def legal_actions(self, history):
    state = history[-1]
    R, C = state[20], state[21]
    Rset, Cset = (R,), (C,)
    if R is None:
      Rset, Cset = range(3), range(3)

    occupied = [state[2*x] | state[2*x+1] for x in range(9)]
    finished = state[18] | state[19]

    actions = [
      (R, C, r, c)
      for R in Rset
      for C in Cset
      for r in range(3)
      for c in range(3)
      if not occupied[3*R+C] & self.positions[(r, c)]
      and not finished & self.positions[(R, C)]
    ]
    return actions

  def current_player(self, state):
    return state[-1]

  def is_ended(self, history):
    state = history[-1]
    p1 = state[18] & ~state[19]
    p2 = state[19] & ~state[18]

    if any(w & p1 == w for w in self.wins):
      return True
    if any(w & p2 == w for w in self.wins):
      return True
    if state[18] | state[19] == 0x1ff:
      return True
    return False

  def end_score(self, history):
    if not self.is_ended(history):
      return

    state = history[-1]
    p1 = state[18] & ~state[19]
    p2 = state[19] & ~state[18]

    if any(w & p1 == w for w in self.wins):
      return {1: 1, 2: 0}
    if any(w & p2 == w for w in self.wins):
      return {1: 0, 2: 1}
    if state[18] | state[19] == 0x1ff:
      return {1: 0.5, 2: 0.5}

  def result(self, winners):
    winners = sorted((v, k) for k, v in winners.items())
    value, winner = winners[-1]
    if value == 0.5:
      return "Draw."
    return "Winner: Player {0}.".format(winner)
