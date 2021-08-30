class Board(object):
    num_players = 2

    positions = dict(
        ((r, c), 1<<(3*r + c))
        for r in xrange(3)
        for c in xrange(3)
    )

    inv_positions = dict(
        (v, P) for P, v in positions.iteritems()
    )

    wins = [
        positions[(r, 0)] | positions[(r, 1)] | positions[(r, 2)]
        for r in xrange(3)
    ] + [
        positions[(0, c)] | positions[(1, c)] | positions[(2, c)]
        for c in xrange(3)
    ] + [
        positions[(0, 0)] | positions[(1, 1)] | positions[(2, 2)],
        positions[(0, 2)] | positions[(1, 1)] | positions[(2, 0)],
    ]

    def starting_state(self):
        # Each of the 9 pairs of player 1 and player 2 board bitmasks
        # plus the win/tie state of the big board for p1 and p2 plus
        # the row and column of the required board for the next action
        # and finally the player number to move.
        return (0, 0) * 10 + (None, None, 1)

    def display(self, state, action='', _unicode=True):
        actions = dict(
            ((R, C, r, c), p)
            for R in xrange(3)
            for C in xrange(3)
            for r in xrange(3)
            for c in xrange(3)
            for i, p in enumerate('XO')
            if state[2*(3*R + C) + i] & self.positions[(r, c)]
        )

        player = state[-1]

        sub = u"\u2564".join(u"\u2550" for x in xrange(3))
        top = u"\u2554" + u"\u2566".join(sub for x in xrange(3)) + u"\u2557\n"

        sub = u"\u256a".join(u"\u2550" for x in xrange(3))
        div = u"\u2560" + u"\u256c".join(sub for x in xrange(3)) + u"\u2563\n"

        sub = u"\u253c".join(u"\u2500" for x in xrange(3))
        sep = u"\u255f" + u"\u256b".join(sub for x in xrange(3)) + u"\u2562\n"

        sub = u"\u2567".join(u"\u2550" for x in xrange(3))
        bot = u"\u255a" + u"\u2569".join(sub for x in xrange(3)) + u"\u255d\n"
        if action:
            bot += u"Last played: {0}\n\n".format(action)
        bot += u"Player{0}'s turn".format(player)

        print (
            top +
            div.join(
                sep.join(
                    u"\u2551" +
                    u"\u2551".join(
                        u"\u2502".join(
                            actions.get((R, C, r, c), " ") for c in xrange(3)
                        )
                        for C in xrange(3)
                    ) +
                    u"\u2551\n"
                    for r in xrange(3)
                )
                for R in xrange(3)
            ) +
            bot
        ).encode('utf-8')

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

        player = state[-1]
        board_index = 2*(3*R + C)
        player_index = player - 1

        # Is the square within the sub-board already taken?
        occupied = state[board_index] | state[board_index+1]
        if self.positions[(r, c)] & occupied:
            return False

        # Is our action unconstrained by the previous action?
        if state[20] is None:
            return True

        # Otherwise, we must play in the proper sub-board.
        print R, C, state[20], state[21]
        return (R, C) == (state[20], state[21])

    def legal_actions(self, history):
        state = history[-1]
        R, C = state[20], state[21]
        Rset, Cset = (R,), (C,)
        if R is None:
            Rset, Cset = range(3), range(3)

        occupied = [
            state[2*x] | state[2*x+1] for x in xrange(9)
        ]
        finished = state[18] | state[19]

        actions = [
            (R, C, r, c)
            for R in Rset
            for C in Cset
            for r in xrange(3)
            for c in xrange(3)
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
        winners = sorted((v, k) for k, v in winners.iteritems())
        value, winner = winners[-1]
        if value == 0.5:
            return "Draw."
        return "Winner: Player {0}.".format(winner)
