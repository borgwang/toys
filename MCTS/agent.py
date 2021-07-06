from __future__ import division

import time
import random
from math import log, sqrt



class Node(object):
    """Class of tree node in Monte-Carlo tree search"""
    def __init__(self, score=0, visits=0):
        self.score = score
        self.visits = visits


class UCT(object):
    def __init__(self, board, **kwargs):
        self.board = board
        self.history = []
        self.nodes = {}

        self.time_limit = float(kwargs.get('time_limit', 30))
        self.max_moves_per_sim = int(kwargs.get('max_moves_per_sim', 100))
        # C is a hyperparameter that controls explore & exploit trade-off when
        # picking action using UCB.
        self.C = kwargs.get('C', 1.4)

        self.end_values = board.end_score

    def update(self, state):
        self.history.append(state)

    def get_action(self):
        # calculate the best move from the current game state and return it.
        self.max_depth = 0
        self.data = {}
        self.nodes.clear()

        state = self.history[-1]
        player = self.board.current_player(state)
        legal = self.board.legal_actions(self.history[:])

        if not legal:
            return
        if len(legal) == 1:
            return legal[0]

        num_sim = 0
        begin = time.time()
        print 'thinking...'
        while time.time() - begin < self.time_limit:
            self.run_simulation()
            num_sim += 1

        self.data.update(games=num_sim, max_depth=self.max_depth,
                        time=time.time()-begin)
        print 'simulated: %d games, time cost: %.4f' % (self.data['games'], self.data['time'])
        print 'Max depth searched: ', self.max_depth

        # store and display the stats for each possible action
        self.data['trajectory'] = self.get_action_values(state, player, legal)
        for m in self.data['trajectory']:
            print '{action}: {percent:.2f}% ({wins} / {plays})'.format(**m)
        # pick up move with the highest winning percentage
        return self.data['trajectory'][0]['action']

    def run_simulation(self):
        # Plays out a "random" game from the current position,
        # then updates the statistics tables with the result.
        nodes = self.nodes
        visit_path = set()
        history_copy = self.history[:]
        state = history_copy[-1]
        player = self.board.current_player(state)

        expand = True
        for t in range(1, self.max_moves_per_sim + 1):
            legal = self.board.legal_actions(history_copy)
            actions_states = [(a, self.board.next_state(state, a)) for a in legal]

            if all((player, S) in nodes for a, S in actions_states):
                log_total = log(
                    sum(nodes[(player, S)].visits for a, S in actions_states)or 1)
                # UCB core formula (first part: exploit   second part: explore)
                score, action, state = max(
                    ((nodes[(player, S)].score / nodes[(player, S)].visits or 1) +
                     self.C * sqrt(log_total / nodes[(player, S)].visits or 1), a, S)
                    for a, S in actions_states
                )
            else:
                action, state = random.choice(actions_states)
            history_copy.append(state)

            # player here refers to the player who moves into that particular state
            if expand and (player, state) not in nodes:
                expand = False
                nodes[(player, state)] = Node()
                if t > self.max_depth:
                    self.max_depth = t
            visit_path.add((player, state))
            # switch player (in simulation)
            player = self.board.current_player(state)

            if self.board.is_ended(history_copy):
                break

        # Back-propagate
        end_values = self.end_values(history_copy)
        for player, state in visit_path:
            if (player, state) not in nodes:
                continue
            S = nodes[(player, state)]
            S.visits += 1
            S.score += end_values[player]

    def get_action_values(self, state, player, legal):
        actions_states = ((a, self.board.next_state(state, a)) for a in legal)
        return sorted(
            ({'action': a,
              'percent': 100 * self.nodes[(player, S)].score / self.nodes[(player, S)].visits,
              'wins': self.nodes[(player, S)].score,
              'plays': self.nodes[(player, S)].visits}
            for a, S in actions_states),
            key=lambda x: (x['percent'], x['plays']),
            reverse=True
        )


class Human(object):
    def __init__(self, board, **kwargs):
        self.board = board
        self.player = None
        self.history = []

    def update(self, state):
        self.history.append(state)

    def get_action(self):
        while True:
            notation = raw_input("Please enter your action: ")
            try:
                R, C, r, c = map(int, notation.split())
            except Exception:
                print 'illegal input!!!'
                continue
            action = (R, C, r, c)

            if not self.board.is_legal(self.history, action):
                print 'illegal action!!!'
            else:
                break
        return action


class Random(object):

    def __init__(self, board, **kwargs):
        self.board = board
        self.player = None
        self.history = []

    def update(self, state):
        self.history.append(state)

    def get_action(self):
        return random.choice(self.board.legal_actions(self.history))
