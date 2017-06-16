import argparse
from itertools import cycle

from board import Board
from agent import UCT, Human

class Game(object):
    agent_map = {'human': Human, 'ai': UCT}

    def __init__(self, board, args):
        self.history = []
        self.board = board
        try:
            agent1 = self.agent_map[args.p1](board, time_limit=args.time_limit)
            agent2 = self.agent_map[args.p2](board, time_limit=args.time_limit)
        except Exception as e:
            print 'illegal player!!!  (human | ai) is allowed.'
            raise
        self.agents = [agent1, agent2]
        self.turns = cycle(self.agents)  # take turns to move

    def _reset(self):
        self.history = []
        state = self.board.starting_state()
        self.history.append(state)

        for a in self.agents:
            a.update(state)

        self.board.display(state)

    def run(self):
        self._reset()

        for agent in self.turns:
            # play
            action = agent.get_action()
            state = self.board.next_state(self.history[-1], action)
            # update
            self.history.append(state)
            if self.board.is_ended(self.history):
                winners = self.board.end_score(self.history)
                print 'Game Over!!!'
                print self.board.result(winners)
                break
            for a in self.agents:
                a.update(state)
            # display
            self.board.display(state, action)

def args_parse():
        parser = argparse.ArgumentParser()
        parser.add_argument('--p1', default='human', help=
                'player1 (*human | ai)')
        parser.add_argument('--p2', default='ai', help=
                'player2 (human | *ai)')
        parser.add_argument('--time_limit', default=30, help=
                'time_limit for ai to think')
        return parser.parse_args()


if __name__ == '__main__':
    args = args_parse()
    game = Game(Board(), args)
    game.run()
