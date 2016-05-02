import copy
import math
import multiprocessing
import random
import time
import yaml

import numpy as np

from keras.models import model_from_yaml
from .. import scoring
from ..bots.playout import PlayoutBot
from ..dataloader.goboard import GoBoard, to_string
from ..processor import SevenPlaneProcessor

__all__ = [
    'MCTSBot',
]


def init_worker():
    processor = SevenPlaneProcessor()
    playout_bot_name = 'cum_a'
    playout_model_file = 'model_zoo/' + playout_bot_name + '_bot.yml'
    playout_weight_file = 'model_zoo/' + playout_bot_name + '_weights.hd5'
    with open(playout_model_file, 'r') as f:
        yml = yaml.load(f)
        playout_model = model_from_yaml(yaml.dump(yml))
        playout_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        playout_model.load_weights(playout_weight_file)
    global playout_bot
    playout_bot = PlayoutBot(playout_model, processor)


def path(node):
    path = []
    while node is not None:
        path.append(node.last_move)
        node = node.parent
    path.reverse()
    return path


class GameNode(object):
    def __init__(self, board, next_to_play, model, processor):
        self.last_move = None
        self.board = board
        self.next_to_play = next_to_play
        self.model = model
        self.processor = processor
        self.num_planes = processor.num_planes
        self.parent = None

        self._unvisited_moves = []
        X, _ = self.processor.feature_and_label(self.next_to_play, (0, 0), board, self.num_planes)
        X = X.reshape((1, X.shape[0], X.shape[1], X.shape[2]))
        pred = np.squeeze(self.model.predict(X))
        #top_idx = pred.argsort()[-self.top_n:]
        top_idx = pred.argsort()[::-1]
        for move_num in top_idx:
            move = (move_num // 19, move_num % 19)
            if board.is_move_legal(self.next_to_play, move):
                self._unvisited_moves.append(move)
            if pred[move_num] < 0.01:
                break
        self.children = []
        self.stats = {'w': 0, 'b': 0}

    def is_expanded(self):
        return not self._unvisited_moves

    def add_child(self):
        move_to_add = self._unvisited_moves.pop(0)
        new_board = copy.deepcopy(self.board)
        new_board.apply_move(self.next_to_play, move_to_add)
        child = GameNode(
            new_board, new_board.other_color(self.next_to_play),
            self.model, self.processor)
        child.last_move = move_to_add
        child.parent = self
        self.children.append(child)
        return child

    def record_win(self, player):
        self.stats[player] += 1

    def win_percentage(self, player):
        return float(self.stats[player]) / sum(self.stats.values())

    def num_visits(self):
        return sum(self.stats.values())

    def __str__(self):
        return ("Next to play: %s" % self.next_to_play +
            "Position:\n" + to_string(self.board))


class TreePolicy(object):
    def select_child(self, node):
        """Given a child, select the node to visit next.

        Parameters
        ----------
        node : GameNode

        Returns
        -------
        GameNode
        """
        total_visits = sum(child.num_visits() for child in node.children)
        log_visits = math.log(total_visits)
        temperature = 1.4
        def ucb1(child):
            expectation = child.win_percentage(node.next_to_play)
            exploration = math.sqrt(log_visits / node.num_visits())
            return expectation + 1.4 * temperature
        scored_children = [(ucb1(child), child) for child in node.children]
        scored_children.sort()
        selected_child = scored_children[-1][1]
        #print "Select", path(selected_child), "win", selected_child.win_percentage(node.next_to_play), "metric", scored_children[-1][0]
        return scored_children[-1][1]


def _game_over(board):
    status = scoring.evaluate_territory(board)
    return (status.num_black_stones + status.num_white_stones > 10) and status.num_dame == 0


class MCTSBot(object):
    def __init__(self, model, processor, playout_bot):
        self.go_board = GoBoard(19)
        self.model = model
        self.processor = processor
        self.playout_bot = playout_bot
        self.processor = processor
        self.num_planes = processor.num_planes
        self.komi = 7.5
        self.top_n = 10
        self.thinking_time = 90.0  # seconds
        self.policy = TreePolicy()
        self.root = None
        self.num_workers = 8
        self.pool = multiprocessing.Pool(self.num_workers, initializer=init_worker)
        self.last_move = None

    def set_board(self, new_board):
        self.go_board = copy.deepcopy(new_board)
        self.root = None

    def apply_move(self, color, move):
        self.go_board.apply_move(color, move)
        # If this was a move we considered, preserve the tree.
        found_move = False
        for child in self.root.children:
            if child.last_move == move:
                self.root = child
                self.root.parent = None
                found_move = True
                #print "Found it in our tree"
                break
        if not found_move:
            self.root = None

    def select_move(self, bot_color):
        if self.root is None:
            self.root = GameNode(self.go_board, bot_color, self.model, self.processor)
        start = time.time()

        while time.time() - start < self.thinking_time:
            node = self.root
            # Find a node that can be expanded.
            while node.is_expanded():
                node = self.policy.select_child(node)

            # Expand this node, if possible.
            if not node.is_expanded():
                node = node.add_child()
            #print "Selected %s" % path(node)

            # Simulate a random game from this node.
            #print "Simulate random game from:"
            #print to_string(node.board) + "\n"
            #print "next to play", node.next_to_play
            args = [(node.board, node.next_to_play, self.komi) for _ in range(self.num_workers)]
            winners = self.pool.map(unpack_args_do_random_playout, args)
            #winner = do_random_playout(self.playout_bot, node.board, node.next_to_play, self.komi)
            #print "Winners", winners

            # Propagate scores back up the tree.
            while node is not None:
                for winner in winners:
                    node.record_win(winner)
                node = node.parent

        # Select the best move.
        ranked_children = sorted(
            self.root.children,
            key=lambda node: (node.num_visits(), node.win_percentage(bot_color)))
        self.go_board.apply_move(bot_color, ranked_children[0].last_move)
        # Make the selected child the new root.
        self.root = ranked_children[0]
        self.root.parent = None
        return self.root.last_move


def do_random_playout(playout_bot, board, next_to_play, komi):
    board = copy.deepcopy(board)
    passes = 0
    while passes < 2:
        move = playout_bot.select_move(board, next_to_play)
        if move is not None:
            board.apply_move(next_to_play, move)
            passes = 0
        else:
            passes += 1
        next_to_play = board.other_color(next_to_play)
    status = scoring.evaluate_territory(board)
    black_area = status.num_black_territory + status.num_black_stones
    white_area = status.num_white_territory + status.num_white_stones
    white_score = white_area + komi
    return 'w' if white_score > black_area else 'b'


def unpack_args_do_random_playout(args):
    global playout_bot
    return do_random_playout(playout_bot, *args)
