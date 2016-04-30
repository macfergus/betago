import copy
import random
import time

import numpy as np

from .. import scoring
from ..dataloader.goboard import GoBoard, to_string

__all__ = [
    'MCTSBot',
]


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
        child = GameNode(new_board, _other(self.next_to_play), self.model, self.processor)
        child.last_move = move_to_add
        child.parent = self
        self.children.append(child)
        return child

    def is_leaf(self):
        return not self.children

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
        return random.choice(node.children)


def _other(player):
    return 'w' if player == 'b' else 'b'


def _game_over(board):
    status = scoring.evaluate_territory(board)
    return (status.num_black_stones + status.num_white_stones > 10) and status.num_dame == 0


def border_moves(board, next_to_play):
    border_points = []
    diagonal_points = []
    for row in range(19):
        for col in range(19):
            move = (row, col)
            if move not in board.board:
                my_neighbors = 0
                other_neighbors = 0
                my_diag, other_diag = 0, 0
                deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                for delta_r, delta_c in deltas:
                    next_r, next_c = row + delta_r, col + delta_c
                    if next_r < 0 or next_r >= board.board_size:
                        continue
                    if next_c < 0 or next_c >= board.board_size:
                        continue
                    neighbor = board.board.get((next_r, next_c))
                    if neighbor == next_to_play:
                        my_neighbors += 1
                    elif neighbor:
                        other_neighbors += 1
                deltas = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
                for delta_r, delta_c in deltas:
                    next_r, next_c = row + delta_r, col + delta_c
                    if next_r < 0 or next_r >= board.board_size:
                        continue
                    if next_c < 0 or next_c >= board.board_size:
                        continue
                    neighbor = board.board.get((next_r, next_c))
                    if neighbor == next_to_play:
                        my_diag += 1
                    elif neighbor:
                        other_diag += 1
                if my_neighbors and other_neighbors:
                    border_points.append(move)
                elif other_neighbors and my_diag:
                    diagonal_points.append(move)
    return border_points, diagonal_points


class MCTSBot(object):
    def __init__(self, playout_model, processor):
        self.go_board = GoBoard(19)
        self.playout_model = playout_model
        self.processor = processor
        self.num_planes = processor.num_planes
        self.komi = 7.5
        self.top_n = 10
        self.thinking_time = 20.0  # seconds
        self.policy = TreePolicy()

    def set_board(self, new_board):
        self.go_board = copy.deepcopy(new_board)

    def apply_move(self, color, move):
        self.go_board.apply_move(color, move)

    def select_move(self, bot_color):
        root = GameNode(self.go_board, bot_color, self.playout_model, self.processor)
        start = time.time()

        while time.time() - start < self.thinking_time:
            node = root
            # Find a node that can be expanded.
            while node.is_expanded():
                node = self.policy.select_child(node)

            # Expand this node, if possible.
            if not node.is_expanded():
                node = node.add_child()

            # Simulate a random game from this node.
            winner = self.do_random_playout(node.board, node.next_to_play)

            # Propagate scores back up the tree.
            while node is not None:
                node.record_win(winner)
                node = node.parent

        # Select the best move.
        ranked_children = sorted(
            root.children,
            key=lambda node: (node.win_percentage(bot_color), node.num_visits()))
        self.go_board.apply_move(bot_color, ranked_children[0].last_move)
        return ranked_children[0].last_move

    def do_random_playout(self, board, next_to_play):
        board = copy.deepcopy(board)
        passes = 0
        while not _game_over(board):
            X, _ = self.processor.feature_and_label(next_to_play, (0, 0), board, self.num_planes)
            X = X.reshape((1, X.shape[0], X.shape[1], X.shape[2]))
            pred = np.squeeze(self.playout_model.predict(X))
            pred *= pred
            top_idx = pred.argsort()[-self.top_n:]
            mask = np.zeros(pred.shape)
            mask[top_idx] = 1
            pred *= mask
            #if pred.sum() < 1e-6:
            #    print "No predictions!"
            #    print to_string(board)
            pred /= pred.sum()
            moves = np.random.choice(19 * 19, size=self.top_n, replace=False, p=pred)
            #s = ' '.join(['(%d, %d @ %.5f)' % (mov // 19, mov % 19, pred[mov]) for mov in moves])
            passes += 1
            did_move = False
            for i, move_number in enumerate(moves):
                row = move_number // 19
                col = move_number % 19
                move = (row, col)
                if board.is_move_legal(next_to_play, move):
                    board.apply_move(next_to_play, move)
                    did_move = True
                    passes = 0
                    break
            if not did_move:
                bp, diag = border_moves(board, next_to_play)
                random.shuffle(bp)
                for move in bp:
                    if board.is_move_legal(next_to_play, move):
                        board.apply_move(next_to_play, move)
                        did_move = True
                        break
                random.shuffle(diag)
                for move in diag:
                    if board.is_move_legal(next_to_play, move):
                        board.apply_move(next_to_play, move)
                        did_move = True
                        break
            next_to_play = _other(next_to_play)
        status = scoring.evaluate_territory(board)
        black_area = status.num_black_territory + status.num_black_stones
        white_area = status.num_white_territory + status.num_white_stones
        white_score = white_area + self.komi
        diff = black_area - white_score
        return 'w' if white_score > black_area else 'b'
