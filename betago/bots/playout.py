import copy
import random

import numpy as np


__all__ = [
    'PlayoutBot',
]


class PlayoutBot(object):
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.num_planes = processor.num_planes
        self.top_n = 10

    def select_move(self, board, color):
        """
        Will not mutate board.
        """
        return get_first_valid_move(board, color, self._generate_moves(board, color))

    def _generate_moves(self, board, color):
        for move in self._model_moves(board, color):
            yield move
        # If the model is out of moves, play randomly along the borders
        # in order to end the game.
        border_moves, diagonal_moves = self._border_moves(board, color)
        for move in generate_in_random_order(border_moves):
            yield move
        for move in generate_in_random_order(diagonal_moves):
            yield move

    def _model_moves(self, board, color):
        X, label = self.processor.feature_and_label(
            color, (0, 0), board, self.num_planes)
        X = X.reshape((1, X.shape[0], X.shape[1], X.shape[2]))
        pred = np.squeeze(self.model.predict(X))
        top_idx = pred.argsort()[-self.top_n:]
        mask = np.zeros(pred.shape)
        mask[top_idx] = 1
        pred *= mask
        num_moves = min(self.top_n, np.count_nonzero(pred))
        if num_moves == 0:
            return
        pred /= pred.sum()
        try:
            moves = np.random.choice(19 * 19, size=num_moves, replace=False, p=pred)
        except ValueError:
            print "huh?"
            print pred
            print num_moves
            raise
        for move_number in moves:
            row = move_number // 19
            col = move_number % 19
            yield (row, col)

    def _border_moves(self, board, color):
        border_moves, diagonal_moves = [], []
        other_color = board.other_color(color)
        sides = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        angles = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
        for row, col in board.empty_points():
            my_neighbors = 0
            enemy_neighbors = 0
            my_diag = 0
            for delta_r, delta_c in sides:
                neighbor_r, neighbor_c = row + delta_r, col + delta_c
                neighbor = board.board.get((neighbor_r, neighbor_c))
                if neighbor == color:
                    my_neighbors += 1
                elif neighbor == other_color:
                    enemy_neighbors += 1
            for delta_r, delta_c in angles:
                neighbor_r, neighbor_c = row + delta_r, col + delta_c
                neighbor = board.board.get((neighbor_r, neighbor_c))
                if neighbor == color:
                    my_diag += 1
            if my_neighbors and enemy_neighbors:
                border_moves.append((row, col))
            elif enemy_neighbors and my_diag:
                diagonal_moves.append((row, col))
        return border_moves, diagonal_moves


def get_first_valid_move(board, color, move_generator):
    for move in move_generator:
        if move is None or board.is_move_legal(color, move):
            return move
    return None


def generate_in_random_order(point_list):
    """Yield all points in the list in a random order."""
    point_list = copy.copy(point_list)
    random.shuffle(point_list)
    for candidate in point_list:
        yield candidate
