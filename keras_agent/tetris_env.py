"""
Tetris Enviorment
Author: Nathan Delcampo
Date: 4/8/2026
Last Modified: 4/19/2026
Python Version: 3.11.14

DESC: Simulates every valid placement and returns the 
    resulting board features for agent.

"""

import numpy as np
from tetris_keras import Tetris


#  Board feature extraction
def _col_heights(board: np.ndarray) -> np.ndarray:
    rows, cols = board.shape
    heights = np.zeros(cols, dtype=np.float32)

    for col in range(cols):
        for row in range(rows):
            if board[row, col] == 1:
                heights[col] = rows - row
                break
    return heights

def _count_holes(board: np.ndarray) -> int:
    holes = 0
    
    for col in range(board.shape[1]):
        block_found = False
        for row in range(board.shape[0]):
            if board[row, col] == 1:
                block_found = True
            elif block_found:
                holes += 1
    return holes

def _bumpiness(heights: np.ndarray) -> float:
    return float(np.abs(np.diff(heights)).sum())

def _height_variance(heights: np.ndarray) -> float:
    return float(heights.var())

def extract_features(board: np.ndarray, lines_cleared: int = 0) -> np.ndarray:
    """
    Extract a feature vector from a board state.

    Features:
        - per-column heights
        - aggregate height
        - holes count
        - bumpiness
        - height variance
        - lines cleared this step
    """
    rows, cols = board.shape
    heights    = _col_heights(board)
    agg_h      = heights.sum()
    holes      = _count_holes(board)
    bump       = _bumpiness(heights)
    variance   = _height_variance(heights)

    features = np.concatenate([
        heights      / rows,       
        [agg_h       / (rows * cols),
         holes        / (rows * cols),
         bump         / (rows * cols),
         variance     / (rows ** 2),
         float(lines_cleared)],
    ]).astype(np.float32)

    return features 

#  TimeStep
class TimeStep:
    FIRST = 0; MID = 1; LAST = 2

    def __init__(self, step_type, observation, reward, discount):
        self.step_type   = step_type
        self.observation = observation
        self.reward      = np.float32(reward)
        self.discount    = np.float32(discount)

    def is_first(self): return self.step_type == self.FIRST
    def is_last(self):  return self.step_type == self.LAST

    @staticmethod
    def restart(obs):                   return TimeStep(TimeStep.FIRST, obs, 0.0, 1.0)
    @staticmethod
    def transition(obs, reward, discount=0.99): return TimeStep(TimeStep.MID, obs, reward, discount)
    @staticmethod
    def termination(obs, reward):       return TimeStep(TimeStep.LAST, obs, reward, 0.0)


#  Board simulator
def _simulate_placement(board: np.ndarray, piece_coords: list, col: int, rotation_angle: int, rows: int, cols: int):
    """
    Simulate dropping a piece at each position on a copy of the board.
    Returns (resulting_board, lines_cleared) or None if the placement is invalid.
    """
    piece = piece_coords[rotation_angle]

    max_x = max(x for x, y in piece)
    min_x = min(x for x, y in piece)

    # Out of bounds
    if col + min_x < 0 or col + max_x >= cols:
        return None

    # Hard drop simulator
    drop_y = 0
    while True:
        colliding = False
        for x, y in piece:
            nx, ny = x + col, y + drop_y + 1
            if ny >= rows or (nx < cols and board[ny, nx] == 1):
                colliding = True
                break
        if colliding:
            break
        drop_y += 1

    # Check spawn collision 
    for x, y in piece:
        nx, ny = x + col, y + drop_y
        if ny < 0 or ny >= rows or nx < 0 or nx >= cols:
            return None
        if board[ny, nx] == 1:
            return None

    # Place piece on a copy
    new_board = board.copy()
    for x, y in piece:
        new_board[y + drop_y, x + col] = 1

    # Clear full lines
    full_rows  = [r for r in range(rows) if new_board[r].sum() == cols]
    lines = len(full_rows)

    if lines:
        new_board = np.delete(new_board, full_rows, axis=0)
        new_board = np.concatenate([np.zeros((lines, cols), dtype=new_board.dtype), new_board])

    return new_board, lines


#  TetrisEnv
class TetrisEnv:
    """
    State/Action Tetris environment for Keras.

    Observation
    -----------------
        Feature Vector from board using extract_features

    Reward
    ------
        Reward is limited as most of data is fed into value network
            - Lines cleared: positive (exponential scaling)
            - Death: negatvie

    Episode termination
    -------------------
    The episode ends when the Tetris game is over.
    """

    # Reward Weights
    w_lines = 10.0
    w_death = 5.0

    def __init__(self, rows: int = 14, cols: int = 8, render: bool = False):
        self._rows   = rows
        self._cols   = cols
        self._render = render

        self._curriculum_level = 0
        self._episode_ended    = False

        # feature_size = cols heights + 5 scalars
        self.feature_size = cols + 5
        self.action_size  = cols * 4   # kept for compatibility, not used by agent

        self._game = Tetris(_rows=rows, _cols=cols, _render=render)

    # ── Core new method ───────────────────────────────────────────

    def get_next_states(self) -> dict:
        """
        Simulate every valid (col, rotation) placement for the current piece.

        Returns: 
            results dict: [(col, rotation), (features, lines_cleared)]

        Only valid placements are included.
        """
        state = {}
        piece_id   = self._game.current_piece
        piece_data = Tetris.tetris_pieces[piece_id]
        board      = self._game.board

        # remove duplicate placements
        seen_boards = set() 

        for rotation, angle in enumerate([0, 90, 180, 270]):
            for col in range(self._cols):

                outcome = _simulate_placement(
                    board, piece_data, col, angle,
                    self._rows, self._cols
                )

                if outcome is None:
                    continue

                new_board, lines = outcome

                # Skip duplicate resulting boards
                board_key = new_board.tobytes()
                if board_key in seen_boards:
                    continue
                seen_boards.add(board_key)

                features = extract_features(new_board, lines_cleared=lines)
                state[(col, rotation)] = (features, lines)

        return state

    # API
    def _reset(self) -> TimeStep:
        self._game.reset()
        
        if self._curriculum_level > 0:
            self._game.prefill_board(self._curriculum_level)
        self._episode_ended = False

        # Return features of the empty board as the initial observation
        obs = extract_features(self._game.board)
        return TimeStep.restart(obs)

    def _step(self, action: tuple) -> TimeStep:
        """
        action: (col, rotation) 
        """
        if self._episode_ended:
            return self._reset()

        col, rotation = action
        score_before = self._game.score

        # Do action and record data
        self._game.step_col(col, rotation)
        
        score_after = self._game.score
        lines_this_step = max(0, score_after - score_before)
        game_over = not self._game.is_alive() or self._game.pieces_placed > 10_000
        board_now = self._game.board
        reward = (self.w_lines * (lines_this_step ** 2))

        # Extact features of board
        obs = extract_features(board_now, lines_cleared=lines_this_step)

        if game_over:
            reward -= self.w_death
            print(f"Pieces Placed: {self._game.get_pieces_placed()}")
            self._episode_ended = True
            return TimeStep.termination(obs, reward=float(reward))

        return TimeStep.transition(obs, reward=float(reward), discount=0.99)

    def render(self):
        self._game._render()

    def reduce_curriculum(self):
        if self._curriculum_level > 0:
            self._curriculum_level -= 1