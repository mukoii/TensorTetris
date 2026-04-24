"""
Step Tetris Implementation:
Author: Nathan Delcampo
Date: 1/13/2026
Last Modified: 4/17/2026
Python Version: 3.11.14

DESC: Tetris implementation with Numpy Array.
    Works off a steps with one action per step.
        - Step: move piece one tile, rotate or hard drop
        - Step_col: Choose a column and rotation to hard drop the piece in 

"""

import random
import numpy as np
from enum import Enum


class Tetris:

    # Contains local coordinates of all tetris pieces at any given rotation
    tetris_pieces = {
        0: { # I
            0:   [(0,0), (1,0), (2,0), (3,0)],
            90:  [(1,0), (1,1), (1,2), (1,3)],
            180: [(3,0), (2,0), (1,0), (0,0)],
            270: [(1,3), (1,2), (1,1), (1,0)],
        },
        1: { # T
            0:   [(1,0), (0,1), (1,1), (2,1)],
            90:  [(0,1), (1,2), (1,1), (1,0)],
            180: [(1,2), (2,1), (1,1), (0,1)],
            270: [(2,1), (1,0), (1,1), (1,2)],
        },
        2: { # L
            0:   [(1,0), (1,1), (1,2), (2,2)],
            90:  [(0,1), (1,1), (2,1), (2,0)],
            180: [(1,2), (1,1), (1,0), (0,0)],
            270: [(2,1), (1,1), (0,1), (0,2)],
        },
        3: { # J
            0:   [(1,0), (1,1), (1,2), (0,2)],
            90:  [(0,1), (1,1), (2,1), (2,2)],
            180: [(1,2), (1,1), (1,0), (2,0)],
            270: [(2,1), (1,1), (0,1), (0,0)],
        },
        4: { # Z
            0:   [(0,0), (1,0), (1,1), (2,1)],
            90:  [(0,2), (0,1), (1,1), (1,0)],
            180: [(2,1), (1,1), (1,0), (0,0)],
            270: [(1,0), (1,1), (0,1), (0,2)],
        },
        5: { # S
            0:   [(2,0), (1,0), (1,1), (0,1)],
            90:  [(0,0), (0,1), (1,1), (1,2)],
            180: [(0,1), (1,1), (1,0), (2,0)],
            270: [(1,2), (1,1), (0,1), (0,0)],
        },
        6: { # Square
            0:   [(1,0), (2,0), (1,1), (2,1)],
            90:  [(1,0), (2,0), (1,1), (2,1)],
            180: [(1,0), (2,0), (1,1), (2,1)],
            270: [(1,0), (2,0), (1,1), (2,1)],
        },
    }

    class Action(Enum):
        LEFT             = 1
        RIGHT            = 2
        COUNTERCLOCKWISE = 3
        CLOCKWISE        = 4
        NOOP             = 5
        HARD_DROP        = 6

    def __init__(self, _rows=14, _cols=6, _render=False):
        self.config = {
            "render": _render,
            "rows":   _rows,
            "cols":   _cols,
        }
        self.reset()

    def reset(self):
        self.cols          = self.config["cols"]
        self.rows          = self.config["rows"]
        self.board         = np.zeros((self.rows, self.cols))
        self.score         = 0
        self.current_piece = random.randint(0, 6)
        self.next_piece    = random.randint(0, 6)
        self.pos           = [1, 1]
        self.rotation      = 0
        self.steps         = 0
        self.pieces_placed = 0
        self.alive         = True
        return self.board

    # Helper functions
    def _clear_lines(self):
        lines_full = [i for i, row in enumerate(self.board) if 0 not in row]
        self.board = np.delete(self.board, lines_full, axis=0)
        new_lines  = np.zeros((len(lines_full), self.cols), dtype=int)
        self.board = np.concatenate((new_lines, self.board), axis=0)

        if len(lines_full) == 4:
            self.score += 8
        else:
            self.score += len(lines_full)

    def _is_colliding(self):
        piece = self.tetris_pieces[self.current_piece][self.rotation]

        for x, y in piece:
            x += self.pos[0]
            y += self.pos[1]
            if (
                x not in range(self.cols)
                or y not in range(self.rows)
                or self.board[y, x] == 1
            ):
                return True
        return False

    def _add_piece(self):
        self.pieces_placed += 1
        piece = self.tetris_pieces[self.current_piece][self.rotation]
        for x, y in piece:
            self.board[y + self.pos[1], x + self.pos[0]] = 1

        self._clear_lines()

        self.current_piece = self.next_piece
        self.next_piece    = random.randint(0, 6)
        self.pos           = [int(self.cols / 2) - 2, 0]
        self.rotation      = 0

        if self._is_colliding():
            self._game_over()

    def _game_over(self):
        with open("score.txt", "a") as f:
            f.write(f"Score: {self.score}\n")
            f.write(f"Steps: {self.steps}\n")
        self.alive = False

    def _rotate(self, angle):
        self.rotation = (self.rotation + angle) % 360
        if self.rotation < 0:
            self.rotation += 360

    def _hard_drop(self):
        while not self._is_colliding():
            self.pos[1] += 1
        self.pos[1] -= 1
        self._add_piece()

  
    """
    Single action step.

    DESC: Move piece one tile, rotate, or hard drop
    """
    def step(self, action: str):
        
        self.steps += 1

        match action:
            case "COUNTERCLOCKWISE": 
                self._rotate(-90)
            case "CLOCKWISE":        
                self._rotate(90)
            case "RIGHT":            
                self.pos[0] += 1
            case "LEFT":             
                self.pos[0] -= 1
            case "HARD_DROP":
                self._hard_drop()
                if self.config["render"]:
                    self._render()
                return self.get_board_copy()

        if self._is_colliding():
            match action:
                case "COUNTERCLOCKWISE": 
                    self._rotate(90)
                case "CLOCKWISE":        
                    self._rotate(-90)
                case "RIGHT":            
                    self.pos[0] -= 1
                case "LEFT":             
                    self.pos[0] += 1

        self.pos[1] += 1
        if self._is_colliding():
            self.pos[1] -= 1
            self._add_piece()

        if self.config["render"]:
            self._render()

        return self.get_board_copy()

    """
    Column action step.

    DESC: Place current piece at a given column and rotation, 
        instantly hard drop
    """
    def step_col(self, col: int, rotation: int):
        """."""
        self.steps += 1

        rotation_angle = [0, 90, 180, 270][rotation]
        piece          = self.tetris_pieces[self.current_piece][rotation_angle]

        max_x_offset = max(x for x, y in piece)
        min_x_offset = min(x for x, y in piece)

        col = max(col, -min_x_offset)
        col = min(col, self.cols - 1 - max_x_offset)

        self.pos[0]   = col
        self.rotation = rotation_angle

        if self._is_colliding():
            self._game_over()
            return self.get_board_copy()

        self._hard_drop()

        if self.config["render"]:
            self._render()

        return self.get_board_copy()

    """
    Deep copy of board. 

    0 = Empty
    1 = Filled
    2 = Current epice
    """
    def get_board_copy(self) -> np.ndarray:
        
        tempboard = np.copy(self.board)
        if self.alive:
            for x, y in self.tetris_pieces[self.current_piece][self.rotation]:
                tempboard[y + self.pos[1], x + self.pos[0]] = 2
        return tempboard

    """
    Fill bottom N rows, leaving the two middle columns empty.
    """
    def prefill_board(self, rows: int):
        for row in range(self.rows - rows, self.rows):
            for col in range(self.cols):
                if col != self.cols // 2 and col != self.cols // 2 + 1:
                    self.board[row, col] = 1

    # Render board in text
    def _render(self):
        for row in self.get_board_copy():
            print(row)
        print()

    # accessing
    def get_pieces_placed(self) -> int:
        return self.pieces_placed

    def is_alive(self) -> bool:
        return self.alive
