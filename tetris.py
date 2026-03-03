"""
Step Tetris Implementation:
Author: Nathan Delcampo
Date: 1/13/2026
Last Modifed: 2/3/2026
Python Version: 3.11.14

DESC: Tetris implementation with Numpy Array.
    Works off a steps with one action per step.
"""

# pyright: ignore[reportMissingImports]

import random
import sys
import numpy as np
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from enum import Enum

class Tetris:

    # Contains local coordinates of all tetris pieces at any given rotation
    tetris_pieces = {
        0: { # I
            0: [(0,0), (1,0), (2,0), (3,0)],
            90: [(1,0), (1,1), (1,2), (1,3)],
            180: [(3,0), (2,0), (1,0), (0,0)],
            270: [(1,3), (1,2), (1,1), (1,0)],
        },
        1: { # T
            0: [(1,0), (0,1), (1,1), (2,1)],
            90: [(0,1), (1,2), (1,1), (1,0)],
            180: [(1,2), (2,1), (1,1), (0,1)],
            270: [(2,1), (1,0), (1,1), (1,2)],
        },
        2: { # L
            0: [(1,0), (1,1), (1,2), (2,2)],
            90: [(0,1), (1,1), (2,1), (2,0)],
            180: [(1,2), (1,1), (1,0), (0,0)],
            270: [(2,1), (1,1), (0,1), (0,2)],
        },
        3: { # J
            0: [(1,0), (1,1), (1,2), (0,2)],
            90: [(0,1), (1,1), (2,1), (2,2)],
            180: [(1,2), (1,1), (1,0), (2,0)],
            270: [(2,1), (1,1), (0,1), (0,0)],
        },
        4: { # Z
            0: [(0,0), (1,0), (1,1), (2,1)],
            90: [(0,2), (0,1), (1,1), (1,0)],
            180: [(2,1), (1,1), (1,0), (0,0)],
            270: [(1,0), (1,1), (0,1), (0,2)],
        },
        5: { # S
            0: [(2,0), (1,0), (1,1), (0,1)],
            90: [(0,0), (0,1), (1,1), (1,2)],
            180: [(0,1), (1,1), (1,0), (2,0)],
            270: [(1,2), (1,1), (0,1), (0,0)],
        },
        6: { # O/Square
            0: [(1,0), (2,0), (1,1), (2,1)],
            90: [(1,0), (2,0), (1,1), (2,1)],
            180: [(1,0), (2,0), (1,1), (2,1)],
            270: [(1,0), (2,0), (1,1), (2,1)],
        }
    }

    class Action(Enum):
        LEFT = 1
        RIGHT = 2
        COUNTERCLOCKWISE = 3
        CLOCKWISE = 4
        NOOP = 5
        HARD_DROP = 6
    
    # Initialze configs and reset board to start game
    def __init__(self, _rows=14, _cols=6, _render=False):
        self.config = {}
        self.config["render"] = _render
        self.config["rows"] = _rows
        self.config["cols"] = _cols
        self.reset()

    # Resets board to blank state
    def reset(self):
        # Get configs
        self.cols = self.config["cols"]
        self.rows = self.config["rows"]
        # print(f"CURRENT CONFIG: {self.config}")

        # Setup board
        self.board = np.zeros((self.rows, self.cols))
        self.score = 0
        self.current_piece = random.randint(0,6)
        self.next_piece = random.randint(0,6) # Maybe change to bag system? (cycle through all pieces)
        self.pos = [int((self.cols / 2)), 0] # Middle of top of board
        self.rotation = 0
        self.steps = 0
        
        return self.board
    
    # TODO: Finish this
    # Condense information into input tensor and return 
    # (maybe have observation and action spec declared at start?)
    def observation_spec(self):
        # observation spec should have predetermined size (board + next piece should never change)
        observation_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=0,
            maximum=85,
            name='action'
        )

        # flatten board and add extra info
        tempArray = self.board.flatten()
        tempArray = np.append(tempArray, self.next_piece)
        observation_spec.from_array(tempArray)

        return observation_spec
        
    def action_spec(self):
        action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=5, name='action')

        return action_spec


    # Clears all lines that are full
    def _clear_lines(self):

        # Get full lines
        lines_full = []
        for index, row in enumerate(self.board):
            if 0 not in row:
                lines_full.append(index)
                # print(index)
            
        
        # Delete full rows
        self.board = np.delete(self.board, lines_full, axis=0)
        
        # Insert blank rows at top
        new_lines = np.zeros((len(lines_full), self.cols), dtype=int)
        self.board = np.concat((new_lines, self.board), axis=0)

        # Give score based on lines cleared
        if (len(lines_full) == 4): # TETRIS (double points)
            self.score += 8
        else:
            self.score += len(lines_full)

    # Check current piece for collisions with sides and board
    def _is_colliding(self):

        # Get local coordinates and use with position to check for collision
        piece = self.tetris_pieces[self.current_piece][self.rotation]

        for x, y in piece:
            x += self.pos[0]
            y += self.pos[1]

            if x not in range(self.cols) \
            or y not in range(self.rows) \
            or self.board[y, x] == 1:
                return True
        return False 
            
    # Adds current piece to board, randomly select another piece, end game if colliding
    def _add_piece(self):

        # Get local coordinates and use with position to add to board
        piece = self.tetris_pieces[self.current_piece][self.rotation]

        for x, y in piece:
            x += self.pos[0]
            y += self.pos[1]

            self.board[y, x] = 1

        # Clear any full lines
        self._clear_lines()

        # Get next piece and reset position
        self.current_piece = self.next_piece
        self.next_piece = random.randint(0,6)
        self.pos = [int((self.cols / 2)), 0]
        self.rotation = 0

        # End game if new piece is colloding
        if self._is_colliding():
            self._game_over()



    def _game_over(self):
        # print(f"Score: {self.score}")
        # print(f"Steps: {self.steps}")
        with open("score.txt", "a") as f:
            f.write(f"Score: {self.score}\n")
            f.write(f"Steps: {self.steps}\n")

        # Reset game
        self.reset()

    # Change rotation based on angle given
    def _rotate(self, angle):
        self.rotation += angle

        if self.rotation == 360:
            self.rotation = 0
        elif self.rotation < 0:
            self.rotation = 270

    # hard drop the piece
    def _hard_drop(self):
        while (not self._is_colliding()):
            # print(self.pos)
            self.pos[1] += 1

        self.pos[1] -= 1
        self._add_piece()       

    # Main 'loop' function of game
    def step(self, action):
        self.steps += 1
        print(f"Doing: {action}")

        # Do given action
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
                if (self.config["render"]):
                    self._render()
                return self.observation_spec

        # Check collisions and reverse action if not allowed
        if self._is_colliding():
            # print("INVALID MOVE")
            # Reverse action
            match action: 
                case "COUNTERCLOCKWISE":
                    self._rotate(90)
                case "CLOCKWISE":
                    self._rotate(-90)
                case "RIGHT":
                    self.pos[0] -= 1
                case "LEFT":
                    self.pos[0] += 1
        
        # Drop piece down by one and check collision, add to board if colliding
        self.pos[1] += 1
        if self._is_colliding():
            # print("PIECE PLACED")
            self.pos[1] -= 1
            self._add_piece()

        # Render by printing if needed
        if (self.config["render"]):
            self._render()

        return self.observation_spec 
    
    # Creates deepcopy of current board state with current piece added
    def get_board_copy(self):
        tempboard = np.copy(self.board)

        for x, y in self.tetris_pieces[self.current_piece][self.rotation]:
            x += self.pos[0]
            y += self.pos[1]

            tempboard[y, x] = 2
        
        return tempboard

    # Render board state by printing by row
    def _render(self):
        # Add current piece to temp board
        tempboard = self.get_board_copy()

        for row in tempboard:
            print(row)
        print("")
        
