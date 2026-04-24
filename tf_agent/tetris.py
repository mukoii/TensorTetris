"""
Step Tetris Implementation
Author: Nathan Delcampo  |  Patch: removed tf_agents
"""

import random
import numpy as np
from enum import Enum


class Tetris:

    tetris_pieces = {
        0: {0:[(0,0),(1,0),(2,0),(3,0)], 90:[(1,0),(1,1),(1,2),(1,3)], 180:[(3,0),(2,0),(1,0),(0,0)], 270:[(1,3),(1,2),(1,1),(1,0)]},
        1: {0:[(1,0),(0,1),(1,1),(2,1)], 90:[(0,1),(1,2),(1,1),(1,0)], 180:[(1,2),(2,1),(1,1),(0,1)], 270:[(2,1),(1,0),(1,1),(1,2)]},
        2: {0:[(1,0),(1,1),(1,2),(2,2)], 90:[(0,1),(1,1),(2,1),(2,0)], 180:[(1,2),(1,1),(1,0),(0,0)], 270:[(2,1),(1,1),(0,1),(0,2)]},
        3: {0:[(1,0),(1,1),(1,2),(0,2)], 90:[(0,1),(1,1),(2,1),(2,2)], 180:[(1,2),(1,1),(1,0),(2,0)], 270:[(2,1),(1,1),(0,1),(0,0)]},
        4: {0:[(0,0),(1,0),(1,1),(2,1)], 90:[(0,2),(0,1),(1,1),(1,0)], 180:[(2,1),(1,1),(1,0),(0,0)], 270:[(1,0),(1,1),(0,1),(0,2)]},
        5: {0:[(2,0),(1,0),(1,1),(0,1)], 90:[(0,0),(0,1),(1,1),(1,2)], 180:[(0,1),(1,1),(1,0),(2,0)], 270:[(1,2),(1,1),(0,1),(0,0)]},
        6: {0:[(1,0),(2,0),(1,1),(2,1)], 90:[(1,0),(2,0),(1,1),(2,1)], 180:[(1,0),(2,0),(1,1),(2,1)], 270:[(1,0),(2,0),(1,1),(2,1)]},
    }

    class Action(Enum):
        LEFT=1; RIGHT=2; COUNTERCLOCKWISE=3; CLOCKWISE=4; NOOP=5; HARD_DROP=6

    def __init__(self, _rows=14, _cols=8, _render=False):
        self.config = {"render": _render, "rows": _rows, "cols": _cols}
        self.reset()

    def reset(self):
        self.cols=self.config["cols"]; self.rows=self.config["rows"]
        self.board=np.zeros((self.rows,self.cols)); self.score=0
        self.current_piece=random.randint(0,6); self.next_piece=random.randint(0,6)
        self.pos=[1,1]; self.rotation=0; self.steps=0; self.pieces_placed=0; self.alive=True
        return self.board

    def _clear_lines(self):
        lines_full=[i for i,row in enumerate(self.board) if 0 not in row]
        self.board=np.delete(self.board,lines_full,axis=0)
        self.board=np.concatenate((np.zeros((len(lines_full),self.cols),dtype=int),self.board),axis=0)
        self.score += 8 if len(lines_full)==4 else len(lines_full)

    def _is_colliding(self):
        for x,y in self.tetris_pieces[self.current_piece][self.rotation]:
            x+=self.pos[0]; y+=self.pos[1]
            if x not in range(self.cols) or y not in range(self.rows) or self.board[y,x]==1:
                return True
        return False

    def _add_piece(self):
        self.pieces_placed+=1
        for x,y in self.tetris_pieces[self.current_piece][self.rotation]:
            self.board[y+self.pos[1],x+self.pos[0]]=1
        self._clear_lines()
        self.current_piece=self.next_piece; self.next_piece=random.randint(0,6)
        self.pos=[int(self.cols/2)-2,0]; self.rotation=0
        if self._is_colliding(): self._game_over()

    def _game_over(self):
        with open("score.txt","a") as f: f.write(f"Score:{self.score}\nSteps:{self.steps}\n")
        self.alive=False

    def _hard_drop(self):
        while not self._is_colliding(): self.pos[1]+=1
        self.pos[1]-=1; self._add_piece()

    def step_col(self, col, rotation):
        self.steps+=1
        angle=[0,90,180,270][rotation]
        piece=self.tetris_pieces[self.current_piece][angle]
        max_x=max(x for x,y in piece); min_x=min(x for x,y in piece)
        col=max(col,-min_x); col=min(col,self.cols-1-max_x)
        self.pos[0]=col; self.rotation=angle
        if self._is_colliding(): self._game_over(); return self.get_board_copy()
        self._hard_drop()
        if self.config["render"]: self._render()
        return self.get_board_copy()

    def get_board_copy(self):
        b=np.copy(self.board)
        if self.alive:
            for x,y in self.tetris_pieces[self.current_piece][self.rotation]:
                ny,nx=y+self.pos[1],x+self.pos[0]
                if 0<=ny<self.rows and 0<=nx<self.cols: b[ny,nx]=2
        return b

    def prefill_board(self, rows):
        for row in range(self.rows-rows,self.rows):
            for col in range(self.cols):
                if col!=self.cols//2 and col!=self.cols//2+1: self.board[row,col]=1

    def _render(self):
        for row in self.get_board_copy(): print(row)
        print()

    def get_pieces_placed(self): return self.pieces_placed
    def is_alive(self): return self.alive
