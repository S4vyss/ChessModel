import re

from evaluation import evaluation
import chess.pgn
import numpy as np

# board format
# r n b q k b n r
# p p p p p p p p
# . . . . . . . .
# . . . . . . . .
# . . . . . . . .
# . . . . . . . .
# P P P P P P P P
# R N B Q K B N R

data = open('fide2000Games.pgn')
data = data.read()
data = data.split('\n\n')
data = [x for x in data if not x.startswith('[')]

for i in range(len(data)):
    data[i] = re.sub('{.*}', '', data[i])

letter_2_num = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7
}

num_2_letter = {
    0: 'a',
    1: "b",
    2: "c",
    3: "d",
    4: "e",
    5: "f",
    6: "g",
    7: "h"
}

def create_rep_layer(board, pieceType):
    s = str(board)
    s = re.sub(f'[^{pieceType}{pieceType.upper()} \n]', '.', s)
    s = re.sub(f'{pieceType}', '-1', s)
    s = re.sub(f'{pieceType.upper()}', '1', s)
    s = re.sub(f'\.', '0', s)

    board_mat = []

    for row in s.split('\n'):
        row = row.split(' ')
        row = [int(x) for x in row]
        board_mat.append(row)
    return np.array(board_mat)

def board_to_rep(board):
    pieces = ['p', 'r', 'n', 'b', 'q', 'k']
    layers = []
    for piece in pieces:
        layers.append(create_rep_layer(board, piece))
    board_rep = np.stack(layers)
    return board_rep

def move_rep(move, board):
    board.push_san(move).uci()
    move = str(board.pop())

    from_output_layer = np.zeros((8, 8))
    from_row = 8 - int(move[1])
    from_column = letter_2_num[move[0]]
    from_output_layer[from_row, from_column] = 1

    to_output_layer = np.zeros((8, 8))
    to_row = 8 - int(move[3])
    to_column = letter_2_num[move[2]]
    to_output_layer[to_row, to_column] = 1

    return np.stack([from_output_layer, to_output_layer])

def create_move_list(s):
    return re.sub('\d*\. ', '', s).split(' ')[:-1]