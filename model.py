import re

import torch

from evaluation import evaluation
import chess.pgn
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os

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

game = chess.Board()

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
    moves = re.sub('\d*\. ', '', s).split(' ')[:-1]
    return list(filter(lambda x: x != '', moves))

class ChessDataset(Dataset):
    def __init__(self, games):
        super(ChessDataset, self).__init__()
        self.games = np.array(games)

    def __len__(self):
        return 2000

    def __getitem__(self, index):
        while True:
            game_i = np.random.randint(self.games.shape[0])
            random_game = data[game_i]
            moves = create_move_list(random_game)
            if moves:
                break

        if len(moves) <= 1:
            return self.__getitem__(index)
        game_state_i = np.random.randint(len(moves) - 1)

        next_move = moves[game_state_i]
        moves = moves[:game_state_i]

        board = chess.Board()

        for move in moves:
            board.push_san(move)

        x = board_to_rep(board)
        y = move_rep(next_move, board) # May want to try to use stockfish's next move as opposed to next move played in the game
        if game_state_i % 2 == 1:
            x *= 1
        return x, y

data_train = ChessDataset(data)
data_train_loader = DataLoader(data_train, batch_size=32, shuffle=True, drop_last=True)

class Module(nn.Module):

    def __init__(self, hidden_size):
        super(Module, self).__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.activation1 = nn.SELU()
        self.activation2 = nn.SELU()

    def forward(self, x):
        x_input = torch.clone(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + x_input
        x = self.activation2(x)
        return x

class ChessNetwork(nn.Module):

    def __init__(self, hidden_layers=4, hidden_size=200):
        super(ChessNetwork, self).__init__()
        self.hidden_layers = hidden_layers
        self.input_layer = nn.Conv2d(6, hidden_size, 3, stride=1, padding=1)
        self.module_list = nn.ModuleList([Module(hidden_size) for i in range(hidden_layers)])
        self.output_layer = nn.Conv2d(hidden_size, 2, 3, stride=1, padding=1)

    def forward(self, x):

        x = x.float()
        x = self.input_layer(x)
        x = F.relu(x)

        for i in range(self.hidden_layers):
            x = self.module_list[i](x)

        x = self.output_layer(x)

        return x

metric_from = nn.CrossEntropyLoss()
metric_to = nn.CrossEntropyLoss()

model = ChessNetwork()

# Optimizer

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
# for epoch in range(10):
#    epoch_loss = 0.0
#    for batch_index, (inputs, targets) in enumerate(data_train_loader):
#
#        optimizer.zero_grad()
#        outputs = model(inputs)
#
#        # Loss
#
#        loss_from = metric_from(outputs[:, 0, :], targets[:, 0, :])
#        loss_to = metric_to(outputs[:, 1, :], targets[:, 1, :])
#        loss = loss_from + loss_to
#
#        loss.backward()
#        optimizer.step()
#
#        # Accumulate the loss
#
#        epoch_loss += loss.item()
#
#    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 10, epoch_loss / len(data_train_loader)))

if not os.path.exists('trainedModel'):
    os.mkdir('trainedModel')

model_path = os.path.join('trainedModel', 'model.pt')
torch.save(model.state_dict(), model_path)