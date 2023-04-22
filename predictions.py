import chess
import numpy as np
import torch
from model import ChessNetwork, board_to_rep

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

model = ChessNetwork()
model.load_state_dict(torch.load('trainedModel/model.pt'))
model.eval()

# def check_mate_single(board):
#     board = board.copy()
#
#     legal_moves = list(board.legal_moves)
#
#     for moveOnBoard in legal_moves:
#         board.push_san(str(moveOnBoard))
#         if board.is_checkmate():
#             moveOnBoard = board.pop()
#             return moveOnBoard

def distribution_over_moves(vals):
    probs = np.array(vals)
    probs = np.exp(probs)
    probs = probs / probs.sum()
    probs = probs ** 3
    probs = probs / probs.sum()
    return probs

# def predict(x):
#
#     with torch.no_grad():
#         output = model(x)
#         print(model(x))
#     return output.argmax().item()

def choose_move(board, player, color):
    legal_moves = list(board.legal_moves)
    best_move = None
    best_score = float('-inf') if player == chess.WHITE else float('inf')
    for move in legal_moves:
        x = torch.Tensor(board_to_rep(board)).float()
        if color == chess.BLACK:
            x *= -1
        x = x.unsqueeze(0)
        output = model(x)  # shape: 1, 2, 8, 8
        score = output[0][0][move.to_square // 8][move.to_square % 8].item()  # extract score from tensor and convert to Python scalar
        if player == chess.BLACK:
            score = -score
        if player == chess.WHITE and score > best_score:
            best_move = move
            best_score = score
        elif player == chess.BLACK and score < best_score:
            best_move = move
            best_score = score
    return best_move




gameBoard = chess.Board()
player = chess.WHITE
color = chess.BLACK

while not gameBoard.is_game_over():
    if gameBoard.turn == player:
        player_move = input("Enter your move: ")
        gameBoard.push_uci(player_move)
    else:
        engine_move = choose_move(gameBoard, player, color)
        print(engine_move)
        print(type(engine_move))
        gameBoard.push_uci(engine_move.uci())
    print(gameBoard)