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

piece_values = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
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

def eval_board(board):
    score = 0

    # Add up the material score
    for piece_type, value in piece_values.items():
        score += value * len(board.pieces(piece_type, chess.WHITE))
        score -= value * len(board.pieces(piece_type, chess.BLACK))

    # Check if either h or a pawn has moved
    has_h_pawn_moved = any(move.from_square == chess.H2 or move.to_square == chess.H2 for move in board.move_stack)
    has_a_pawn_moved = any(move.from_square == chess.A2 or move.to_square == chess.A2 for move in board.move_stack)

    # Penalize for starting with h or a pawns
    if not has_h_pawn_moved:
        score -= 0.2
    if not has_a_pawn_moved:
        score -= 0.2

    # Penalize hanging pieces
    for square, piece in board.piece_map().items():
        if piece.color == board.turn:
            attackers = [attacker for attacker in board.attackers(not piece.color, square)]
            try:
                attackers = [attacker for attacker in attackers
                             if piece_values.get(attacker.piece_type, 0) >= piece_values.get(piece.piece_type, 0)]
            except AttributeError:
                print()

            if not attackers and piece_values.get(piece.piece_type, 0) > 1:
                score -= piece_values.get(piece.piece_type, 0) / 2

    # Add a bonus for controlling the center
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    king_square = board.king(board.turn)

    for square in center_squares:
        piece = board.piece_at(square)
        if piece is not None and piece.color == board.turn:
            score += 0.4 * piece_values.get(piece.piece_type, 0)

    # Add a bonus for having a more active position
    for piece in board.piece_map().values():
        if isinstance(piece, chess.Piece) and piece.color == board.turn:
            try:
                mobility = len(board.attacks(piece.square))
                score += 0.5 * piece_values.get(piece.piece_type, 0) * mobility
            except AttributeError:
                print()
    for square in center_squares:
        if board.is_controlled(square) and chess.square_distance(square, king_square) <= 4:
            score += 0.1

        # Add a bonus for piece mobility near the king
    for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        for piece in board.pieces(piece_type, board.turn):
            if chess.square_distance(piece, king_square) <= 4:
                mobility = len(board.attacks(piece))
                score += 0.2 * piece_values.get(piece_type, 0) * mobility

    print("Score: {}".format(score))

    return score

def choose_move(board, player, color):
    legal_moves = list(board.legal_moves)
    best_move = None
    best_score = float('-inf') if player == chess.WHITE else float('inf')
    for move in legal_moves:
        # check if move leads to hanging piece
        if board.is_capture(move):
            piece_captured = board.piece_at(move.to_square)
            if piece_captured is not None:
                eval_board_after_move = eval_board(board)
                board.push(move)
                if board.is_check():
                    eval_board_after_move -= 1.0 * piece_values[piece_captured.piece_type]
                if eval_board_after_move < eval_board(board):
                    board.pop()
                    continue
                # check if capturing piece has equal or lower value compared to the captured piece
                piece_capturing = board.piece_at(move.from_square)
                if piece_capturing is not None and piece_values.get(piece_capturing.piece_type, 0) <= piece_values.get(piece_captured.piece_type, 0):
                    best_score += 0.5 * piece_values[piece_captured.piece_type]  # add a fixed bonus to the score
                board.pop()

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