from evaluation import evaluation
import chess.pgn

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

game = chess.pgn.read_game(data)

board = game.board()

for move in game.mainline_moves():
    print(f'Played move: {move}\n Best move: {evaluation(board)}')
    board.push(move)