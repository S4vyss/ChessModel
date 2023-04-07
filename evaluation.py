import chess.pgn
import chess
import subprocess

def evaluation(board):
        engine = subprocess.Popen(
                './evaluation/stockfish_15.1_win_x64_avx2/stockfish-windows-2022-x86-64-avx2.exe',
                universal_newlines=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
        )

        engine.stdin.write("uci\n")
        engine.stdin.flush()

        while True:
                text = engine.stdout.readline().strip()
                if text == "uciok":
                        break
        engine.stdin.write(f'position fen {board.fen()}\n')
        engine.stdin.write('go depth 10\n')
        engine.stdin.flush()

        best_move = None

        while True:
                text = engine.stdout.readline().strip()
                if text.startswith('bestmove'):
                        best_move = chess.Move.from_uci(text.split()[1])
                        break

        engine.stdin.write('quit\n')
        engine.stdin.flush()
        engine.kill()

        return best_move

