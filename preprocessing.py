import chess
import chess.pgn
import chess.engine
import csv
import time
import sys

ENGINE_DEPTH = 10
MAX_GAMES = 5000
PGN_PATH = "data/lichess_db_standard_rated_2018-02.1.pgn"
STOCKFISH_PATH = "stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe"
OUTPUT_CSV = "data/processed_data.csv"

def is_standard_game(game):
    """Return True if the PGN game is standard chess."""
    variant = game.headers.get("Event", "Rated Standard Game")
    return variant.lower() == "rated standard game"

def print_progress(done, total, start_time):
    elapsed = time.time() - start_time
    avg = elapsed / done if done else 0
    remaining = (total - done) * avg

    bar_len = 30
    filled = int(bar_len * done / total)
    bar = "█" * filled + "-" * (bar_len - filled)

    sys.stdout.write(
        f"\r[{bar}] {done}/{total} "
        f"Elapsed: {elapsed:.1f}s "
        f"ETA: {remaining:.1f}s"
    )
    sys.stdout.flush()

def eval_pgn_to_csv(pgn_path):
    start_time = time.time()
    games_done = 0

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["moves", "evals", "white_elo", "black_elo"])

        with open(pgn_path, encoding="utf-8") as pgn_file, \
             chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:

            while games_done < MAX_GAMES:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                if not is_standard_game(game):
                    continue

                board = game.board()
                moves = []
                evals = []

                for move in game.mainline_moves():
                    san = board.san(move)
                    board.push(move)

                    info = engine.analyse(
                        board,
                        chess.engine.Limit(depth=ENGINE_DEPTH)
                    )

                    score = info["score"].pov(chess.WHITE)

                    if score.is_mate():
                        eval_score = f"M{score.mate()}"
                    else:
                        eval_score = score.score()

                    moves.append(san)
                    evals.append(eval_score)

                white_elo = game.headers.get("WhiteElo", "")
                black_elo = game.headers.get("BlackElo", "")

                writer.writerow([
                    moves,
                    evals,
                    white_elo,
                    black_elo
                ])

                games_done += 1
                print_progress(games_done, MAX_GAMES, start_time)

    print(f"\n\nFinished. Saved {games_done} games to {OUTPUT_CSV}")

eval_pgn_to_csv(PGN_PATH)