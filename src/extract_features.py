from __future__ import annotations
import chess
import chess.pgn
import chess.engine
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


@dataclass
class GameFeatures:
    X_white: np.ndarray  # (Tw, F)
    X_black: np.ndarray  # (Tb, F)
    white_elo: int
    black_elo: int


def material_score(board: chess.Board, color: chess.Color) -> int:
    s = 0
    for pt, v in PIECE_VALUES.items():
        s += len(board.pieces(pt, color)) * v
    return s


def score_to_cp(
    score: chess.engine.PovScore, pov: chess.Color, mate_cp: int = 1000
) -> float:
    """Convert Score to centipawns from pov, clipping mates to +/- mate_cp."""
    s = score.pov(pov)
    if s.is_mate():
        m = s.mate()
        # mate in N: positive if pov mates, negative if getting mated
        return float(np.sign(m) * mate_cp) if m is not None else float(mate_cp)
    cp = s.score(mate_score=mate_cp)
    if cp is None:
        return 0.0
    return float(cp)


def clip_cp(x: float, cap: float = 1000.0) -> float:
    return float(np.clip(x, -cap, cap))


def is_capture(board: chess.Board, move: chess.Move) -> int:
    return int(board.is_capture(move))


def is_check_after(board: chess.Board, move: chess.Move) -> int:
    b2 = board.copy(stack=False)
    b2.push(move)
    return int(b2.is_check())


def has_castled(board: chess.Board, color: chess.Color) -> int:
    # crude but fine: king not on starting square and rook moved accordingly is complex;
    # simplest proxy: check if king has moved from start square.
    king_sq = board.king(color)
    return int(
        (color == chess.WHITE and king_sq != chess.E1)
        or (color == chess.BLACK and king_sq != chess.E8)
    )


def get_eval_and_topk(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    pov: chess.Color,
    depth: int = 10,
    multipv: int = 10,
) -> Tuple[float, List[float]]:
    """Returns (best_eval_cp, topk_evals_cp)."""
    info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
    # python-chess returns list when multipv>1
    if isinstance(info, dict):
        info = [info]
    evals = []
    for item in info:
        sc = item.get("score")
        if sc is None:
            continue
        evals.append(clip_cp(score_to_cp(sc, pov)))
    if not evals:
        evals = [0.0]
    evals_sorted = evals[:]  # already best-first typically
    best = evals_sorted[0]
    # pad if fewer than multipv returned
    while len(evals_sorted) < multipv:
        evals_sorted.append(evals_sorted[-1])
    return best, evals_sorted[:multipv]


def extract_one_game(
    engine: chess.engine.SimpleEngine,
    game: chess.pgn.Game,
    depth: int = 10,
    multipv: int = 10,
    cap_cp: float = 1000.0,
) -> Optional[GameFeatures]:
    headers = game.headers
    if "WhiteElo" not in headers or "BlackElo" not in headers:
        return None
    try:
        white_elo = int(headers["WhiteElo"])
        black_elo = int(headers["BlackElo"])
    except ValueError:
        return None

    board = game.board()
    Xw, Xb = [], []
    ply = 0

    for move in game.mainline_moves():
        ply += 1
        mover = board.turn  # who is about to move
        pov = mover

        # engine before
        best_before, top10 = get_eval_and_topk(
            engine, board, pov=pov, depth=depth, multipv=multipv
        )

        # played move -> after position
        after_board = board.copy(stack=False)
        after_board.push(move)

        # engine after (just best line is enough)
        info_after = engine.analyse(after_board, chess.engine.Limit(depth=depth))
        sc_after = info_after.get("score")
        played_after = (
            clip_cp(score_to_cp(sc_after, pov), cap=cap_cp) if sc_after else 0.0
        )

        # core engineered stats
        cpl = max(0.0, best_before - played_after)  # mover perspective
        e1, e2, e5 = top10[0], top10[1], top10[4]
        gap_1_2 = e1 - e2
        gap_1_5 = e1 - e5
        std_top10 = float(np.std(top10))
        count_near_best_50 = float(sum(1 for e in top10 if (e1 - e) <= 50.0))

        ply_norm = min(ply / 200.0, 1.0)
        cap = float(is_capture(board, move))
        chk = float(is_check_after(board, move))
        castled = float(has_castled(board, mover))

        mat_m = float(material_score(board, mover))
        mat_o = float(material_score(board, not mover))
        mat_d = mat_m - mat_o

        # (20 dims target; here we do 12 dims. We'll pad with simple extras below.)
        feat = [
            clip_cp(best_before, cap_cp),
            clip_cp(played_after, cap_cp),
            clip_cp(cpl, cap_cp),
            clip_cp(gap_1_2, cap_cp),
            clip_cp(gap_1_5, cap_cp),
            clip_cp(std_top10, cap_cp),
            count_near_best_50,
            ply_norm,
            cap,
            chk,
            castled,
            mat_m,
            mat_o,
            mat_d,
        ]

        # Add a few cheap extras to reach ~20 dims (optional but helpful)
        # - absolute eval magnitudes, volatility proxy, and move number buckets
        feat += [
            abs(clip_cp(best_before, cap_cp)) / cap_cp,
            abs(clip_cp(played_after, cap_cp)) / cap_cp,
            float(ply <= 20),  # opening-ish
            float(20 < ply <= 60),  # midgame-ish
            float(ply > 60),  # endgame-ish
            float(board.fullmove_number) / 100.0,
            float(board.is_repetition(2)),  # repetition-ish
        ]

        feat = np.array(feat, dtype=np.float32)

        if mover == chess.WHITE:
            Xw.append(feat)
        else:
            Xb.append(feat)

        board.push(move)

    if len(Xw) == 0 or len(Xb) == 0:
        return None

    return GameFeatures(
        X_white=np.stack(Xw),
        X_black=np.stack(Xb),
        white_elo=white_elo,
        black_elo=black_elo,
    )


def extract_from_pgn(
    pgn_path: str,
    stockfish_path: str,
    out_path: str,
    depth: int = 10,
    max_games: int = 5000,
):
    games_out = []
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
            for i in range(max_games):
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                gf = extract_one_game(engine, game, depth=depth, multipv=10)
                if gf is None:
                    continue
                games_out.append(gf)

    # save as npz with variable-length arrays via object arrays
    np.savez_compressed(
        out_path,
        X_white=np.array([g.X_white for g in games_out], dtype=object),
        X_black=np.array([g.X_black for g in games_out], dtype=object),
        white_elo=np.array([g.white_elo for g in games_out], dtype=np.int32),
        black_elo=np.array([g.black_elo for g in games_out], dtype=np.int32),
    )
    print(f"Saved {len(games_out)} games to {out_path}")


if __name__ == "__main__":
    extract_from_pgn(
        pgn_path="data/games.pgn",
        stockfish_path="stockfish/stockfish",
        out_path="data/features.npz",
        depth=10,
        max_games=2000,
    )
