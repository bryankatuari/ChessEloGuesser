import ast
import numpy as np
import pandas as pd
import chess

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def parse_eval(x, mate_cp=1000.0):
    # x can be int/float, or strings like 'M-1', 'M0'
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if isinstance(x, str):
        s = x.strip().strip("'").strip('"')
        if s.startswith("M"):
            # e.g. M-1, M0, M3
            try:
                n = int(s[1:])
                # map mate to +/- mate_cp (sign from n)
                if n == 0:
                    return mate_cp
                return float(np.sign(n) * mate_cp)
            except:
                return 0.0
        try:
            return float(s)
        except:
            return 0.0
    return 0.0


def clip(x, cap=1000.0):
    return float(np.clip(x, -cap, cap))


def material_score(board: chess.Board, color: chess.Color) -> int:
    s = 0
    for pt, v in PIECE_VALUES.items():
        s += len(board.pieces(pt, color)) * v
    return s


def has_castled(board: chess.Board, color: chess.Color) -> float:
    king_sq = board.king(color)
    if color == chess.WHITE:
        return 1.0 if king_sq != chess.E1 else 0.0
    else:
        return 1.0 if king_sq != chess.E8 else 0.0


def phase_onehot(board: chess.Board):
    # simple material-based phase
    # compute total "non-pawn" material on board
    total = 0
    for color in [chess.WHITE, chess.BLACK]:
        total += len(board.pieces(chess.QUEEN, color)) * 9
        total += len(board.pieces(chess.ROOK, color)) * 5
        total += (
            len(board.pieces(chess.BISHOP, color))
            + len(board.pieces(chess.KNIGHT, color))
        ) * 3

    # thresholds are rough but fine
    if total >= 36:  # lots of pieces
        return (1.0, 0.0, 0.0)  # opening
    elif total >= 18:
        return (0.0, 1.0, 0.0)  # midgame
    else:
        return (0.0, 0.0, 1.0)  # endgame


def san_flags(san: str):
    san = san.strip()
    is_cap = 1.0 if "x" in san else 0.0
    is_chk = 1.0 if ("+" in san or "#" in san) else 0.0
    is_mate = 1.0 if "#" in san else 0.0
    is_promo = 1.0 if "=" in san else 0.0
    return is_cap, is_chk, is_mate, is_promo


def extract_features_from_csv(csv_path: str, out_npz: str, cap_cp=1000.0):
    df = pd.read_csv(csv_path)

    X_white_list = []
    X_black_list = []
    white_elos = []
    black_elos = []

    skipped_parse = 0
    skipped_len = 0

    for _, row in df.iterrows():
        moves = ast.literal_eval(row["moves"])
        evals_raw = ast.literal_eval(row["evals"])

        if len(moves) == 0 or len(evals_raw) != len(moves):
            skipped_len += 1
            continue

        evals_white = [parse_eval(e) for e in evals_raw]

        board = chess.Board()
        Xw, Xb = [], []

        prev_eval_white = 0.0

        ok = True
        for i, san in enumerate(moves):
            mover_is_white = i % 2 == 0
            mover = chess.WHITE if mover_is_white else chess.BLACK

            # evals are assumed from White POV *after* this move
            e_after_white = evals_white[i]
            e_before_white = prev_eval_white

            # convert to mover POV
            sign = 1.0 if mover_is_white else -1.0
            e_before = clip(sign * e_before_white, cap_cp)
            e_after = clip(sign * e_after_white, cap_cp)

            delta = clip(e_after - e_before, cap_cp)
            swing = clip(abs(delta), cap_cp)

            # SAN -> move (to get capture/check reliably + update board for material/phase)
            try:
                move = board.parse_san(san)
            except Exception:
                ok = False
                break

            is_cap, is_chk, is_mate, is_promo = san_flags(san)

            # game context features
            ply = i + 1
            ply_norm = min(ply / 200.0, 1.0)
            ph_o, ph_m, ph_e = phase_onehot(board)

            # material snapshot BEFORE the move (from mover perspective)
            mat_m = float(material_score(board, mover))
            mat_o = float(material_score(board, not mover))
            mat_d = mat_m - mat_o

            castled = has_castled(board, mover)

            # build feature vector (~20 dims)
            feat = np.array(
                [
                    e_before,  # 0
                    e_after,  # 1
                    delta,  # 2
                    swing,  # 3
                    is_cap,  # 4
                    is_chk,  # 5
                    is_mate,  # 6
                    is_promo,  # 7
                    ply_norm,  # 8
                    ph_o,
                    ph_m,
                    ph_e,  # 9-11
                    mat_m,  # 12
                    mat_o,  # 13
                    mat_d,  # 14
                    castled,  # 15
                    abs(e_before) / cap_cp,  # 16
                    abs(e_after) / cap_cp,  # 17
                    float(board.is_repetition(2)),  # 18
                    float(board.can_claim_threefold_repetition()),  # 19
                ],
                dtype=np.float32,
            )

            if mover_is_white:
                Xw.append(feat)
            else:
                Xb.append(feat)

            # advance board
            board.push(move)
            prev_eval_white = e_after_white

        if not ok or len(Xw) == 0 or len(Xb) == 0:
            skipped_parse += 1
            continue

        X_white_list.append(np.stack(Xw))
        X_black_list.append(np.stack(Xb))
        white_elos.append(int(row["white_elo"]))
        black_elos.append(int(row["black_elo"]))

    np.savez_compressed(
        out_npz,
        X_white=np.array(X_white_list, dtype=object),
        X_black=np.array(X_black_list, dtype=object),
        white_elo=np.array(white_elos, dtype=np.int32),
        black_elo=np.array(black_elos, dtype=np.int32),
    )

    F = X_white_list[0].shape[1] if X_white_list else 0
    print(f"Saved {len(white_elos)} games to {out_npz}. Feature dim = {F}")
    print(
        f"Skipped (len mismatch): {skipped_len}, Skipped (SAN parse fail/empty): {skipped_parse}"
    )


if __name__ == "__main__":
    extract_features_from_csv(
        csv_path="data/processed_data.csv",
        out_npz="data/features.npz",
        cap_cp=1000.0,
    )
