import re
import argparse
import chess.pgn

EVAL_RE = re.compile(r"\[%eval\s+([^\]]+)\]")


def parse_eval_from_comment(comment: str):
    """
    Returns eval in centipawns from a PGN comment containing [%eval ...]
    Examples:
      [%eval 0.34]   -> +34 cp
      [%eval -1.2]   -> -120 cp
      [%eval #3]     -> mate in 3 (returned as None by default)
    """
    if not comment:
        return None
    m = EVAL_RE.search(comment)
    if not m:
        return None

    raw = m.group(1).strip()
    # mate scores like "#3" or "#-2"
    if raw.startswith("#"):
        return None  # keep it simple; you can handle mate separately if you want

    try:
        # value is in pawns; convert to centipawns
        return int(round(float(raw) * 100))
    except ValueError:
        return None


def extract_moves_and_evals(pgn_path: str, game_index: int = 0):
    with open(pgn_path, "r", encoding="utf-8") as f:
        # read up to game_index
        game = None
        for _ in range(game_index + 1):
            game = chess.pgn.read_game(f)
            if game is None:
                raise SystemExit(f"Could not read game #{game_index} from PGN.")

    board = game.board()
    out = []

    prev_eval = None
    node = game

    # Walk mainline: node -> child -> child -> ...
    while node.variations:
        next_node = node.variations[0]
        move = next_node.move
        san = board.san(move)
        board.push(move)

        # eval is typically stored on the node you just moved to
        eval_cp = parse_eval_from_comment(next_node.comment)

        delta = None
        if eval_cp is not None and prev_eval is not None:
            delta = eval_cp - prev_eval

        out.append(
            {
                "ply": board.ply(),
                "san": san,
                "eval_cp": eval_cp,  # centipawns, + = white better
                "delta_cp": delta,  # change vs previous eval
            }
        )

        if eval_cp is not None:
            prev_eval = eval_cp

        node = next_node

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn", help="Path to PGN file")
    ap.add_argument(
        "--game", type=int, default=0, help="0-based game index inside the PGN"
    )
    args = ap.parse_args()

    rows = extract_moves_and_evals(args.pgn, args.game)

    # simple printing
    for r in rows:
        ply = r["ply"]
        san = r["san"]
        e = r["eval_cp"]
        d = r["delta_cp"]
        e_str = "NA" if e is None else f"{e:+d}cp"
        d_str = "NA" if d is None else f"{d:+d}cp"
        print(f"{ply:>3}. {san:<8} eval={e_str:<8} Δ={d_str}")


if __name__ == "__main__":
    main()
