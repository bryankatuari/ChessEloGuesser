import ast
import numpy as np
import pandas as pd


def parse_eval(x, mate_cp=1000.0):
    # x may be int/float, or strings like 'M-1', 'M0'
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if isinstance(x, str):
        x = x.strip().strip("'").strip('"')
        if x.startswith("M"):
            # 'M-1' => losing mate soon for side to move in that POV convention.
            # We map mate to +/- mate_cp using sign of number after M.
            try:
                n = int(x[1:])
                if n == 0:
                    return mate_cp
                return np.sign(n) * mate_cp
            except:
                return 0.0
        try:
            return float(x)
        except:
            return 0.0
    return 0.0


def clip(x, cap=1000.0):
    return float(np.clip(x, -cap, cap))


def san_is_capture(san: str) -> float:
    return 1.0 if "x" in san else 0.0


def san_is_check(san: str) -> float:
    return 1.0 if ("+" in san or "#" in san) else 0.0


def extract_features_from_csv(csv_path: str, out_npz: str, cap_cp=1000.0):
    df = pd.read_csv(csv_path)

    X_white_list = []
    X_black_list = []
    white_elos = []
    black_elos = []

    for _, row in df.iterrows():
        moves = ast.literal_eval(row["moves"])
        evals_raw = ast.literal_eval(row["evals"])

        if len(moves) == 0 or len(evals_raw) != len(moves):
            continue

        evals = [parse_eval(e) for e in evals_raw]

        Xw, Xb = [], []
        prev_eval_white = 0.0

        for i, san in enumerate(moves):
            mover_is_white = i % 2 == 0

            e_after_white = evals[i]
            e_before_white = prev_eval_white

            # mover-perspective conversion
            sign = 1.0 if mover_is_white else -1.0
            e_before = clip(sign * e_before_white, cap_cp)
            e_after = clip(sign * e_after_white, cap_cp)

            delta = clip(e_after - e_before, cap_cp)
            swing = clip(abs(delta), cap_cp)

            feat = np.array(
                [
                    e_before,
                    e_after,
                    delta,
                    swing,
                    san_is_capture(san),
                    san_is_check(san),
                ],
                dtype=np.float32,
            )

            if mover_is_white:
                Xw.append(feat)
            else:
                Xb.append(feat)

            prev_eval_white = e_after_white

        if len(Xw) == 0 or len(Xb) == 0:
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
    print(
        f"Saved {len(white_elos)} games to {out_npz}. Feature dim = {X_white_list[0].shape[1] if X_white_list else 'n/a'}"
    )


if __name__ == "__main__":
    extract_features_from_csv(
        csv_path="data/processed_data.csv",
        out_npz="data/features.npz",
    )
