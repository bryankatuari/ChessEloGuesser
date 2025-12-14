import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import EloGuesser
from bins import make_bins
from dataset import EloDataset, pad_collate
from train import get_device, expected_elo
from preprocessing import eval_pgn_to_csv
from extract_from_csv import extract_features_from_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to trained model (.pt)")
    parser.add_argument("pgn", help="PGN file")
    args = parser.parse_args()

    device = get_device()
    _, mids = make_bins(num_bins=39, lo=0, hi=4000)

    # ----------------------------
    # PGN → CSV → NPZ
    # ----------------------------
    csv_path = "__tmp_features.csv"
    npz_path = "__tmp_features.npz"

    eval_pgn_to_csv(args.pgn, csv_path)
    extract_features_from_csv(csv_path, npz_path)

    # ----------------------------
    # Dummy labels (required by Dataset API)
    # ----------------------------
    feats = np.load(npz_path, allow_pickle=True)
    n = len(feats["X_white"])

    yw_dummy = np.zeros((n, 39), dtype=np.float32)
    yb_dummy = np.zeros((n, 39), dtype=np.float32)

    # ----------------------------
    # Dataset + Loader (same as training)
    # ----------------------------
    ds = EloDataset(npz_path, yw_dummy, yb_dummy)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=pad_collate)

    # Infer feature dim EXACTLY like training
    in_dim = ds[0][0].shape[1]

    model = EloGuesser(in_dim)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    # ----------------------------
    # Predict
    # ----------------------------
    with torch.no_grad():
        for i, (Xw, Mw, Xb, Mb, _, _) in enumerate(dl):
            Xw, Mw = Xw.to(device), Mw.to(device)
            Xb, Mb = Xb.to(device), Mb.to(device)

            pw, pb = model(Xw, Mw, Xb, Mb)

            w_elo = expected_elo(pw, mids, device).item()
            b_elo = expected_elo(pb, mids, device).item()

            print(f"Game {i+1}")
            print(f"  White Elo: {w_elo:.0f}")
            print(f"  Black Elo: {b_elo:.0f}")


if __name__ == "__main__":
    main()
