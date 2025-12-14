import argparse
import torch
import numpy as np

from model import EloGuesser
from bins import make_bins
from train import get_device, expected_elo

# YOUR existing pipeline
from preprocessing import eval_pgn_to_csv
from extract_from_csv import extract_features_from_csv


def load_model(model_path, in_dim, device):
    model = EloGuesser(in_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_from_features(npz_path, model, mids, device):
    """
    Uses the same feature format as training.
    """
    feats = np.load(npz_path, allow_pickle=True)

    Xw = torch.tensor(feats["Xw"], dtype=torch.float32, device=device)
    Mw = torch.tensor(feats["Mw"], dtype=torch.bool, device=device)
    Xb = torch.tensor(feats["Xb"], dtype=torch.float32, device=device)
    Mb = torch.tensor(feats["Mb"], dtype=torch.bool, device=device)

    with torch.no_grad():
        pw, pb = model(Xw, Mw, Xb, Mb)

    w_elo = expected_elo(pw, mids, device).item()
    b_elo = expected_elo(pb, mids, device).item()

    return w_elo, b_elo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to trained model (.pt)")
    parser.add_argument("pgn", help="PGN file to evaluate")
    parser.add_argument(
        "--tmp", default="__tmp_features.npz",
        help="Temporary feature file"
    )
    args = parser.parse_args()

    device = get_device()
    edges, mids = make_bins(num_bins=39, lo=0, hi=4000)

    # 1) PGN → CSV (your code)
    csv_path = args.tmp.replace(".npz", ".csv")
    eval_pgn_to_csv(args.pgn, csv_path)

    # 2) CSV → features (your code)
    extract_features_from_csv(csv_path, args.tmp)

    # 3) Infer feature dim exactly like training
    sample = np.load(args.tmp, allow_pickle=True)
    in_dim = sample["Xw"].shape[2]

    model = load_model(args.model, in_dim, device)

    # 4) Predict
    w_elo, b_elo = predict_from_features(args.tmp, model, mids, device)

    print(f"White Elo: {w_elo:.0f}")
    print(f"Black Elo: {b_elo:.0f}")


if __name__ == "__main__":
    main()