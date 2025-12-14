import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import EloDataset, pad_collate
from model import EloGuesser
from bins import make_bins, gaussian_soft_labels


def kl_loss(p_pred: torch.Tensor, p_true: torch.Tensor) -> torch.Tensor:
    """
    KL(p_true || p_pred). Both are distributions over bins.
    """
    eps = 1e-8
    return (
        (p_true * (torch.log(p_true + eps) - torch.log(p_pred + eps))).sum(dim=1).mean()
    )


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def expected_elo(
    p: torch.Tensor, bin_mids: np.ndarray, device: torch.device
) -> torch.Tensor:
    mids = torch.tensor(bin_mids, dtype=torch.float32, device=device)
    return (p * mids).sum(dim=1)


def mae(pred: torch.Tensor, true: torch.Tensor) -> float:
    return (pred - true).abs().mean().item()


def main():
    device = get_device()
    print("Using device:", device)

    # Load raw elos (for MAE) and to create soft-label targets
    feats = np.load("data/features.npz", allow_pickle=True)
    white_elo = feats["white_elo"].astype(np.float32)
    black_elo = feats["black_elo"].astype(np.float32)

    N = len(white_elo)
    if N < 5:
        raise RuntimeError(f"Need more games than {N} to do a train/val split.")

    # Bins + soft labels
    edges, mids = make_bins(num_bins=39, lo=0, hi=4000)
    yw_all = gaussian_soft_labels(white_elo, edges, sigma=200.0)
    yb_all = gaussian_soft_labels(black_elo, edges, sigma=200.0)

    # Train/val split
    idx = np.arange(N)
    np.random.seed(0)
    np.random.shuffle(idx)
    split = int(0.8 * N)
    train_idx = idx[:split]
    val_idx = idx[split:]

    ds_train = EloDataset("data/features.npz", yw_all, yb_all, indices=train_idx)
    ds_val = EloDataset("data/features.npz", yw_all, yb_all, indices=val_idx)

    # Batch sizes: small for tiny datasets; bigger when you have many games
    batch_size = 8 if N < 200 else 32

    dl_train = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, collate_fn=pad_collate
    )
    dl_val = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False, collate_fn=pad_collate
    )

    # Infer feature dimension
    in_dim = ds_train[0][0].shape[1]
    model = EloGuesser(in_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # True Elo tensors for MAE (val)
    w_true_val = torch.tensor(white_elo[val_idx], dtype=torch.float32, device=device)
    b_true_val = torch.tensor(black_elo[val_idx], dtype=torch.float32, device=device)

    for epoch in range(15):
        # ---- train ----
        model.train()
        total_loss = 0.0

        for Xw, Mw, Xb, Mb, yw_b, yb_b in dl_train:
            Xw, Mw = Xw.to(device), Mw.to(device)
            Xb, Mb = Xb.to(device), Mb.to(device)
            yw_b, yb_b = yw_b.to(device), yb_b.to(device)

            pw, pb = model(Xw, Mw, Xb, Mb)
            loss = kl_loss(pw, yw_b) + kl_loss(pb, yb_b)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += float(loss.item())

        train_loss = total_loss / max(1, len(dl_train))

        # ---- validate ----
        model.eval()
        with torch.no_grad():
            pw_list, pb_list = [], []
            for Xw, Mw, Xb, Mb, yw_b, yb_b in dl_val:
                Xw, Mw = Xw.to(device), Mw.to(device)
                Xb, Mb = Xb.to(device), Mb.to(device)

                pw, pb = model(Xw, Mw, Xb, Mb)
                pw_list.append(pw)
                pb_list.append(pb)

            pw = torch.cat(pw_list, dim=0)
            pb = torch.cat(pb_list, dim=0)

            w_pred = expected_elo(pw, mids, device)
            b_pred = expected_elo(pb, mids, device)

            w_mae = mae(w_pred, w_true_val)
            b_mae = mae(b_pred, b_true_val)

        print(
            f"Epoch {epoch:02d} | TrainLoss {train_loss:.4f} | ValMAE W {w_mae:.1f} B {b_mae:.1f}"
        )
    torch.save(model.state_dict(), "elo_guesser.pt")


if __name__ == "__main__":
    main()
