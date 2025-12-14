import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import EloDataset, pad_collate
from model import EloGuesser
from bins import make_bins, gaussian_soft_labels


def kl_loss(p_pred: torch.Tensor, p_true: torch.Tensor) -> torch.Tensor:
    """KL(p_true || p_pred). Both are distributions over bins."""
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Note: MPS can still be nondeterministic; this is "best effort".


def expected_elo(
    p: torch.Tensor, bin_mids: np.ndarray, device: torch.device
) -> torch.Tensor:
    mids = torch.tensor(bin_mids, dtype=torch.float32, device=device)
    return (p * mids).sum(dim=1)


def mae(pred: torch.Tensor, true: torch.Tensor) -> float:
    return (pred - true).abs().mean().item()


def ensure_dirs():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("predictions", exist_ok=True)


def train_one_seed(seed: int, epochs: int = 15, patience: int = 4) -> dict:
    """
    Train for one random seed. Returns dict with best metrics and file paths.
    Early stops based on overall val MAE (avg of white+black).
    """
    ensure_dirs()
    device = get_device()
    set_seed(seed)

    feats = np.load("data/features.npz", allow_pickle=True)
    white_elo = feats["white_elo"].astype(np.float32)
    black_elo = feats["black_elo"].astype(np.float32)

    N = len(white_elo)
    if N < 50:
        print(f"[seed {seed}] Warning: only {N} games. Validation will be noisy.")

    edges, mids = make_bins(num_bins=39, lo=0, hi=4000)
    yw_all = gaussian_soft_labels(white_elo, edges, sigma=100.0)
    yb_all = gaussian_soft_labels(black_elo, edges, sigma=100.0)

    # Split (seed-dependent)
    idx = np.arange(N)
    np.random.shuffle(idx)
    split = int(0.8 * N)
    train_idx = idx[:split]
    val_idx = idx[split:]

    # Baseline: train-mean
    w_base = float(np.mean(white_elo[train_idx]))
    b_base = float(np.mean(black_elo[train_idx]))
    w_base_mae = float(np.mean(np.abs(white_elo[val_idx] - w_base)))
    b_base_mae = float(np.mean(np.abs(black_elo[val_idx] - b_base)))
    base_all = 0.5 * (w_base_mae + b_base_mae)

    # Random baseline (uniform over observed Elo range)
    rng = np.random.default_rng(seed)

    elo_min = min(white_elo[train_idx].min(), black_elo[train_idx].min())
    elo_max = max(white_elo[train_idx].max(), black_elo[train_idx].max())

    w_rand = rng.uniform(elo_min, elo_max, size=len(val_idx))
    b_rand = rng.uniform(elo_min, elo_max, size=len(val_idx))

    w_rand_mae = float(np.mean(np.abs(white_elo[val_idx] - w_rand)))
    b_rand_mae = float(np.mean(np.abs(black_elo[val_idx] - b_rand)))
    rand_all = 0.5 * (w_rand_mae + b_rand_mae)

    print(
        f"[seed {seed}] Random  ValMAE: "
        f"W {w_rand_mae:.1f}  B {b_rand_mae:.1f}  ALL {rand_all:.1f}"
    )

    ds_train = EloDataset("data/features.npz", yw_all, yb_all, indices=train_idx)
    ds_val = EloDataset("data/features.npz", yw_all, yb_all, indices=val_idx)

    batch_size = 8 if N < 200 else 32
    dl_train = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, collate_fn=pad_collate
    )
    dl_val = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False, collate_fn=pad_collate
    )

    in_dim = ds_train[0][0].shape[1]
    model = EloGuesser(in_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    w_true_val = torch.tensor(white_elo[val_idx], dtype=torch.float32, device=device)
    b_true_val = torch.tensor(black_elo[val_idx], dtype=torch.float32, device=device)

    best = {
        "seed": seed,
        "best_epoch": -1,
        "best_w_mae": float("inf"),
        "best_b_mae": float("inf"),
        "best_all_mae": float("inf"),
        "baseline_w_mae": w_base_mae,
        "baseline_b_mae": b_base_mae,
        "baseline_all_mae": base_all,
        "checkpoint_path": f"checkpoints/best_seed{seed}.pt",
        "pred_csv_path": f"predictions/val_preds_seed{seed}.csv",
    }

    no_improve = 0

    print(f"\n[seed {seed}] Using device: {device}")
    print(
        f"[seed {seed}] Baseline ValMAE: W {w_base_mae:.1f}  B {b_base_mae:.1f}  ALL {base_all:.1f}"
    )
    print(
        f"[seed {seed}] in_dim: {in_dim}, train games: {len(train_idx)}, val games: {len(val_idx)}"
    )

    for epoch in range(epochs):
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
            for Xw, Mw, Xb, Mb, _, _ in dl_val:
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
            all_mae = 0.5 * (w_mae + b_mae)

        print(
            f"[seed {seed}] Epoch {epoch:02d} | TrainLoss {train_loss:.4f} | "
            f"ValMAE W {w_mae:.1f} B {b_mae:.1f} ALL {all_mae:.1f}"
        )

        # ---- early stopping / checkpoint ----
        if all_mae < best["best_all_mae"] - 0.1:
            best["best_all_mae"] = all_mae
            best["best_w_mae"] = w_mae
            best["best_b_mae"] = b_mae
            best["best_epoch"] = epoch
            no_improve = 0

            # Save checkpoint
            torch.save(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "in_dim": in_dim,
                    "bin_mids": mids,
                    "val_idx": val_idx,
                },
                best["checkpoint_path"],
            )

            # Dump predictions CSV for this best epoch
            # Format: w_true,w_pred,b_true,b_pred
            w_true_np = w_true_val.detach().cpu().numpy()
            b_true_np = b_true_val.detach().cpu().numpy()
            w_pred_np = w_pred.detach().cpu().numpy()
            b_pred_np = b_pred.detach().cpu().numpy()
            out = np.stack([w_true_np, w_pred_np, b_true_np, b_pred_np], axis=1)
            np.savetxt(
                best["pred_csv_path"],
                out,
                delimiter=",",
                header="white_true,white_pred,black_true,black_pred",
                comments="",
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                print(
                    f"[seed {seed}] Early stopping (no ALL-MAE improvement for {patience} epochs)."
                )
                break

    print(
        f"[seed {seed}] BEST epoch {best['best_epoch']} | "
        f"W {best['best_w_mae']:.1f} B {best['best_b_mae']:.1f} ALL {best['best_all_mae']:.1f} | "
        f"Baseline ALL {best['baseline_all_mae']:.1f}"
    )
    print(f"[seed {seed}] Saved checkpoint: {best['checkpoint_path']}")
    print(f"[seed {seed}] Saved val preds:   {best['pred_csv_path']}")
    return best


def main():
    # Adjust these as you like
    seeds = [0, 1, 2]  # 3-seed validation
    epochs = 25  # allow more, early stopping will cut it
    patience = 4

    results = []
    for s in seeds:
        results.append(train_one_seed(seed=s, epochs=epochs, patience=patience))

    all_maes = np.array([r["best_all_mae"] for r in results], dtype=np.float32)
    w_maes = np.array([r["best_w_mae"] for r in results], dtype=np.float32)
    b_maes = np.array([r["best_b_mae"] for r in results], dtype=np.float32)

    base_all = np.array([r["baseline_all_mae"] for r in results], dtype=np.float32)

    print("\n=== Summary over seeds ===")
    print(f"Baseline ALL MAE: {base_all.mean():.1f} ± {base_all.std(ddof=1):.1f}")
    print(f"Model    W   MAE: {w_maes.mean():.1f} ± {w_maes.std(ddof=1):.1f}")
    print(f"Model    B   MAE: {b_maes.mean():.1f} ± {b_maes.std(ddof=1):.1f}")
    print(f"Model    ALL MAE: {all_maes.mean():.1f} ± {all_maes.std(ddof=1):.1f}")

    best_idx = int(np.argmin(all_maes))
    print("\nBest seed run:")
    print(results[best_idx])


if __name__ == "__main__":
    main()
