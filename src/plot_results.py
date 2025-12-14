import glob
import os
import numpy as np
import matplotlib.pyplot as plt


def load_preds(path: str):
    # CSV header: white_true,white_pred,black_true,black_pred
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    w_true, w_pred, b_true, b_pred = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    return w_true, w_pred, b_true, b_pred


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def pick_best_csv(pattern="predictions/val_preds_seed*.csv"):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files found matching {pattern}. Run training first."
        )

    best_path = None
    best_all = float("inf")

    for f in files:
        w_true, w_pred, b_true, b_pred = load_preds(f)
        all_mae = 0.5 * (mae(w_pred, w_true) + mae(b_pred, b_true))
        if all_mae < best_all:
            best_all = all_mae
            best_path = f

    return best_path, best_all


def scatter_true_vs_pred(true, pred, title, out_path):
    plt.figure()
    plt.scatter(true, pred, s=10)

    lo = min(true.min(), pred.min())
    hi = max(true.max(), pred.max())
    plt.plot([lo, hi], [lo, hi])  # y=x reference line

    plt.xlabel("True Elo")
    plt.ylabel("Predicted Elo")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def hist_abs_error(err, title, out_path):
    plt.figure()
    plt.hist(err, bins=40)
    plt.xlabel("Absolute Error (Elo)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def scatter_error_vs_true(true, err, title, out_path):
    plt.figure()
    plt.scatter(true, err, s=10)
    plt.xlabel("True Elo")
    plt.ylabel("Absolute Error (Elo)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    os.makedirs("plots", exist_ok=True)

    best_csv, best_all = pick_best_csv()
    w_true, w_pred, b_true, b_pred = load_preds(best_csv)

    w_mae = mae(w_pred, w_true)
    b_mae = mae(b_pred, b_true)
    all_mae = 0.5 * (w_mae + b_mae)

    print("Best predictions file:", best_csv)
    print(f"MAE: White {w_mae:.1f}, Black {b_mae:.1f}, ALL {all_mae:.1f}")

    # True vs Pred scatter
    scatter_true_vs_pred(
        w_true,
        w_pred,
        title=f"White Elo: True vs Pred (MAE {w_mae:.1f})",
        out_path="plots/true_vs_pred_white.png",
    )
    scatter_true_vs_pred(
        b_true,
        b_pred,
        title=f"Black Elo: True vs Pred (MAE {b_mae:.1f})",
        out_path="plots/true_vs_pred_black.png",
    )

    # Absolute error histograms
    w_err = np.abs(w_pred - w_true)
    b_err = np.abs(b_pred - b_true)
    all_err = np.concatenate([w_err, b_err], axis=0)

    hist_abs_error(
        w_err,
        title="White Absolute Error Histogram",
        out_path="plots/abs_error_hist_white.png",
    )
    hist_abs_error(
        b_err,
        title="Black Absolute Error Histogram",
        out_path="plots/abs_error_hist_black.png",
    )
    hist_abs_error(
        all_err,
        title="Overall Absolute Error Histogram",
        out_path="plots/abs_error_hist_all.png",
    )

    # Error vs true (bias / where it struggles)
    scatter_error_vs_true(
        w_true,
        w_err,
        title="White |Error| vs True Elo",
        out_path="plots/error_vs_true_white.png",
    )
    scatter_error_vs_true(
        b_true,
        b_err,
        title="Black |Error| vs True Elo",
        out_path="plots/error_vs_true_black.png",
    )

    print("Saved plots to ./plots/")


if __name__ == "__main__":
    main()
