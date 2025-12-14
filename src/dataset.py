import numpy as np
import torch
from torch.utils.data import Dataset


class EloDataset(Dataset):
    """
    Loads variable-length sequences from features.npz and provides
    (Xw, Xb, yw_dist, yb_dist). Supports selecting a subset via indices.
    """

    def __init__(self, npz_path: str, yw: np.ndarray, yb: np.ndarray, indices=None):
        d = np.load(npz_path, allow_pickle=True)
        self.Xw = d["X_white"]
        self.Xb = d["X_black"]
        self.yw = yw
        self.yb = yb
        self.indices = np.arange(len(self.yw)) if indices is None else np.array(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        return (
            self.Xw[idx].astype(np.float32),
            self.Xb[idx].astype(np.float32),
            self.yw[idx].astype(np.float32),
            self.yb[idx].astype(np.float32),
        )


def pad_collate(batch):
    Xw_list, Xb_list, yw_list, yb_list = zip(*batch)

    def pad(seq_list):
        lens = [s.shape[0] for s in seq_list]
        F = seq_list[0].shape[1]
        T = max(lens)

        x = torch.zeros(len(seq_list), T, F, dtype=torch.float32)
        m = torch.zeros(len(seq_list), T, dtype=torch.bool)

        for i, s in enumerate(seq_list):
            t = s.shape[0]
            x[i, :t] = torch.from_numpy(s)
            m[i, :t] = True

        return x, m

    Xw, Mw = pad(Xw_list)
    Xb, Mb = pad(Xb_list)

    yw = torch.from_numpy(np.stack(yw_list)).float()
    yb = torch.from_numpy(np.stack(yb_list)).float()

    return Xw, Mw, Xb, Mb, yw, yb
