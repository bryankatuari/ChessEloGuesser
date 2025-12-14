import torch
import torch.nn as nn


class PlayerEncoder(nn.Module):
    def __init__(self, in_dim, hidden=64, out_bins=39):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.rnn = nn.GRU(in_dim, hidden, bidirectional=True, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, out_bins),
        )

    def forward(self, x, mask):
        x = self.norm(x)
        h, _ = self.rnn(x)
        mask = mask.unsqueeze(-1)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return torch.softmax(self.head(pooled), dim=-1)


class EloGuesser(nn.Module):
    def __init__(self, in_dim, hidden=64, out_bins=39):
        super().__init__()
        self.encoder = PlayerEncoder(in_dim, hidden, out_bins)

    def forward(self, Xw, Mw, Xb, Mb):
        return self.encoder(Xw, Mw), self.encoder(Xb, Mb)
