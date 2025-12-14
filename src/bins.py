import numpy as np
from math import erf, sqrt


def make_bins(num_bins=39, lo=0, hi=4000):
    edges = np.linspace(lo, hi, num_bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    return edges, mids


def gaussian_soft_labels(elos, edges, sigma=200.0):
    def norm_cdf(x, mu, s):
        return 0.5 * (1.0 + erf((x - mu) / (s * sqrt(2.0))))

    N = len(elos)
    B = len(edges) - 1
    y = np.zeros((N, B), dtype=np.float32)

    for i, mu in enumerate(elos):
        for b in range(B):
            a, c = edges[b], edges[b + 1]
            y[i, b] = norm_cdf(c, mu, sigma) - norm_cdf(a, mu, sigma)
        y[i] /= y[i].sum() + 1e-8

    return y
