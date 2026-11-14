"""Pure-helper tests for the nnsight lab (nnsight itself is not required)."""

import numpy as np
import torch

from src.explainability.nnsight_lab import _sig, linear_cka, pool_point, scatter_mean


def test_scatter_mean_matches_loop():
    x = torch.randn(7, 4)
    idx = torch.tensor([0, 0, 1, 2, 2, 2, 4])
    out = scatter_mean(x, idx, 5)
    assert torch.allclose(out[0], x[:2].mean(0))
    assert torch.allclose(out[2], x[3:6].mean(0))
    assert torch.allclose(out[3], torch.zeros(4))  # empty group -> 0
    assert torch.allclose(out[4], x[6])


def test_pool_point_takes_last_window_and_averages_types():
    L, n_graphs = 3, 6  # 2 examples x 3 windows
    xa, xb = torch.randn(6, 4), torch.randn(6, 4)
    ng = {"network": torch.arange(6), "host": torch.arange(6)}
    out = pool_point("hgt_conv1", {"network": xa, "host": xb}, ng, n_graphs, L)
    assert out.shape == (2, 4)
    assert torch.allclose(out[0], (xa[2] + xb[2]) / 2)
    assert torch.allclose(out[1], (xa[5] + xb[5]) / 2)
    assert torch.equal(pool_point("spatial_out", xa, None, n_graphs, L), xa[2::3])
    h = torch.randn(2, 2, 4)
    assert torch.equal(pool_point("gru", (None, h), None, n_graphs, L), h[-1])


def test_linear_cka_identity_and_invariance():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8))
    Q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    assert abs(linear_cka(X, X) - 1.0) < 1e-9
    assert abs(linear_cka(X, 3.0 * X @ Q) - 1.0) < 1e-6  # invariant to rotation and scale
    assert linear_cka(X, rng.normal(size=(200, 8))) < 0.3


def test_sigmoid_temperature():
    assert abs(_sig(0.0) - 0.5) < 1e-12
    assert _sig(2.0, 2.0) == _sig(1.0, 1.0)
