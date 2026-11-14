"""Plots for the nnsight lab (src/explainability/nnsight_lab.py).

Every function takes plain numpy / dict inputs (the lab's return values or the
JSON written by train_interpretability) and returns a matplotlib Figure, so it
works in a notebook, in a script (fig.savefig) and in tests. `embedding_plotly`
is the one interactive (hover) view.
"""

from __future__ import annotations

import numpy as np

from src.explainability.nnsight_lab import linear_cka
from src.scoring.attack_stage import STAGE_NAMES

LAYER_NAMES = {
    "input": "raw input", "hgt_conv1": "HGT 1", "hgt_conv2": "HGT 2", "spatial_out": "spatial out", "gru": "GRU",
    "context_norm": "z_t (final)", "predictor_l1": "predictor L1", "predictor_l2": "predictor L2", "predictor_l3": "predictor L3 (z_t+1)",
}


def _plt():
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
    return plt


def _label_colors(labels):
    plt = _plt()
    uniq = sorted(set(map(str, labels)), key=lambda s: (s != "BENIGN", s))
    cmap = plt.get_cmap("tab20")
    return {u: ("#9aa0a6" if u == "BENIGN" else cmap(i % 20)) for i, u in enumerate(uniq)}


def _project(Z, method="pca", seed=0, max_points=4000):
    from sklearn.decomposition import PCA

    idx = np.arange(len(Z))
    if len(Z) > max_points:
        idx = np.random.default_rng(seed).choice(len(Z), max_points, replace=False)
    X = Z[idx]
    if method == "tsne":
        from sklearn.manifold import TSNE

        P = TSNE(n_components=2, init="pca", perplexity=30, random_state=seed).fit_transform(PCA(min(30, X.shape[1])).fit_transform(X))
        return idx, P, None
    pca = PCA(2, random_state=seed).fit(X)
    return idx, pca.transform(X), pca.explained_variance_ratio_


# ── 1. embedding space ──────────────────────────────────────────────────────
def embedding_overview(Z, labels, stage, p_attack, method="pca"):
    """Three maps of the same embedding: by true attack family, by MITRE stage, by P(attack)."""
    plt = _plt()
    idx, P, var = _project(Z, method)
    labels, stage, p = np.asarray(labels)[idx].astype(str), np.asarray(stage)[idx], np.asarray(p_attack)[idx]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    cols = _label_colors(labels)
    for name, c in cols.items():
        m = labels == name
        ax[0].scatter(P[m, 0], P[m, 1], s=7, color=c, alpha=0.6, label=f"{name} ({m.sum()})")
    ax[0].legend(markerscale=2, fontsize=7, loc="best"); ax[0].set_title("true attack family")
    sc_cols = plt.get_cmap("tab10")
    for s in sorted(set(stage.tolist())):
        m = stage == s
        ax[1].scatter(P[m, 0], P[m, 1], s=7, color="#cccccc" if s < 0 else sc_cols(s % 10), alpha=0.6, label="n/a" if s < 0 else STAGE_NAMES[s])
    ax[1].legend(markerscale=2, fontsize=7); ax[1].set_title("MITRE stage label")
    h = ax[2].scatter(P[:, 0], P[:, 1], s=7, c=p, cmap="magma", vmin=0, vmax=1, alpha=0.7)
    plt.colorbar(h, ax=ax[2], label="P(infiltration)"); ax[2].set_title("model score")
    for a in ax:
        a.set_xlabel("PC1" if method == "pca" else "dim 1"); a.set_ylabel("PC2" if method == "pca" else "dim 2")
    if var is not None:
        fig.suptitle(f"z_t embedding, PCA (explains {var.sum():.0%} of variance)")
    fig.tight_layout()
    return fig


def embedding_plotly(Z, labels, p_attack, method="pca", extra_hover=None):
    """Interactive scatter (hover shows label, score). Returns a plotly Figure."""
    import plotly.express as px

    idx, P, _ = _project(Z, method)
    hover = {"label": np.asarray(labels)[idx].astype(str), "p_attack": np.round(np.asarray(p_attack)[idx], 3), "x": P[:, 0], "y": P[:, 1]}
    if extra_hover is not None:
        hover["info"] = np.asarray(extra_hover)[idx]
    import pandas as pd

    return px.scatter(pd.DataFrame(hover), x="x", y="y", color="label", hover_data=[c for c in hover if c not in ("x", "y")], opacity=0.7,
                      title="z_t embedding (interactive)")


def capture_timeline(p_cur, p_fut, labels, stage_pred, threshold=0.5):
    """Score over time with the ground-truth attack ribbon and predicted stage ribbon."""
    plt = _plt()
    n = len(p_cur)
    fig, ax = plt.subplots(3, 1, figsize=(15, 6.5), sharex=True, gridspec_kw={"height_ratios": [4, 0.6, 0.6]})
    ax[0].plot(p_cur, lw=1.2, label="P(attack) now"); ax[0].plot(p_fut, lw=1.0, alpha=0.8, label="forecast t+1")
    ax[0].axhline(threshold, color="k", ls="--", lw=0.8); ax[0].set_ylabel("probability"); ax[0].legend(loc="upper right")
    cols = _label_colors(labels)
    for i, l in enumerate(np.asarray(labels).astype(str)):
        ax[1].axvspan(i, i + 1, color=cols[l], lw=0)
    ax[1].set_yticks([]); ax[1].set_ylabel("truth", rotation=0, ha="right"); ax[1].grid(False)
    cm = plt.get_cmap("tab10")
    for i, s in enumerate(stage_pred):
        ax[2].axvspan(i, i + 1, color=cm(int(s) % 10), lw=0)
    ax[2].set_yticks([]); ax[2].set_ylabel("pred. stage", rotation=0, ha="right"); ax[2].set_xlim(0, n); ax[2].set_xlabel("window"); ax[2].grid(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cols.values()]
    ax[1].legend(handles, cols.keys(), ncol=min(8, len(cols)), fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 2.2))
    fig.tight_layout()
    return fig


# ── 2. layer-wise views ─────────────────────────────────────────────────────
def lens_trajectory_plot(lens: dict, labels):
    """Mean P(attack) at every layer per attack family: where does the model 'decide'?"""
    plt = _plt()
    pts = lens["points"]
    labels = np.asarray(labels).astype(str)
    cols = _label_colors(labels)
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
    for name, c in cols.items():
        m = labels == name
        if m.sum() < 5:
            continue
        mean = [lens["traj"][p]["p"][m].mean() for p in pts]
        ax[0].plot(range(len(pts)), mean, "-o", color=c, label=f"{name} ({m.sum()})", lw=2 if name == "BENIGN" else 1.5)
    ax[0].set_xticks(range(len(pts))); ax[0].set_xticklabels([LAYER_NAMES.get(p, p) for p in pts], rotation=25, ha="right")
    ax[0].set_ylabel("mean P(infiltration) read at this layer"); ax[0].set_ylim(-0.02, 1.02); ax[0].legend(fontsize=7)
    ax[0].set_title("tuned lens: when does each family become 'attack'?")
    sep = []
    for p in pts:
        pr = lens["traj"][p]["p"]
        b, a = pr[labels == "BENIGN"], pr[labels != "BENIGN"]
        sep.append(float(a.mean() - b.mean()) if len(a) and len(b) else np.nan)
    ax[1].bar(range(len(pts)), sep, color="#4c78a8"); ax[1].set_xticks(range(len(pts)))
    ax[1].set_xticklabels([LAYER_NAMES.get(p, p) for p in pts], rotation=25, ha="right")
    ax[1].set_ylabel("mean P(attack): attacks - benign"); ax[1].set_title("separation grows through the network")
    fig.tight_layout()
    return fig


def cka_heatmap(acts: dict, points=None, max_n=3000):
    """Linear CKA between layers: which layers compute similar things?"""
    plt = _plt()
    points = points or [p for p in ["input", "hgt_conv1", "hgt_conv2", "spatial_out", "gru", "context_norm", "predictor_l1", "predictor_l2", "predictor_l3"] if p in acts]
    n = min(max_n, len(acts[points[0]]))
    M = np.array([[linear_cka(acts[a][:n], acts[b][:n]) for b in points] for a in points])
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
    names = [LAYER_NAMES.get(p, p) for p in points]
    ax.set_xticks(range(len(points))); ax.set_xticklabels(names, rotation=40, ha="right"); ax.set_yticks(range(len(points))); ax.set_yticklabels(names)
    for i in range(len(points)):
        for j in range(len(points)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w" if M[i, j] < 0.6 else "k", fontsize=8)
    ax.grid(False); plt.colorbar(im, label="linear CKA"); ax.set_title("representation similarity between layers")
    fig.tight_layout()
    return fig


def probe_heatmap(probes: dict):
    """Linear-probe balanced accuracy, concept x layer, next to the gain over each
    concept's own chance level (1/classes; a 6-class concept is not at chance at 0.5)."""
    plt = _plt()
    layers, rows = probes["layers"], probes["concepts"]
    M = np.array([[r["layers"].get(l, {}).get("balanced_accuracy", np.nan) for l in layers] for r in rows])
    chance = np.array([1.0 / max(r.get("classes", 2), 2) for r in rows])
    G = M - chance[:, None]
    fig, ax = plt.subplots(1, 2, figsize=(2.6 * len(layers) + 5, 0.5 * len(rows) + 2.5), sharey=True)
    for a, data, vmin, vmax, cmap, title, fmt in ((ax[0], M, 0.0, 1.0, "viridis", "balanced accuracy", "{:.2f}"),
                                                   (ax[1], G, 0.0, 0.5, "magma", "gain over chance", "{:+.2f}")):
        im = a.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
        a.set_xticks(range(len(layers))); a.set_xticklabels([LAYER_NAMES.get(l, l) for l in layers], rotation=30, ha="right")
        a.set_yticks(range(len(rows))); a.set_yticklabels([f"{r['concept']} (chance {c:.2f})" for r, c in zip(rows, chance)])
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if not np.isnan(data[i, j]):
                    v = (data[i, j] - vmin) / (vmax - vmin)
                    a.text(j, i, fmt.format(data[i, j]), ha="center", va="center", color="w" if v < 0.6 else "k", fontsize=8)
        a.grid(False); plt.colorbar(im, ax=a); a.set_title(f"linear probes: {title}")
    fig.tight_layout()
    return fig


# ── 3. causal views (nnsight interventions) ─────────────────────────────────
def patching_heatmap(grid: dict):
    """Layer x subspace causal-effect map from Lab.patching_grid()."""
    plt = _plt()
    pts = [LAYER_NAMES.get(p, p) for p in grid["points"]]
    fig, ax = plt.subplots(1, 2, figsize=(16, 4.8), sharey=True)
    for a, key, title in ((ax[0], "cur", "effect on CURRENT score"), (ax[1], "fut", "effect on t+1 FORECAST")):
        im = a.imshow(np.nan_to_num(grid[key], nan=0.0), vmin=0, vmax=1, cmap="inferno", aspect="auto")
        a.set_xticks(range(grid[key].shape[1])); a.set_xlabel("embedding subspace (block of dims)")
        a.set_yticks(range(len(pts))); a.set_yticklabels(pts); a.grid(False); a.set_title(title)
        for i in range(grid[key].shape[0]):
            for j in range(grid[key].shape[1]):
                v = grid[key][i, j]
                if not np.isnan(v):
                    a.text(j, i, f"{v:.2f}", ha="center", va="center", color="w" if v < 0.6 else "k", fontsize=7)
        plt.colorbar(im, ax=a, label="share of gap closed")
    fig.suptitle(f"activation patching (benign -> flagged), {grid['n_windows']} windows; valid gaps: now={grid['n_valid_cur']}, t+1={grid['n_valid_fut']}")
    fig.tight_layout()
    return fig


def node_type_effects(grid: dict):
    plt = _plt()
    nt = grid["node_type"]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    width = 0.35
    types = sorted({t for d in nt.values() for t in d})
    for i, t in enumerate(types):
        ax.bar(np.arange(len(nt)) + i * width, [nt[p].get(t, np.nan) for p in nt], width, label=t)
    ax.set_xticks(np.arange(len(nt)) + width / 2); ax.set_xticklabels([LAYER_NAMES.get(p, p) for p in nt])
    ax.set_ylabel("share of gap closed"); ax.set_ylim(0, 1); ax.legend(title="node type"); ax.set_title("graph layers: network node vs host nodes")
    fig.tight_layout()
    return fig


def dim_importance(abl: dict, z_by_label: dict | None = None, top=30):
    """Which embedding dimensions the decision relies on (+ their class means)."""
    plt = _plt()
    cur, fut = abl["cur"], abl["fut"]
    order = np.argsort(-cur)[:top]
    ncol = 3 if z_by_label else 2
    fig, ax = plt.subplots(1, ncol, figsize=(6 * ncol, 5))
    ax[0].bar(range(len(cur)), np.sort(cur)[::-1], color="#4c78a8"); ax[0].set_xlabel("dimension (ranked)"); ax[0].set_ylabel("mean |dP| when mean-ablated")
    ax[0].set_title("embedding dimension importance (current score)")
    ax[1].scatter(cur, fut, s=14, alpha=0.7)
    for d in order[:8]:
        ax[1].annotate(str(d), (cur[d], fut[d]), fontsize=8)
    ax[1].set_xlabel("importance for current score"); ax[1].set_ylabel("importance for forecast"); ax[1].set_title("same dimensions drive now and next?")
    if z_by_label:
        names = list(z_by_label)
        M = np.array([z_by_label[n][order] for n in names])
        im = ax[2].imshow(M, cmap="coolwarm", aspect="auto", vmin=-np.abs(M).max(), vmax=np.abs(M).max())
        ax[2].set_yticks(range(len(names))); ax[2].set_yticklabels(names, fontsize=7)
        ax[2].set_xticks(range(len(order))); ax[2].set_xticklabels(order, rotation=90, fontsize=7); ax[2].grid(False)
        plt.colorbar(im, ax=ax[2], label="mean z"); ax[2].set_title(f"top-{top} dimensions: mean value per family")
    fig.tight_layout()
    return fig


# ── 4. SAE views ────────────────────────────────────────────────────────────
def sae_overview(sae, Z, labels, labels_json: dict, top_per_class=6):
    """Activation frequency, dead features, class x feature heatmap, decoder similarity."""
    import torch

    plt = _plt()
    with torch.no_grad():
        H = sae.encode(torch.tensor(Z, dtype=torch.float32)).numpy()
    labels = np.asarray(labels).astype(str)
    freq = (H > 0).mean(0)
    alive = int((freq > 0).sum())
    fig, ax = plt.subplots(2, 2, figsize=(16, 11))
    ax[0, 0].hist(np.log10(freq[freq > 0]), bins=40, color="#4c78a8")
    ax[0, 0].set_xlabel("log10 activation frequency"); ax[0, 0].set_ylabel("# features")
    ax[0, 0].set_title(f"{alive}/{H.shape[1]} features alive ({H.shape[1] - alive} dead)")
    classes = sorted(set(labels), key=lambda s: (s != "BENIGN", s))
    means = np.array([H[labels == c].mean(0) for c in classes])
    feats = []
    for i, c in enumerate(classes):
        spec = means[i] - np.delete(means, i, axis=0).mean(0) if len(classes) > 1 else means[i]
        feats += [int(f) for f in np.argsort(-spec)[:top_per_class] if f not in feats]
    M = means[:, feats]
    im = ax[0, 1].imshow(M / (M.max(0, keepdims=True) + 1e-9), aspect="auto", cmap="magma")
    ax[0, 1].set_yticks(range(len(classes))); ax[0, 1].set_yticklabels(classes, fontsize=7)
    ax[0, 1].set_xticks(range(len(feats))); ax[0, 1].set_xticklabels(feats, rotation=90, fontsize=6); ax[0, 1].grid(False)
    ax[0, 1].set_title("most class-specific SAE features (column-normalised mean activation)"); plt.colorbar(im, ax=ax[0, 1])
    W = sae.decoder.weight.detach()[:, feats].numpy()
    W = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-9)
    im2 = ax[1, 0].imshow(W.T @ W, vmin=-1, vmax=1, cmap="coolwarm")
    ax[1, 0].set_xticks(range(len(feats))); ax[1, 0].set_xticklabels(feats, rotation=90, fontsize=6)
    ax[1, 0].set_yticks(range(len(feats))); ax[1, 0].set_yticklabels(feats, fontsize=6); ax[1, 0].grid(False)
    ax[1, 0].set_title("decoder direction cosine similarity"); plt.colorbar(im2, ax=ax[1, 0])
    purity = [(v.get("purity", 0), v.get("dominant_label", "?")) for v in labels_json.values()]
    if purity:
        ax[1, 1].hist([p for p, _ in purity], bins=20, color="#59a14f")
        ax[1, 1].set_xlabel("label purity of top-20 activating windows"); ax[1, 1].set_ylabel("# features")
        ax[1, 1].set_title("how monosemantic are the features?")
    fig.tight_layout()
    return fig, H


def sae_raster(H, labels, n_features=40, max_windows=400):
    """Which SAE features fire in which windows (sorted by dominant class)."""
    plt = _plt()
    labels = np.asarray(labels).astype(str)
    n = min(max_windows, len(H))
    top = np.argsort(-(H[:n] > 0).sum(0))[:n_features]
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.imshow((H[:n][:, top]).T, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top, fontsize=6); ax.set_xlabel("window"); ax.set_ylabel("SAE feature"); ax.grid(False)
    cols = _label_colors(labels)
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, -1.8), 1, 1.2, color=cols[labels[i]], clip_on=False))
    ax.set_title("SAE feature activations over a capture (colour bar on top = true label)")
    fig.tight_layout()
    return fig


def sae_effects_bar(eff: dict, title="causal effect of ablating each SAE feature"):
    plt = _plt()
    rows = sorted(eff["rows"], key=lambda r: r["delta_cur"])
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(rows) + 1.5))
    y = np.arange(len(rows))
    ax.barh(y - 0.2, [r["delta_cur"] for r in rows], 0.4, label="current score", color="#e45756")
    ax.barh(y + 0.2, [r["delta_fut"] for r in rows], 0.4, label="t+1 forecast", color="#4c78a8")
    ax.set_yticks(y); ax.set_yticklabels([f"#{r['feature']} {r['label'][:48]}" for r in rows], fontsize=8)
    ax.axvline(0, color="k", lw=0.8); ax.set_xlabel(f"change in mean P(attack)  (baseline now={eff['base_cur']:.2f}, t+1={eff['base_fut']:.2f})")
    ax.legend(); ax.set_title(title)
    fig.tight_layout()
    return fig


def steering_curves(curves: dict):
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    for f, c in curves.items():
        ax[0].plot(c["alphas"], c["cur"], "-o", label=f"#{f} {c['label'][:36]}")
        ax[1].plot(c["alphas"], c["fut"], "-o")
    ax[0].set_title("steering benign windows: current score"); ax[1].set_title("steering benign windows: t+1 forecast")
    for a in ax:
        a.set_xlabel("steering strength (x typical activation)"); a.set_ylim(-0.02, 1.02)
    ax[0].set_ylabel("mean P(attack)"); ax[0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def sae_fidelity_bar(fid: dict):
    plt = _plt()
    names = list(fid)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].bar(names, [fid[n]["decision_agreement"] for n in names], color=["#59a14f", "#f28e2b", "#e15759"])
    ax[0].set_ylim(0, 1.02); ax[0].set_title("decision agreement with the real z_t")
    ax[1].bar(names, [fid[n]["mean_abs_dp"] for n in names], color=["#59a14f", "#f28e2b", "#e15759"]); ax[1].set_title("mean |change in P(attack)|")
    for a in ax:
        a.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return fig


# ── 5. single-window views ──────────────────────────────────────────────────
def attribution_bar(attr: dict, top=15):
    plt = _plt()
    a, names = attr["attr"], attr["names"]
    order = np.argsort(-np.abs(a))[:top][::-1]
    fig, ax = plt.subplots(figsize=(8, 0.35 * top + 1.5))
    ax.barh([names[i] for i in order], a[order], color=["#e45756" if a[i] > 0 else "#4c78a8" for i in order])
    ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("(x - x_ref) * dlogit/dx  (red pushes toward attack)")
    ax.set_title(f"input attribution (logit {attr['logit']:+.2f})")
    fig.tight_layout()
    return fig


def rollout_plot(bands: dict, p_now: float | None = None):
    plt = _plt()
    k = np.arange(1, len(bands["p50"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
    ax[0].fill_between(k, bands["p10"], bands["p90"], alpha=0.25, label="p10-p90"); ax[0].plot(k, bands["p50"], "-o", label="median")
    if p_now is not None:
        ax[0].axhline(p_now, color="gray", ls=":", label="now")
    ax[0].set_ylim(0, 1); ax[0].set_xlabel("steps ahead"); ax[0].set_ylabel("P(infiltration)"); ax[0].legend()
    ax[0].set_title(f"forecast band (P(attack within horizon) = {bands['p_any_attack']:.0%})")
    im = ax[1].imshow(bands["stage_mean"].T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax[1].set_yticks(range(len(STAGE_NAMES))); ax[1].set_yticklabels(STAGE_NAMES, fontsize=8); ax[1].set_xticks(range(len(k))); ax[1].set_xticklabels(k)
    ax[1].set_xlabel("steps ahead"); ax[1].grid(False); ax[1].set_title("predicted MITRE stage distribution"); plt.colorbar(im, ax=ax[1])
    fig.tight_layout()
    return fig


# ── 6. model-level quality ──────────────────────────────────────────────────
def confusion_plot(stage_true, stage_pred):
    from sklearn.metrics import confusion_matrix

    plt = _plt()
    m = np.asarray(stage_true) >= 0
    C = confusion_matrix(np.asarray(stage_true)[m], np.asarray(stage_pred)[m], labels=range(len(STAGE_NAMES)))
    Cn = C / np.maximum(C.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(Cn, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(STAGE_NAMES))); ax.set_xticklabels(STAGE_NAMES, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(STAGE_NAMES))); ax.set_yticklabels(STAGE_NAMES, fontsize=8); ax.grid(False)
    for i in range(len(STAGE_NAMES)):
        for j in range(len(STAGE_NAMES)):
            ax.text(j, i, f"{Cn[i, j]:.2f}\n({C[i, j]})", ha="center", va="center", fontsize=7, color="w" if Cn[i, j] > 0.5 else "k")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("stage head confusion (row-normalised)"); plt.colorbar(im)
    fig.tight_layout()
    return fig


def calibration_plot(p, y, bins=10):
    plt = _plt()
    p, y = np.asarray(p), np.asarray(y)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    conf = [p[idx == b].mean() if (idx == b).any() else np.nan for b in range(bins)]
    acc = [y[idx == b].mean() if (idx == b).any() else np.nan for b in range(bins)]
    cnt = [(idx == b).sum() for b in range(bins)]
    ece = float(sum(c * abs(a - f) for c, a, f in zip(cnt, acc, conf) if c) / max(len(p), 1))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8); ax[0].plot(conf, acc, "-o"); ax[0].set_xlabel("predicted probability"); ax[0].set_ylabel("observed attack rate")
    ax[0].set_title(f"reliability diagram (ECE {ece:.3f})")
    ax[1].hist([p[y == 0], p[y == 1]], bins=bins, range=(0, 1), label=["benign", "attack"], stacked=True, color=["#9aa0a6", "#e45756"])
    ax[1].set_yscale("log"); ax[1].set_xlabel("predicted probability"); ax[1].legend(); ax[1].set_title("score distribution")
    fig.tight_layout()
    return fig
