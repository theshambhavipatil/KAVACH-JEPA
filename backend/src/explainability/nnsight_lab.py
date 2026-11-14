"""Mechanistic-interpretability lab for NetJEPA, built on nnsight.

nnsight wraps any torch.nn.Module and lets us *read* and *write* the output
of any submodule during a forward pass ("trace"). That is the primitive behind
activation patching, ablation and steering. This module re-implements the
repo's hand-rolled hook code (model_forward.capture / patch_partial) on top of
nnsight and extends it:

* collect()          batched, pooled activations at every hook point
* patching_grid()    layer x subspace causal-effect map, averaged over many windows
* dim_ablation()     which of the 128 embedding dimensions the decision depends on
* sae_effects()      causal effect of ablating individual SAE features
* sae_steering()     push a benign window along an SAE feature and watch the score
* sae_fidelity()     does substituting the SAE reconstruction change decisions?
* lens_trajectory()  tuned-lens read-out of every layer (plain torch)
* rollout_bands()    Monte-Carlo forecast bands from the probabilistic predictor
* grad_x_input()     input attribution through the real HGT path
* selfcheck()        proves the nnsight activations equal the torch-hook ones

Design notes
------------
* The traced module is `Joint`, bundling NetJEPA + both scoring heads so the
  final logits are inside the traced graph and interventions can be read out
  end to end.
* Graph inputs (lists of PyG HeteroData) are NOT passed through nnsight's
  argument handling. They are stored on the module before the trace and the
  trace is called with a dummy tensor, keeping nnsight out of the data path.
* Windows go through the batched PyG path (one HGT call per batch), so node
  rows are mapped back to windows with the Batch node->graph index. Only the
  LAST window of each example's context is patched; history is untouched.
* Everything returned is plain numpy / python so src.explainability.viz can
  plot it without knowing about nnsight.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.explainability import model_forward as mf
from src.graph.state_builder import CANONICAL_FEATURE_NAMES, build_graph_state

HOOKS = ["hgt_conv1", "hgt_conv2", "spatial_out", "gru", "context_norm", "predictor_l1", "predictor_l2", "predictor_l3"]
PREDICTOR_HOOKS = {"predictor_l1", "predictor_l2", "predictor_l3"}
INTERP_DIR = Path("data/processed/interpretability")


class Joint(nn.Module):
    """NetJEPA + infiltration head + stage head as ONE traced module.
    Returns (cur_logit, fut_logit, stage_now, stage_fut); "fut" is the
    predictor's t+1 embedding read by the same heads."""

    def __init__(self, model, infil_head, stage_clf):
        super().__init__()
        self.net, self.ih, self.sc = model, infil_head, stage_clf
        self._seqs: list = []

    def forward(self, _dummy=None):
        z_pred, _, z = self.net(self._seqs)
        return mf.infil_logit(self.ih, z), mf.infil_logit(self.ih, z_pred), self.sc(z), self.sc(z_pred)


# ── pure helpers (no nnsight; unit-tested) ──────────────────────────────────
def scatter_mean(x: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Mean of rows of x grouped by `index` into n groups (empty group -> 0)."""
    sums = torch.zeros(n, x.shape[1], dtype=x.dtype, device=x.device).index_add_(0, index, x)
    counts = torch.bincount(index, minlength=n).clamp(min=1).to(x.dtype).unsqueeze(1)
    return sums / counts


def pool_point(point: str, raw, node_graph: dict | None, n_graphs: int, L: int) -> torch.Tensor:
    """Reduce a raw hook output to one (N, dim) row per example, the way NetJEPA
    pools (mean over a type's nodes, then over node types), taking the LAST
    window of each example for the spatial layers."""
    if point in ("hgt_conv1", "hgt_conv2"):
        per_type = [scatter_mean(v, node_graph[k], n_graphs) for k, v in raw.items()]
        return torch.stack(per_type).mean(0)[L - 1 :: L]
    if point == "spatial_out":
        return raw[L - 1 :: L]
    if point == "gru":
        return raw[1][-1]
    return raw


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al. 2019) between two (N, d) representations."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    num = np.linalg.norm(Y.T @ X, "fro") ** 2
    den = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(num / den) if den > 0 else 0.0


def _val(x):
    """nnsight versions differ: saved values are proxies with .value, or already real."""
    return x.value if hasattr(x, "value") and not isinstance(x, torch.Tensor) else x


def _detach(x):
    if isinstance(x, dict):
        return {k: _detach(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return tuple(_detach(v) for v in x)
    return x.detach().cpu() if isinstance(x, torch.Tensor) else x


def _sig(x, T=1.0):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64) / (T or 1.0)))


def _last_dim(point: str, raw) -> int:
    v = raw[point]
    if point == "gru":
        return v[1][-1].shape[-1]
    if isinstance(v, dict):
        return next(iter(v.values())).shape[-1]
    return v.shape[-1]


# ── the lab ─────────────────────────────────────────────────────────────────
class Lab:
    def __init__(self, model, infil_head, stage_clf, temperature: float | None = None, context_len: int = 3,
                 sae=None, sae_labels: dict | None = None, translators: dict | None = None, x_ref: np.ndarray | None = None):
        from nnsight import NNsight  # lazy: the helpers above work without nnsight

        self.model, self.ih, self.sc = model.eval(), infil_head.eval(), stage_clf.eval()
        self.joint = Joint(self.model, self.ih, self.sc).eval()
        self.nn = NNsight(self.joint)
        self.T = temperature or 1.0
        self.L = context_len
        self.sae, self.sae_labels, self.translators, self.x_ref = sae, sae_labels or {}, translators, x_ref
        self.meta: dict = {}

    @classmethod
    def from_checkpoint(cls, ckpt_path, interp_dir=INTERP_DIR) -> "Lab":
        from src.explainability.sparse_autoencoder import SparseAutoencoder
        from src.models.netjepa import NetJEPA
        from src.orchestrator.model_registry import _pick
        from src.scoring.attack_stage import ATTACKStageClassifier
        from src.scoring.infiltration import InfiltrationHead

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        emb, hid = ck.get("embedding_dim", 128), ck.get("predictor_hidden_dim", 256)
        model = NetJEPA(ck.get("feature_dim", len(CANONICAL_FEATURE_NAMES)), emb, hid, probabilistic=bool(ck.get("probabilistic", True)))
        ih, sc = InfiltrationHead(emb), ATTACKStageClassifier(emb)
        model.load_state_dict(_pick(ck, "model")); ih.load_state_dict(_pick(ck, "infil_head")); sc.load_state_dict(_pick(ck, "stage_clf"))
        d = Path(interp_dir)
        sae, labels, trans, xref = None, {}, None, None
        if (d / "sae_weights.pt").exists():
            s = torch.load(d / "sae_weights.pt", map_location="cpu", weights_only=False)
            sae = SparseAutoencoder(s["input_dim"], s["hidden_dim"], s.get("k", 0)); sae.load_state_dict(s["state_dict"]); sae.eval()
        if (d / "sae_labels.json").exists():
            labels = json.loads((d / "sae_labels.json").read_text())
        if (d / "tuned_lens.pt").exists():
            trans = torch.load(d / "tuned_lens.pt", map_location="cpu", weights_only=False)
        if (d / "benign_reference.json").exists():
            xref = np.asarray(json.loads((d / "benign_reference.json").read_text())["feature_vector"], dtype=np.float32)
        lab = cls(model, ih, sc, ck.get("temperature"), ck.get("context_len", 3), sae, labels, trans, xref)
        lab.meta = {k: ck.get(k) for k in ("decision_threshold", "stage_temperature", "validation_metrics", "epoch", "horizon")}
        return lab

    # -- plumbing -----------------------------------------------------------
    def _envoy(self, point: str):
        enc = self.nn.net.context_encoder
        return {
            "hgt_conv1": enc.spatial.conv1, "hgt_conv2": enc.spatial.conv2, "spatial_out": enc.spatial.out,
            "gru": enc.temporal.gru, "context_norm": self.nn.net.context_norm,
            "predictor_l1": self.nn.net.predictor.net[1], "predictor_l2": self.nn.net.predictor.net[3],
            "predictor_l3": self.nn.net.predictor.net[4],
        }[point]

    @staticmethod
    def _node_graph(seqs) -> tuple[dict, int]:
        from torch_geometric.data import Batch

        flat = [g for s in seqs for g in s]
        b = Batch.from_data_list(flat)
        return {nt: b[nt].batch for nt in b.node_types}, len(flat)

    @torch.no_grad()
    def run(self, seqs: list, save=("context_norm",), patches: dict | None = None, keep_raw: bool = False) -> dict:
        """One nnsight trace over a batch of example sequences.
        patches: {hook_point: fn(envoy)}; each fn mutates `envoy.output` in
        place. Edits are applied in forward order, then values are saved."""
        patches = patches or {}
        self.joint._seqs = seqs
        pts = [p for p in HOOKS if p in save or p in patches]
        saved = {}
        with self.nn.trace(torch.zeros(1)):
            for p in pts:
                env = self._envoy(p)
                if p in patches:
                    patches[p](env)
                if p in save:
                    saved[p] = env.output.save()
            out = self.nn.output.save()
        out = [_val(o).detach().cpu() for o in _val(out)]
        raw = {p: _detach(_val(v)) for p, v in saved.items()}
        node_graph, n_graphs = self._node_graph(seqs)
        res = {
            "cur_logit": out[0].numpy(), "fut_logit": out[1].numpy(), "stage_now": out[2].numpy(), "stage_fut": out[3].numpy(),
            "pooled": {p: pool_point(p, v, node_graph, n_graphs, self.L).numpy() for p, v in raw.items()},
            "node_graph": node_graph, "n_graphs": n_graphs,
        }
        if keep_raw:
            res["raw"] = raw
        return res

    def seqs(self, cache, examples) -> list:
        return [[cache.get(ex, j) for j in range(self.L)] for ex in examples]

    def _probs(self, res: dict):
        return _sig(res["cur_logit"], self.T), _sig(res["fut_logit"], self.T)

    # -- 1. bulk collection --------------------------------------------------
    def collect(self, cache, examples, batch_size: int = 128) -> dict:
        """Pooled activations + outputs for many examples (chunked)."""
        acts = {p: [] for p in HOOKS}
        outs = {k: [] for k in ("cur_logit", "fut_logit", "stage_now", "stage_fut")}
        for i in range(0, len(examples), batch_size):
            r = self.run(self.seqs(cache, examples[i : i + batch_size]), save=tuple(HOOKS))
            for p in HOOKS:
                acts[p].append(r["pooled"][p])
            for k in outs:
                outs[k].append(r[k])
        res = {"acts": {p: np.concatenate(v) for p, v in acts.items()}, **{k: np.concatenate(v) for k, v in outs.items()}}
        res["acts"]["input"] = np.asarray([build_graph_state(ex.context[-1])["feature_vector"] for ex in examples], dtype=np.float32)
        res["z"] = res["acts"]["context_norm"]
        res["p_cur"], res["p_fut"] = _sig(res["cur_logit"], self.T), _sig(res["fut_logit"], self.T)
        res["y"] = np.asarray([ex.infil[0] for ex in examples])
        res["stage"] = np.asarray([ex.stage[0] for ex in examples])
        return res

    # -- 2. causal patching grid --------------------------------------------
    def _slice_patch(self, point: str, a: int, b: int, clean_raw: dict, node_type, node_graph: dict, G: int):
        """Patch fn writing clean activations into dims [a:b) of the LAST
        window of every example at `point`."""
        L = self.L
        if point in ("hgt_conv1", "hgt_conv2"):
            plan = {}
            for nt, ng in node_graph.items():
                if node_type is not None and nt != node_type:
                    continue
                rows = torch.nonzero(ng % L == L - 1).flatten()
                plan[nt] = (rows, clean_raw[point][nt][rows][:, a:b].clone())

            def fn(env):
                for nt, (rows, vals) in plan.items():
                    env.output[nt][rows, a:b] = vals
        elif point == "spatial_out":
            rows = torch.arange(L - 1, G, L)
            vals = clean_raw[point][rows][:, a:b].clone()

            def fn(env):
                env.output[rows, a:b] = vals
        elif point == "gru":
            vals = clean_raw[point][1][-1][:, a:b].clone()

            def fn(env):
                env.output[1][-1, :, a:b] = vals
        else:
            vals = clean_raw[point][:, a:b].clone()

            def fn(env):
                env.output[:, a:b] = vals
        return fn

    @staticmethod
    def replace_context_norm(z_new: np.ndarray):
        v = torch.as_tensor(z_new, dtype=torch.float32)

        def fn(env):
            env.output[:] = v
        return fn

    def patching_grid(self, cache, examples, n_blocks: int = 8, min_gap: float = 0.05) -> dict:
        """Mean causal effect of patching each layer subspace from a benign
        reference into the flagged windows. effect = share of the (flagged -
        clean) probability gap the patch closes, averaged over windows whose
        gap > min_gap. Reported for the current score and the t+1 forecast."""
        assert self.x_ref is not None, "benign_reference.json missing -- run train_interpretability"
        xr = torch.tensor(self.x_ref)
        seqs_m = self.seqs(cache, examples)
        seqs_c = [[*s[:-1], mf.graph_with_features(build_graph_state(ex.context[-1]), xr)] for s, ex in zip(seqs_m, examples)]
        mal = self.run(seqs_m, save=tuple(HOOKS), keep_raw=True)
        cln = self.run(seqs_c, save=tuple(HOOKS), keep_raw=True)
        pm_c, pm_f = self._probs(mal)
        pc_c, pc_f = self._probs(cln)
        gap_c, gap_f = pm_c - pc_c, pm_f - pc_f
        G = mal["n_graphs"]
        out = {"points": [], "cur": [], "fut": [], "node_type": {}, "n_windows": len(examples),
               "n_valid_cur": int((np.abs(gap_c) > min_gap).sum()), "n_valid_fut": int((np.abs(gap_f) > min_gap).sum()),
               "mean_p_flagged": float(pm_c.mean()), "mean_p_clean": float(pc_c.mean())}

        def effect(p_patch_c, p_patch_f):
            res = []
            for gap, base, patched in ((gap_c, pm_c, p_patch_c), (gap_f, pm_f, p_patch_f)):
                ok = np.abs(gap) > min_gap
                e = np.clip((base - patched) / np.where(ok, gap, 1.0), 0.0, 1.0)
                res.append(float(e[ok].mean()) if ok.any() else float("nan"))
            return res

        for point in HOOKS:
            dim = _last_dim(point, cln["raw"])
            size = max(1, dim // n_blocks)
            cur_row, fut_row = [], []
            for b in range(n_blocks):
                a, e_ = b * size, (dim if b == n_blocks - 1 else (b + 1) * size)
                r = self.run(seqs_m, save=(), patches={point: self._slice_patch(point, a, e_, cln["raw"], None, mal["node_graph"], G)})
                c, f = effect(*self._probs(r))
                cur_row.append(c); fut_row.append(f)
            out["points"].append(point); out["cur"].append(cur_row); out["fut"].append(fut_row)
            if point in ("hgt_conv1", "hgt_conv2"):
                out["node_type"][point] = {}
                for nt in mal["node_graph"]:
                    r = self.run(seqs_m, save=(), patches={point: self._slice_patch(point, 0, dim, cln["raw"], nt, mal["node_graph"], G)})
                    out["node_type"][point][nt] = effect(*self._probs(r))[0]
        out["cur"], out["fut"] = np.array(out["cur"]), np.array(out["fut"])
        return out

    # -- 3. embedding-dimension ablation -------------------------------------
    def dim_ablation(self, cache, examples, z_mean: np.ndarray) -> dict:
        """Mean-ablate each embedding dimension of z_t and measure the mean
        |change| in infiltration probability."""
        seqs = self.seqs(cache, examples)
        base = self.run(seqs, save=("context_norm",))
        pb_c, pb_f = self._probs(base)
        D = len(z_mean)
        d_cur, d_fut = np.zeros(D), np.zeros(D)
        for d in range(D):
            vals = torch.full((len(examples), 1), float(z_mean[d]))

            def fn(env, d=d, vals=vals):
                env.output[:, d : d + 1] = vals
            pc, pf = self._probs(self.run(seqs, save=(), patches={"context_norm": fn}))
            d_cur[d], d_fut[d] = np.abs(pb_c - pc).mean(), np.abs(pb_f - pf).mean()
        return {"cur": d_cur, "fut": d_fut, "base_p": float(pb_c.mean())}

    # -- 4. SAE feature interventions ----------------------------------------
    def _decoder_col(self, f: int) -> np.ndarray:
        return self.sae.decoder.weight.detach()[:, f].numpy()

    @torch.no_grad()
    def sae_effects(self, cache, examples, features: list[int] | None = None, top_k: int = 10) -> dict:
        """Ablate SAE features one at a time (z' = z - h_f * W_dec[:, f]) and
        report the change in current and forecast probability."""
        assert self.sae is not None, "SAE not trained"
        seqs = self.seqs(cache, examples)
        base = self.run(seqs, save=("context_norm",))
        z = base["pooled"]["context_norm"]
        h = self.sae.encode(torch.tensor(z)).numpy()
        if features is None:
            features = [int(i) for i in np.argsort(-h.mean(0))[:top_k]]
        pb_c, pb_f = self._probs(base)
        rows = []
        for f in features:
            z2 = z - h[:, [f]] * self._decoder_col(f)[None, :]
            pc, pf = self._probs(self.run(seqs, save=(), patches={"context_norm": self.replace_context_norm(z2)}))
            lab = self.sae_labels.get(str(f), {})
            rows.append({"feature": f, "label": lab.get("label", f"feature {f}"), "dominant_label": lab.get("dominant_label"),
                         "mean_activation": float(h[:, f].mean()), "delta_cur": float((pc - pb_c).mean()), "delta_fut": float((pf - pb_f).mean())})
        return {"rows": rows, "base_cur": float(pb_c.mean()), "base_fut": float(pb_f.mean())}

    @torch.no_grad()
    def sae_steering(self, cache, benign_examples, features: list[int], alphas=(0, 0.5, 1, 2, 4, 8)) -> dict:
        """Add alpha * (typical activation) * decoder direction of one feature to
        BENIGN embeddings. If the feature is causally attack-related, the
        infiltration probability should rise with alpha."""
        assert self.sae is not None, "SAE not trained"
        seqs = self.seqs(cache, benign_examples)
        base = self.run(seqs, save=("context_norm",))
        z = base["pooled"]["context_norm"]
        h = self.sae.encode(torch.tensor(z)).numpy()
        curves = {}
        for f in features:
            scale = float(np.percentile(h[h[:, f] > 0, f], 90)) if (h[:, f] > 0).any() else 1.0
            scale = scale if scale > 0 else 1.0
            ys = []
            for a in alphas:
                z2 = z + a * scale * self._decoder_col(f)[None, :]
                r = self.run(seqs, save=(), patches={"context_norm": self.replace_context_norm(z2)})
                ys.append(self._probs(r))
            curves[f] = {"label": self.sae_labels.get(str(f), {}).get("label", f"feature {f}"), "alphas": list(alphas),
                         "cur": [float(y[0].mean()) for y in ys], "fut": [float(y[1].mean()) for y in ys]}
        return curves

    @torch.no_grad()
    def sae_fidelity(self, cache, examples) -> dict:
        """Substitute z_t by its SAE reconstruction (and two destructive
        baselines) and measure how often the decision survives."""
        assert self.sae is not None, "SAE not trained"
        seqs = self.seqs(cache, examples)
        base = self.run(seqs, save=("context_norm",))
        z = base["pooled"]["context_norm"]
        recon = self.sae(torch.tensor(z))[0].numpy()
        variants = {"SAE reconstruction": recon, "mean-ablation": np.tile(z.mean(0), (len(z), 1)), "zero-ablation": np.zeros_like(z)}
        pb = _sig(base["cur_logit"], self.T)
        thr = self.meta.get("decision_threshold") or 0.5
        out = {}
        for name, z2 in variants.items():
            r = self.run(seqs, save=(), patches={"context_norm": self.replace_context_norm(z2)})
            p = _sig(r["cur_logit"], self.T)
            out[name] = {"mean_abs_dp": float(np.abs(p - pb).mean()), "decision_agreement": float(((p >= thr) == (pb >= thr)).mean())}
        return out

    # -- 5. non-nnsight views (plain torch, same checkpoint) -------------------
    @torch.no_grad()
    def lens_trajectory(self, acts: dict) -> dict:
        """P(infiltration) and stage distribution read from every layer via the tuned lens."""
        assert self.translators is not None, "tuned lens not trained"
        pts = [p for p in ["input", "hgt_conv1", "hgt_conv2", "spatial_out", "gru", "context_norm"] if p in self.translators and p in acts]
        traj = {}
        for p in pts:
            t = self.translators[p]
            zl = torch.tensor(acts[p]) @ t["W"] + t["b"]
            traj[p] = {"p": _sig(mf.infil_logit(self.ih, zl).numpy(), self.T), "stage": self.sc(zl).softmax(-1).numpy()}
        return {"points": pts, "traj": traj}

    @torch.no_grad()
    def rollout_bands(self, z: np.ndarray, steps: int = 5, n_samples: int = 256, seed: int = 0) -> dict:
        """Monte-Carlo trajectories from the probabilistic predictor for ONE z (D,)."""
        g = torch.Generator().manual_seed(seed)
        traj = self.model.predictor.sample_rollout(torch.tensor(z, dtype=torch.float32), steps, n_samples, generator=g)  # (S, K, D)
        S, K, D = traj.shape
        flat = traj.reshape(-1, D)
        p = _sig(mf.infil_logit(self.ih, flat).numpy().reshape(S, K), self.T)
        st = self.sc(flat).softmax(-1).numpy().reshape(S, K, -1).mean(0)
        return {"p10": np.percentile(p, 10, axis=0), "p50": np.percentile(p, 50, axis=0), "p90": np.percentile(p, 90, axis=0),
                "p_any_attack": float((p.max(1) >= 0.5).mean()), "stage_mean": st}

    def grad_x_input(self, cache, ex) -> dict:
        """(x - x_ref) * d logit / d x through the real HGT path, last window."""
        state = build_graph_state(ex.context[-1])
        hist = [cache.get(ex, j) for j in range(self.L - 1)]
        x = mf.feature_tensor(state).clone().requires_grad_(True)
        with torch.enable_grad():
            logit = mf.score_fn(self.model, self.ih, state, "current", hist)(x.unsqueeze(0)).sum()
            logit.backward()
        ref = torch.tensor(self.x_ref) if self.x_ref is not None else torch.zeros_like(x)
        return {"names": list(CANONICAL_FEATURE_NAMES), "attr": ((x.detach() - ref) * x.grad).numpy(), "x": x.detach().numpy(), "logit": float(logit)}

    # -- 6. self check --------------------------------------------------------
    @torch.no_grad()
    def selfcheck(self, cache, examples, n: int = 8) -> dict:
        """(a) nnsight activations == torch-hook activations, (b) patching a
        layer with its own output is a no-op, (c) outputs == direct forward."""
        seqs = self.seqs(cache, examples[:n])
        r = self.run(seqs, save=tuple(HOOKS), keep_raw=True)
        L = self.L
        with mf.capture(self.model) as store:
            _, _, z = self.model(seqs)
        ref = {}
        for name, calls in store.items():
            if not calls:
                continue
            if len(calls) == 1 and isinstance(calls[0], torch.Tensor) and calls[0].dim() == 2:
                ref[name] = calls[0]
            elif name in ("hgt_conv1", "hgt_conv2", "spatial_out"):
                ref[name] = torch.stack([mf._pool(c) for c in calls])[L - 1 :: L]
            else:
                ref[name] = torch.stack([mf._pool(c) for c in calls])
        diffs = {p: float((torch.tensor(r["pooled"][p]) - ref[p]).abs().max()) for p in HOOKS if p in ref}
        direct = mf.infil_logit(self.ih, z).numpy()
        noop = {}
        for p in ("hgt_conv1", "spatial_out", "gru", "context_norm", "predictor_l2"):
            r2 = self.run(seqs, save=(), patches={p: self._slice_patch(p, 0, _last_dim(p, r["raw"]), r["raw"], None, r["node_graph"], r["n_graphs"])})
            noop[p] = float(np.abs(r2["cur_logit"] - r["cur_logit"]).max())
        out_diff = float(np.abs(r["cur_logit"] - direct).max())
        return {"activation_max_abs_diff_vs_hooks": diffs, "output_max_abs_diff_vs_direct": out_diff,
                "self_patch_noop_max_abs_logit_change": noop,
                "ok": bool(max(diffs.values()) < 1e-4 and out_diff < 1e-4 and max(noop.values()) < 1e-4)}
