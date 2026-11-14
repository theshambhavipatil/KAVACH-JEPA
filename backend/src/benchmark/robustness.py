"""PGD robustness check for a trained checkpoint.

Run: python -m src.benchmark.robustness [--n 30]
Reports how small a perturbation (in z-scored feature units, sigma) flips a detection.
"""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch

from src.benchmark.adversarial import robustness_report
from src.graph.state_builder import build_graph_state
from src.orchestrator import model_registry
from src.training.dataset_v2 import build_examples_v2, load_sources
from src.training.train_v2 import GraphCache


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--sources", default="csv")
    args = ap.parse_args(argv)
    model, infil_head, _, trained = model_registry.get_models()
    assert trained, "no trained checkpoint at data/processed/netjepa_weights.pt"
    meta = model_registry.get_model_meta()
    L, H = int(meta.get("context_len") or 3), int(meta.get("horizon") or 3)
    sources = load_sources(include=tuple(args.sources.split(",")))
    train, val = build_examples_v2(sources, L, H, 0.2)
    cache = GraphCache(sources)
    attacks = [ex for ex in val if ex.infil[0] == 1]
    random.Random(0).shuffle(attacks)
    items = [(build_graph_state(ex.context[-1]), [cache.get(ex, j) for j in range(L - 1)]) for ex in attacks[: args.n * 4]]
    X = np.array([build_graph_state(ex.context[-1])["feature_vector"] for ex in train[::10]])
    mask = torch.tensor((X.std(0) > 1e-6).astype(np.float32))
    rep = robustness_report(model, infil_head, items, meta.get("decision_threshold", 0.5), meta.get("temperature"),
                            max_windows=args.n, feature_mask=mask)
    print(json.dumps(rep, indent=1, default=float))
    return rep


if __name__ == "__main__":
    main()
