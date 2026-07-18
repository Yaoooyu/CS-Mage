"""Compact, reproducible CS-Mage multimodal sentiment experiments.

Evaluation: fixed 70/10/20 stratified clip split, seed 2026.  The pickle is
feature-only, so temporal tensors are mask-pooled with the provided lengths.
"""
import argparse
import json
import math
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stratified_split(y, seed=2026):
    """70/10/20 split, stratified by the five-level multimodal label."""
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for c in np.unique(y):
        ids = np.where(y == c)[0]
        rng.shuffle(ids)
        n = len(ids); n_train = round(n * .70); n_val = round(n * .10)
        parts[0].extend(ids[:n_train]); parts[1].extend(ids[n_train:n_train+n_val]); parts[2].extend(ids[n_train+n_val:])
    return tuple(np.asarray(sorted(x), dtype=np.int64) for x in parts)


def pool_with_lengths(x, lengths):
    """Mean+standard deviation over valid timesteps, then concatenate."""
    x = x.astype(np.float32, copy=False)
    n, t, _ = x.shape
    mask = (np.arange(t)[None, :] < np.asarray(lengths)[:, None]).astype(np.float32)
    denom = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
    mean = (x * mask[..., None]).sum(axis=1) / denom
    var = ((x - mean[:, None, :]) ** 2 * mask[..., None]).sum(axis=1) / denom
    return np.concatenate([mean, np.sqrt(var + 1e-6)], axis=1)


def pool_text(x):
    # Text padding is zero-filled; no explicit text-length vector is supplied.
    active = (np.abs(x).sum(axis=2) > 0).astype(np.float32)
    denom = np.maximum(active.sum(axis=1, keepdims=True), 1.0)
    mean = (x * active[..., None]).sum(axis=1) / denom
    var = ((x - mean[:, None, :]) ** 2 * active[..., None]).sum(axis=1) / denom
    return np.concatenate([mean, np.sqrt(var + 1e-6)], axis=1), active.sum(axis=1)


def standardize(x, train_ids):
    mu = x[train_ids].mean(axis=0, keepdims=True)
    sigma = x[train_ids].std(axis=0, keepdims=True)
    return ((x - mu) / np.maximum(sigma, 1e-5)).astype(np.float32)


def ordinal_targets(y):
    return (torch.arange(4, device=y.device)[None] < y[:, None]).float()


def macro_f1(pred, true):
    scores = []
    for c in range(5):
        tp = np.sum((pred == c) & (true == c))
        fp = np.sum((pred == c) & (true != c))
        fn = np.sum((pred != c) & (true == c))
        scores.append((2 * tp) / max(2 * tp + fp + fn, 1))
    return float(np.mean(scores))


def metrics(logits, y):
    prob = torch.sigmoid(logits) if logits.shape[1] == 4 else torch.softmax(logits, dim=1)
    if logits.shape[1] == 4:
        score = prob.sum(dim=1)
        pred = torch.clamp(torch.round(score), 0, 4).long()
    else:
        pred = logits.argmax(dim=1)
        score = (prob * torch.arange(5, device=logits.device)).sum(dim=1)
    p, t, s = pred.cpu().numpy(), y.cpu().numpy(), score.cpu().numpy()
    corr = float(np.corrcoef(s, t)[0, 1]) if np.std(s) > 0 and np.std(t) > 0 else 0.0
    return {"acc5": float((p == t).mean()), "macro_f1": macro_f1(p, t),
            "mae_class": float(np.abs(s - t).mean()), "corr": corr}


class LateFusionMLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(sum(dims), 384), nn.LayerNorm(384), nn.GELU(), nn.Dropout(.30),
                                 nn.Linear(384, 160), nn.GELU(), nn.Dropout(.15), nn.Linear(160, 5))
    def forward(self, xs, lengths, training=False):
        return self.net(torch.cat(xs, dim=1)), None


class QualityGatedFusion(nn.Module):
    def __init__(self, dims, ordinal=False, auxiliary=False, modality_dropout=0.0):
        super().__init__()
        self.ordinal, self.auxiliary, self.modality_dropout = ordinal, auxiliary, modality_dropout
        self.enc = nn.ModuleList([nn.Sequential(nn.Linear(d, 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(.18)) for d in dims])
        self.gate = nn.Sequential(nn.Linear(192 * 3 + 3, 160), nn.GELU(), nn.Linear(160, 3))
        self.head = nn.Sequential(nn.Linear(192 * 4, 256), nn.GELU(), nn.Dropout(.25), nn.Linear(256, 4 if ordinal else 5))
        self.uni_heads = nn.ModuleList([nn.Linear(192, 4) for _ in range(3)]) if auxiliary else None

    def forward(self, xs, lengths, training=False):
        h = [m(x) for m, x in zip(self.enc, xs)]
        if training and self.modality_dropout:
            keep = (torch.rand((xs[0].shape[0], 3), device=xs[0].device) > self.modality_dropout).float()
            # Never remove every modality for one sample.
            keep[keep.sum(dim=1) == 0, 0] = 1.0
            h = [z * keep[:, i:i+1] for i, z in enumerate(h)]
        gates = torch.softmax(self.gate(torch.cat(h + [lengths.float()], dim=1)), dim=1)
        fused = sum(gates[:, i:i+1] * h[i] for i in range(3))
        out = self.head(torch.cat([fused] + h, dim=1))
        aux = [head(z) for head, z in zip(self.uni_heads, h)] if self.auxiliary else None
        return out, aux


def train_one(name, model, data, split, seed, epochs=180):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xs, q, y, uni = data
    xs = [torch.from_numpy(x).to(device) for x in xs]
    q = torch.from_numpy(q).to(device); y = torch.from_numpy(y).long().to(device)
    uni = [torch.from_numpy(u).long().to(device) for u in uni]
    tr, va, te = [torch.from_numpy(x).long().to(device) for x in split]
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-3)
    best, best_state, stale = -1.0, None, 0
    batch = 96
    for epoch in range(epochs):
        model.train(); order = tr[torch.randperm(len(tr), device=device)]
        for start in range(0, len(order), batch):
            ids = order[start:start+batch]
            out, aux = model([x[ids] for x in xs], q[ids], training=True)
            if out.shape[1] == 4:
                loss = F.binary_cross_entropy_with_logits(out, ordinal_targets(y[ids]))
                expected = torch.sigmoid(out).sum(dim=1)
                loss = loss + .18 * F.smooth_l1_loss(expected, y[ids].float())
                if aux is not None:
                    loss = loss + .14 * sum(F.binary_cross_entropy_with_logits(a, ordinal_targets(u[ids])) for a, u in zip(aux, uni))
            else:
                loss = F.cross_entropy(out, y[ids])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0); opt.step()
        model.eval()
        with torch.no_grad():
            val = metrics(model([x[va] for x in xs], q[va])[0], y[va])
        score = val["acc5"] + .20 * val["macro_f1"] - .03 * val["mae_class"]
        if score > best + 1e-6:
            best, stale = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 30: break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): result = metrics(model([x[te] for x in xs], q[te])[0], y[te])
    result.update({"method": name, "seed": seed, "epochs": epoch + 1, "n_test": int(len(te))})
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="CS-MSASR_fulldata.pkl")
    ap.add_argument("--out", default="results.json"); ap.add_argument("--seeds", default="42,52,62")
    args = ap.parse_args()
    with open(args.data, "rb") as f: d = pickle.load(f)
    y = np.asarray(d["classification_labels"], dtype=np.int64) - 1
    uni = [np.asarray(d[k], dtype=np.int64) - 1 for k in ["classification_labels_T", "classification_labels_A", "classification_labels_V"]]
    split = stratified_split(y)
    text, text_len = pool_text(d["text"])
    audio = pool_with_lengths(d["audio"], d["audio_lengths"])
    vision = pool_with_lengths(d["vision"], d["vision_lengths"])
    tr = split[0]
    xs = [standardize(x, tr) for x in (text, audio, vision)]
    quality = np.log1p(np.stack([text_len, np.asarray(d["audio_lengths"]), np.asarray(d["vision_lengths"])], axis=1).astype(np.float32))
    quality = standardize(quality, tr)
    data = (xs, quality, y, uni)
    dims = [x.shape[1] for x in xs]
    methods = [
        ("LateFusionMLP", lambda: LateFusionMLP(dims)),
        ("QualityGatedFusion", lambda: QualityGatedFusion(dims)),
        ("OrdinalAuxModDrop", lambda: QualityGatedFusion(dims, ordinal=True, auxiliary=True, modality_dropout=.16)),
    ]
    results = []
    for name, factory in methods:
        for seed in map(int, args.seeds.split(",")):
            # Initialise weights under the same explicit seed as data shuffling.
            seed_everything(seed)
            results.append(train_one(name, factory(), data, split, seed))
            print(json.dumps(results[-1]), flush=True)
    grouped = {}
    for name, _ in methods:
        rows = [r for r in results if r["method"] == name]
        grouped[name] = {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows]))}
                         for k in ["acc5", "macro_f1", "mae_class", "corr"]}
    output = {"protocol": {"split": "stratified 70/10/20 by multimodal five-level label", "split_seed": 2026,
                            "input": "mean+std pooling, training-fold standardization", "labels": "classification_labels - 1"},
              "per_run": results, "summary": grouped}
    Path(args.out).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(grouped, indent=2), flush=True)


if __name__ == "__main__": main()
