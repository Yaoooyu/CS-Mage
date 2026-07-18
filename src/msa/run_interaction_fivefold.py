"""Fixed-five-fold screen of reliability-aware cross-modal interaction fusion.

Uses only the released text/audio/vision feature tensors and reports the same
SIMS regression metrics used by MMSA, so results are directly comparable to
the reconstructed benchmark table.
"""
import argparse
import copy
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as F

from run_experiments import pool_text, pool_with_lengths, standardize


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def join(parts, key):
    values = [p[key] for p in parts]
    return np.concatenate(values) if isinstance(values[0], np.ndarray) else sum((list(v) for v in values), [])


def sims_metrics(pred, truth):
    pred = np.clip(np.asarray(pred).reshape(-1), -1, 1)
    truth = np.clip(np.asarray(truth).reshape(-1), -1, 1)
    def bins(x, edges):
        return np.digitize(x, edges[1:-1], right=True)
    y2, p2 = bins(truth, [-1.01, 0., 1.01]), bins(pred, [-1.01, 0., 1.01])
    y3, p3 = bins(truth, [-1.01, -.1, .1, 1.01]), bins(pred, [-1.01, -.1, .1, 1.01])
    y5, p5 = bins(truth, [-1.01, -.7, -.1, .1, .7, 1.01]), bins(pred, [-1.01, -.7, -.1, .1, .7, 1.01])
    corr = float(np.corrcoef(pred, truth)[0, 1]) if np.std(pred) > 0 else 0.
    return {"Acc2": float((p2 == y2).mean()), "Acc3": float((p3 == y3).mean()),
            "Acc5": float((p5 == y5).mean()), "F1": float(f1_score(y2, p2, average="weighted")),
            "MAE": float(np.abs(pred - truth).mean()), "Corr": corr}


class QualityGateRegressor(nn.Module):
    def __init__(self, dims, interaction=False, auxiliary=False):
        super().__init__(); self.interaction, self.auxiliary = interaction, auxiliary
        self.enc = nn.ModuleList([nn.Sequential(nn.Linear(d, 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(.22)) for d in dims])
        self.gate = nn.Sequential(nn.Linear(192 * 3 + 3, 192), nn.GELU(), nn.Linear(192, 3))
        if interaction:
            self.pairs = nn.ModuleList([nn.Sequential(nn.Linear(192, 128), nn.GELU()) for _ in range(3)])
            head_dim = 192 * 4 + 128 * 3
        else:
            head_dim = 192 * 4
        self.head = nn.Sequential(nn.Linear(head_dim, 320), nn.LayerNorm(320), nn.GELU(), nn.Dropout(.28), nn.Linear(320, 128), nn.GELU())
        self.cls = nn.Linear(128, 5); self.reg = nn.Linear(128, 1)
        self.aux = nn.ModuleList([nn.Linear(192, 5) for _ in range(3)])

    def forward(self, xs, quality):
        hs = [enc(x) for enc, x in zip(self.enc, xs)]
        gates = torch.softmax(self.gate(torch.cat(hs + [quality], 1)), 1)
        fused = sum(gates[:, i:i + 1] * hs[i] for i in range(3))
        body = [fused] + hs
        if self.interaction:
            for (i, j), proj in zip(((0, 1), (0, 2), (1, 2)), self.pairs):
                body.append(proj(hs[i] * hs[j]) * torch.sqrt(gates[:, i:i+1] * gates[:, j:j+1] + 1e-6))
        h = self.head(torch.cat(body, 1))
        return self.cls(h), torch.tanh(self.reg(h)).squeeze(1), [head(z) for head, z in zip(self.aux, hs)], gates


def evaluate(model, xs, q, yreg, test):
    model.eval()
    with torch.no_grad(): pred = model([x[test] for x in xs], q[test])[1].detach().cpu().numpy()
    return sims_metrics(pred, yreg[test].detach().cpu().numpy())


def train_one(name, model, data, split, seed, epochs=180):
    seed_everything(seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xs, q, ycls, yreg, uni = data
    xs = [torch.from_numpy(x).to(device) for x in xs]; q = torch.from_numpy(q).to(device)
    ycls = torch.from_numpy(ycls).long().to(device); yreg = torch.from_numpy(yreg).float().to(device)
    uni = [torch.from_numpy(x).long().to(device) for x in uni]
    tr, va, te = [torch.from_numpy(x).long().to(device) for x in split]
    model = model.to(device); opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=2e-4)
    best, best_state, stale = -1e9, None, 0
    for epoch in range(epochs):
        model.train(); cls, reg, aux, _ = model([x[tr] for x in xs], q[tr])
        loss = F.cross_entropy(cls, ycls[tr]) + .75 * F.smooth_l1_loss(reg, yreg[tr])
        if model.auxiliary: loss = loss + .08 * sum(F.cross_entropy(a, u[tr]) for a, u in zip(aux, uni))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0); opt.step()
        metric = evaluate(model, xs, q, yreg, va)
        score = metric["Corr"] - metric["MAE"] + .35 * metric["Acc2"]
        if score > best:
            best, best_state, stale = score, copy.deepcopy(model.state_dict()), 0
        else: stale += 1
        if stale >= 30: break
    model.load_state_dict(best_state); result = evaluate(model, xs, q, yreg, te)
    result.update({"method": name, "seed": seed, "epochs": epoch + 1, "n_test": int(len(te))})
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--seed", type=int, default=20260715); args = ap.parse_args()
    root = Path(args.root); rows = []
    methods = [("QualityGate-MT", lambda d: QualityGateRegressor(d, interaction=False, auxiliary=True)),
               ("ReliabilityInteraction-MT", lambda d: QualityGateRegressor(d, interaction=True, auxiliary=True))]
    for fold_path in sorted((root / "data").glob("fold_*.pkl")):
        with fold_path.open("rb") as f: fold = pickle.load(f)
        parts = [fold["train"], fold["valid"], fold["test"]]; counts = [len(x["id"]) for x in parts]
        keys = ["text", "audio", "vision", "audio_lengths", "vision_lengths", "classification_labels", "regression_labels", "classification_labels_T", "classification_labels_A", "classification_labels_V"]
        d = {key: join(parts, key) for key in keys}
        split = (np.arange(counts[0]), np.arange(counts[0], counts[0] + counts[1]), np.arange(counts[0] + counts[1], sum(counts)))
        text, text_len = pool_text(d["text"])
        xs = [standardize(x, split[0]) for x in (text, pool_with_lengths(d["audio"], d["audio_lengths"]), pool_with_lengths(d["vision"], d["vision_lengths"]))]
        q = np.log1p(np.stack([text_len, d["audio_lengths"], d["vision_lengths"]], 1).astype(np.float32)); q = standardize(q, split[0])
        ycls = d["classification_labels"].astype(np.int64) - 1; yreg = d["regression_labels"].astype(np.float32)
        uni = [d[k].astype(np.int64) - 1 for k in ("classification_labels_T", "classification_labels_A", "classification_labels_V")]
        for name, make in methods:
            result = train_one(name, make([x.shape[1] for x in xs]), (xs, q, ycls, yreg, uni), split, args.seed)
            result["fold"] = int(fold_path.stem[-1]); rows.append(result); print(json.dumps(result), flush=True)
    (root / "interaction_fivefold.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
