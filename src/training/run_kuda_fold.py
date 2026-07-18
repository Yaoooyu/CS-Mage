"""Five-fold CS-Mage adapter for the official KuDA codebase.

The official model/training objectives are retained.  This wrapper only adapts
CS-Mage feature dimensions, the local BERT path, and the changed PyTorch
TransformerEncoder return convention.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def patch_transformer_encoder() -> None:
    """Restore KuDA's required (last_state, intermediate_states) interface."""
    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=None):
        out = src
        hidden = [out]
        for layer in self.layers:
            out = layer(out, src_mask=mask, src_key_padding_mask=src_key_padding_mask,
                        is_causal=bool(is_causal))
            hidden.append(out)
        if self.norm is not None:
            out = self.norm(out)
        return out, hidden
    nn.TransformerEncoder.forward = forward


def make_args(fold: Path, seed: int) -> AttrDict:
    return AttrDict(
        datasetName="sims", dataPath=str(fold), seq_lens=[39, 321, 553],
        num_workers=0, train_mode="regression", fusion_layers=3,
        dropout=0.3, hidden_size=256, ffn_size=512, seed=seed,
        batch_size=32, lr=3e-5, weight_decay=1e-5, n_epochs=50,
    )


def patch_cs_mage_encoders(assets: Path) -> None:
    from models.Encoder_KIAdapter import UniPretrain, UnimodalEncoder

    def init(self, opt, bert_pretrained=str(assets)):
        nn.Module.__init__(self)
        self.enc_t = UniPretrain("T", pretrained=str(assets),
                                 num_patches=opt.seq_lens[0], proj_fea_dim=768)
        self.enc_v = UniPretrain("V", num_patches=opt.seq_lens[1], fea_size=347)
        self.enc_a = UniPretrain("A", num_patches=opt.seq_lens[2], fea_size=13)
    UnimodalEncoder.__init__ = init


def inputs_from(batch, device):
    return {
        "V": batch["vision"].to(device),
        "A": batch["audio"].to(device),
        "T": batch["text"].to(device),
        "mask": {
            "V": batch["vision_padding_mask"][:, 1:].to(device),
            "A": batch["audio_padding_mask"][:, 1:].to(device),
            "T": [],
        },
    }


def predict_metrics(model, loader, device, metric):
    model.eval(); pred, truth = [], []
    with torch.no_grad():
        for batch in loader:
            out, _ = model(inputs_from(batch, device), None)
            pred.append(out.cpu())
            truth.append(batch["labels"]["M"].view(-1, 1).cpu())
    return metric(torch.cat(pred), torch.cat(truth))


def pretrain(modality, data, args, device, metric, epochs, path):
    from models.Encoder_KIAdapter import UniPretrain
    dims = {"T": (39, 768), "V": (321, 347), "A": (553, 13)}
    length, dim = dims[modality]
    model = UniPretrain(modality, pretrained=args.bert_assets,
                        num_patches=length, fea_size=dim,
                        proj_fea_dim=768 if modality == "T" else 128).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    loss_fn = nn.MSELoss()
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in data["train"]:
            output = model(inputs_from(batch, device))[1]
            label = batch["labels"][modality].to(device).view(-1, 1)
            loss = loss_fn(output, label)
            loss.backward(); optim.step(); optim.zero_grad()
        if epoch == 1 or epoch == epochs or epoch % 10 == 0:
            print(f"PRETRAIN {modality} epoch={epoch}/{epochs} loss={loss.item():.6f}", flush=True)
    torch.save(model.state_dict(), path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--fold", required=True)
    ap.add_argument("--run-dir", required=True); ap.add_argument("--assets", required=True)
    ap.add_argument("--seed", type=int, default=20260715)
    ap.add_argument("--pretrain-epochs", nargs=3, type=int, default=[100, 100, 50])
    ap.add_argument("--fusion-epochs", type=int, default=50)
    ns = ap.parse_args()
    repo, fold, run_dir, assets = (Path(ns.repo).resolve(), Path(ns.fold).resolve(),
                                   Path(ns.run_dir).resolve(), Path(ns.assets).resolve())
    run_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo)); os.chdir(repo)
    random.seed(ns.seed); np.random.seed(ns.seed); torch.manual_seed(ns.seed)
    torch.cuda.manual_seed_all(ns.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    patch_transformer_encoder(); patch_cs_mage_encoders(assets)
    from core.dataset import MMDataLoader
    from core.metric import MetricsTop
    from models.OverallModal import build_model
    from core.utils import calculate_ratio_senti
    args = make_args(fold, ns.seed); args.bert_assets = str(assets)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = MMDataLoader(args); metric = MetricsTop().getMetics("sims")
    ckpts = {}
    for modality, epochs in zip(("T", "V", "A"), ns.pretrain_epochs):
        ckpts[modality] = pretrain(modality, data, args, device, metric, epochs,
                                   run_dir / f"pretrain_{modality}.pt")
    model = build_model(args).to(device)
    for m, path in ckpts.items():
        getattr(model.UniEncKI, f"enc_{m.lower()}").load_state_dict(torch.load(path, map_location=device))
    for name, parameter in model.UniEncKI.named_parameters():
        if "adapter" in name or "decoder" in name:
            parameter.requires_grad = False
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    best, best_state = float("inf"), None
    for epoch in range(1, ns.fusion_epochs + 1):
        model.train()
        for batch in data["train"]:
            label = batch["labels"]["M"].to(device).view(-1, 1)
            prediction, nce = model(inputs_from(batch, device), label.detach().clone())
            loss = loss_fn(prediction, label) + 0.1 * nce
            loss.backward(); optim.step(); optim.zero_grad()
        valid = predict_metrics(model, data["valid"], device, metric)
        if valid["MAE"] < best:
            best = valid["MAE"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch == ns.fusion_epochs or epoch % 10 == 0:
            print(f"FUSION epoch={epoch}/{ns.fusion_epochs} valid={valid}", flush=True)
    model.load_state_dict(best_state)
    result = {k: float(v) for k, v in predict_metrics(model, data["test"], device, metric).items()}
    result.update(seed=ns.seed, selected_valid_mae=float(best))
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
