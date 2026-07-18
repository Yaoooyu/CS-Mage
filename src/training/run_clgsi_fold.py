"""Run the official CLGSI implementation on one CS-Mage fold.

Only dataset-dependent quantities (paths, lengths and feature dimensions) are
overridden.  The model and its original trainer/loss remain from the official
CLGSI repository.
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


class AttrDict(dict):
    """Match CLGSI's original Storage object (mapping + attribute access)."""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def args_for_fold(path: Path, run_dir: Path, seed: int) -> AttrDict:
    return AttrDict(
        modelName="clgsi", datasetName="sims", train_mode="regression",
        dataPath=str(path), num_workers=0, gpu_ids=[0],
        model_save_dir=str(run_dir / "models"),
        res_save_dir=str(run_dir / "results"),
        model_save_path=str(run_dir / "models" / "clgsi-sims-regression.pth"),
        seed=seed, cur_time=1,
        need_data_aligned=False, need_model_aligned=False,
        need_normalized=False, use_bert=True, use_finetune=True,
        save_labels=False, dividing_line=0.4,
        batch_size=32, learning_rate_bert=5e-5,
        learning_rate_audio=5e-4, learning_rate_video=5e-4,
        learning_rate_other=5e-4, weight_decay_bert=0.001,
        weight_decay_audio=0.01, weight_decay_video=0.01,
        weight_decay_other=0.001, fusion_filter_nums=16,
        a_encoder_heads=1, v_encoder_heads=1, a_encoder_layers=2,
        v_encoder_layers=2, text_out=768, audio_out=13, video_out=347,
        t_bert_dropout=0.1, post_fusion_dim=256, post_text_dim=128,
        post_audio_dim=128, post_video_dim=128, post_fusion_dropout=0.1,
        post_text_dropout=0.4, post_audio_dropout=0.4,
        post_video_dropout=0.4, skip_net_reduction=8, warm_up_epochs=95,
        gamma=0.38, update_epochs=1, early_stop=8, H=1.0,
        train_samples=759, num_classes=3, language="cn", KeyEval="MAE",
        feature_dims=(768, 13, 347), seq_lens=(39, 553, 321),
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )


def patch_local_bert(assets: Path) -> None:
    from transformers import AutoModel, AutoTokenizer
    from models.subNets.BertTextEncoder import BertTextEncoder

    def init(self, language="cn", use_finetune=True):
        torch.nn.Module.__init__(self)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(assets), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(assets), local_files_only=True)
        self.use_finetune = use_finetune

    BertTextEncoder.__init__ = init


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--fold", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--assets", required=True)
    p.add_argument("--seed", type=int, default=20260715)
    ns = p.parse_args()

    repo, run_dir = Path(ns.repo).resolve(), Path(ns.run_dir).resolve()
    fold_path = Path(ns.fold).resolve()
    assets_path = Path(ns.assets).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(exist_ok=True)
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    random.seed(ns.seed); np.random.seed(ns.seed); torch.manual_seed(ns.seed)
    torch.cuda.manual_seed_all(ns.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    patch_local_bert(assets_path)
    from data.load_data import MMDataLoader
    from models.AMIO import AMIO
    from trains.ATIO import ATIO

    args = args_for_fold(fold_path, run_dir, ns.seed)
    loader = MMDataLoader(args)
    model = AMIO(args).to(args.device)
    trainer = ATIO().getTrain(args)
    trainer.do_train(model, loader)
    state = torch.load(args.model_save_path, map_location=args.device)
    model.load_state_dict(state)
    model.to(args.device)
    metrics = trainer.do_test(model, loader["test"], mode="TEST")
    metrics = {k: float(v) for k, v in metrics.items()}
    metrics["seed"] = ns.seed
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
