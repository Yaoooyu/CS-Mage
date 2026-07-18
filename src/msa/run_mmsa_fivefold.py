"""Run paper baselines with MMSA model definitions on the fixed CS-Mage folds."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict as edict

from MMSA.config import get_config_regression
from MMSA.data_loader import MMDataLoader
from MMSA.models import AMIO
from MMSA.trains import ATIO
from MMSA.utils import setup_seed


MODELS = ['tfn', 'lmf', 'lf_dnn', 'ef_lstm', 'mfn', 'graph_mfn', 'mfm', 'mult', 'mtfn', 'mlf_dnn', 'mlmf', 'self_mm', 'misa', 'mmim', 'tetfn']


def run_one(model_name, fold_file, root, seed, bert_dir=None):
    setup_seed(seed)
    args = get_config_regression(model_name, 'sims')
    args = edict(args)
    run_dir = root / 'runs' / model_name / fold_file.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    args.update({
        'custom_feature': str(fold_file), 'feature_T': '', 'feature_A': '', 'feature_V': '',
        'device': torch.device('cuda:0'), 'train_mode': 'regression', 'cur_seed': seed,
        'model_save_path': run_dir / 'best.pth', 'early_stop': min(int(args.early_stop), 20),
    })
    if bert_dir and model_name in {'self_mm', 'misa', 'mmim'}:
        args['pretrained'] = str(bert_dir)
    loaders = MMDataLoader(args, num_workers=0)
    model = AMIO(args).to(args.device)
    trainer = ATIO().getTrain(args)
    trainer.do_train(model, loaders)
    state = torch.load(args.model_save_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state); model.to(args.device)
    result = trainer.do_test(model, loaders['test'], mode='TEST')
    result = {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in result.items()}
    result.update({'model': model_name, 'fold': int(fold_file.stem.split('_')[-1]), 'seed': seed})
    (run_dir / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    del model; torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--folds', default='1,2,3,4,5')
    ap.add_argument('--seed', type=int, default=20260715)
    ap.add_argument('--bert-dir', default='')
    args = ap.parse_args()
    root = Path(args.root); data = root / 'data'
    results = []
    for model in [x.strip().lower() for x in args.models.split(',') if x.strip()]:
        if model not in MODELS: raise ValueError(f'unknown model {model}')
        for fold in [int(x) for x in args.folds.split(',')]:
            result = run_one(model, data / f'fold_{fold}.pkl', root, args.seed, Path(args.bert_dir) if args.bert_dir else None)
            results.append(result); print(json.dumps(result), flush=True)
    (root / 'raw_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
