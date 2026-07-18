"""Create the immutable sample-level stratified 5-fold CS-Mage protocol.

Each fold contains 70% train, 10% validation and 20% test samples.  The
outer test fold and inner validation partition are stratified by the five-level
multimodal class.  The resulting files follow the MMSA feature-file schema.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


KEEP = [
    'audio_lengths', 'vision_lengths', 'audio', 'vision', 'raw_text',
    'classification_labels_T', 'regression_labels_T',
    'classification_labels_A', 'regression_labels_A',
    'classification_labels_V', 'regression_labels_V',
    'classification_labels', 'regression_labels', 'text', 'id',
]


def select(value, indices):
    if isinstance(value, np.ndarray):
        return value[indices]
    return [value[i] for i in indices]


def counts(y, ids):
    values, nums = np.unique(y[ids], return_counts=True)
    return {str(int(v)): int(n) for v, n in zip(values, nums)}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--seed', type=int, default=20260715)
    args = ap.parse_args()
    source, output = Path(args.input), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as f:
        data = pickle.load(f)
    assert isinstance(data, dict) and all(k in data for k in KEEP)
    n = len(data['id']); y = np.asarray(data['classification_labels'], dtype=np.int64)
    assert n == 1085 and all(len(v) == n for v in data.values() if hasattr(v, '__len__'))
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    manifest = {
        'protocol': 'sample-level stratified 5-fold; inner validation stratified from each outer training partition',
        'seed': args.seed,
        'source_file': str(source), 'source_sha256': digest(source),
        'n_samples': n, 'label_field': 'classification_labels',
        'folds': []
    }
    covered = []
    for fold, (outer_train, test) in enumerate(outer.split(np.zeros(n), y), start=1):
        train, valid = train_test_split(
            outer_train, test_size=0.125, random_state=args.seed + fold,
            shuffle=True, stratify=y[outer_train]
        )
        train, valid, test = map(lambda x: np.sort(np.asarray(x, dtype=np.int64)), (train, valid, test))
        assert not (set(train) & set(valid) or set(train) & set(test) or set(valid) & set(test))
        assert len(train) + len(valid) + len(test) == n
        split_data = {mode: {k: select(data[k], indices) for k in KEEP}
                      for mode, indices in [('train', train), ('valid', valid), ('test', test)]}
        fold_path = output / f'fold_{fold}.pkl'
        with fold_path.open('wb') as f:
            pickle.dump(split_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        manifest['folds'].append({
            'fold': fold, 'file': str(fold_path),
            'n_train': int(len(train)), 'n_valid': int(len(valid)), 'n_test': int(len(test)),
            'train_indices': train.tolist(), 'valid_indices': valid.tolist(), 'test_indices': test.tolist(),
            'test_ids': [data['id'][i] for i in test],
            'label_counts': {'train': counts(y, train), 'valid': counts(y, valid), 'test': counts(y, test)},
        })
        covered.extend(test.tolist())
        print(f'fold={fold} train={len(train)} valid={len(valid)} test={len(test)} file={fold_path}', flush=True)
    assert sorted(covered) == list(range(n)), 'outer test folds do not cover every sample exactly once'
    manifest_path = output / 'fivefold_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'manifest={manifest_path}', flush=True)


if __name__ == '__main__':
    main()
