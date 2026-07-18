"""Append BERT token inputs to each fixed CS-Mage fold feature file."""
import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
from transformers import BertTokenizer


def encode(tokenizer, texts, max_length):
    out = tokenizer(list(texts), padding='max_length', truncation=True,
                    max_length=max_length, return_tensors='np')
    token_type = out.get('token_type_ids', np.zeros_like(out['input_ids']))
    return np.stack([out['input_ids'], out['attention_mask'], token_type], axis=1).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--vocab', required=True)
    ap.add_argument('--max-length', type=int, default=39)
    args = ap.parse_args()
    root = Path(args.data_dir)
    tokenizer = BertTokenizer(vocab_file=args.vocab)
    manifest = {'tokenizer': 'bert-base-chinese vocab', 'vocab': args.vocab, 'max_length': args.max_length, 'folds': []}
    for path in sorted(root.glob('fold_*.pkl')):
        with path.open('rb') as f: data = pickle.load(f)
        record = {'file': path.name}
        for split in ('train', 'valid', 'test'):
            data[split]['text_bert'] = encode(tokenizer, data[split]['raw_text'], args.max_length)
            record[split] = list(data[split]['text_bert'].shape)
        temp = path.with_suffix('.bert.tmp')
        with temp.open('wb') as f: pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp, path)
        manifest['folds'].append(record)
        print(record, flush=True)
    (root / 'bert_input_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

if __name__ == '__main__': main()
