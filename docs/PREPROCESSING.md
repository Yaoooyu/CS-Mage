# Preprocessing

## MSA

The MSA experiments consume pre-extracted features: BERT-base-Chinese token
representations (768-D), MFCC audio frames (13-D), and OpenFace 2.0 visual
frames (347-D). This repository retains the fold builder and BERT-input helper.
Obtain the source data/features under the dataset terms.

MSA audio features are pre-extracted MFCCs. They are not inputs to ASR.

## ASR

`src/asr/prepare_data.py` builds manifests and validates WAV inputs. ASR uses raw waveform input, checked as 16-kHz 16-bit mono. `normalize_text.py` applies Unicode NFKC, ASCII lowercasing, CJK/0-9/a-z retention, and removes punctuation, whitespace, emoji, and model tags. WER uses Jieba 0.42.1. CER is corpus-level edit distance; SER is utterance-level exact mismatch rate.

OpenFace is an external dependency. Use OpenFace 2.0 to generate the visual
features required by the documented 347-dimensional interface; the executable
itself is not bundled with this repository.
