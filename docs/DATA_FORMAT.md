# Data format

The feature package is distributed separately as `CS-Mage_fulldata.pkl`; see
`DATA_ACCESS_AND_ETHICS.md` for the download locations and data-use terms. The
MSA loader expects the original fold pkl schema with `text`, `audio`, `vision`,
their valid lengths, four label sets, `raw_text`, and `id`. The observed feature
tensors are text 768-D, audio 13-D MFCC frames, and visual 347-D OpenFace
frames; their padded maximum lengths are 39, 553, and 321.

`CS-Mage_fulldata.pkl` contains the complete collection and does not embed the
official train/validation/test assignment. Use the fixed split metadata in
`../splits/msa/` for MSA and `../splits/asr/` for ASR rather than resampling a
new split.

The ASR manifest is a CSV keyed by `sample_id`. The release split files contain no absolute audio path. Users must map IDs to their locally obtained WAV files through `CS_MAGE_ASR_DATA_ROOT`.
