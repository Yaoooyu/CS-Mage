"""Run the official ConFEDE SIMS implementation on one reconstructed CS-Mage fold.

The official project hard-codes SIMS dimensions and paths.  This adapter changes
only those dataset-specific settings, retaining its three unimodal contrastive
pretraining stages and TVA fusion stage.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True, help='Path to ConFEDE repository')
    ap.add_argument('--fold', required=True, help='CS-Mage fold pkl')
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--seed', type=int, default=20260715)
    ap.add_argument('--skip-pretrain', action='store_true', help='Reuse completed unimodal checkpoints.')
    args = ap.parse_args()

    repo = Path(args.repo).resolve() / 'SIMS'
    fold = Path(args.fold).resolve()
    run_dir = Path(args.run_dir).resolve()
    for name in ('encoders', 'fusion', 'result'):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    os.chdir(repo)
    sys.path.insert(0, str(repo))

    import config
    # ConFEDE was written for a Transformers release that exposed AdamW from
    # transformers.optimization; map that deprecated alias to PyTorch AdamW.
    import torch
    import transformers
    transformers.optimization.AdamW = torch.optim.AdamW
    # CS-Mage released feature dimensions and padded lengths.
    config.seed = [args.seed]
    config.SIMS.path.raw_data_path = str(fold)
    config.SIMS.path.encoder_path = str(run_dir / 'encoders') + '/'
    config.SIMS.path.model_path = str(run_dir / 'fusion') + '/'
    config.SIMS.path.result_path = str(run_dir / 'result') + '/'
    config.SIMS.downStream.text_fea_dim = 768
    config.SIMS.downStream.audio_fea_dim = 13
    config.SIMS.downStream.vision_fea_dim = 347
    config.SIMS.downStream.audio_seq_len = 553
    config.SIMS.downStream.video_seq_len = 321
    config.SIMS.downStream.vision_dim_feedforward = 347

    # The released ConFEDE SIMS encoders hard-code 55 visual and 400 audio
    # positions.  Replace only their positional-embedding constructors so the
    # unchanged encoder/fusion architecture accepts CS-Mage's 321/553 frames.
    from torch import nn
    def position_encoder(num_patches, modality):
        class CSPositionEncoding(nn.Module):
            def __init__(self, fea_size=None, tf_hidden_dim=None, drop_out=None, config=config):
                super().__init__()
                if fea_size is None:
                    fea_size = getattr(config.SIMS.downStream, f'{modality}_fea_dim')
                if tf_hidden_dim is None:
                    tf_hidden_dim = config.SIMS.downStream.encoder_fea_dim
                if drop_out is None:
                    drop_out = getattr(config.SIMS.downStream, f'{modality}_drop_out')
                self.cls_token = nn.Parameter(torch.ones(1, 1, tf_hidden_dim))
                self.proj = nn.Linear(fea_size, tf_hidden_dim)
                self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, tf_hidden_dim))
                self.dropout = nn.Dropout(drop_out)
            def forward(self, embeddings):
                x = self.proj(embeddings)
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
                return self.dropout(x + self.position_embeddings)
        return CSPositionEncoding
    import model.net.constrastive.vision_encoder_finetune as vision_encoder
    import model.net.constrastive.audio_encoder_fintune as audio_encoder
    vision_encoder.PositionEncodingTraining = position_encoder(321, 'vision')
    audio_encoder.PositionEncodingTraining = position_encoder(553, 'audio')

    # The new pytorch-metric-learning base class forwards reference embeddings
    # and labels to compute_loss.  ConFEDE's two custom losses implement the
    # older three-argument form; their mathematics is unchanged by this shim.
    import util.metrics as confede_metrics
    for loss_class in (confede_metrics.cont_NTXentLoss, confede_metrics.sds_NTXentLoss):
        legacy_compute = loss_class.compute_loss
        def compatible_compute(self, embeddings, labels, indices_tuple=None,
                               ref_emb=None, ref_labels=None, _legacy=legacy_compute):
            return _legacy(self, embeddings, labels, indices_tuple)
        loss_class.compute_loss = compatible_compute

    # The original repository passes prefetch_factor with num_workers=0, which
    # PyTorch 2.x rejects.  Patch its loader in-memory without changing model code.
    from torch.utils.data import DataLoader
    from pytorch_metric_learning import samplers
    from dataloader.SIMS import SIMSDataset
    def compatible_loader(name, batch_size=None, shuffle=True, use_sampler=False,
                          use_similarity=False, simi_return_mono=False, **_):
        dataset = SIMSDataset(name, use_similarity=use_similarity, simi_return_mono=simi_return_mono)
        sampler, drop_last = None, False
        if use_sampler:
            shuffle, drop_last = False, True
            sampler = samplers.MPerClassSampler(labels=dataset.data['regression_labels'], m=1,
                                                batch_size=None, length_before_new_iter=batch_size * 21)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                          pin_memory=True, drop_last=drop_last, num_workers=0)
    import train.constrastive.Ttrain as text_train
    import train.constrastive.Vtrain as vision_train
    import train.constrastive.Atrain as audio_train
    import train.constrastive.TVA_fusion_train as fusion_train
    for module in (text_train, vision_train, audio_train, fusion_train):
        module.SIMSDataloader = compatible_loader
    from util.common import set_random_seed

    set_random_seed(args.seed)
    check = {'MAE': 10000}
    if not args.skip_pretrain:
        text_train.Ttrain(check=check, config=config); text_train.Ttest(check_list=['MAE'], config=config)
        vision_train.Vtrain(check=check, config=config); vision_train.Vtest(check_list=['MAE'], config=config)
        audio_train.Atrain(check=check, config=config); audio_train.Atest(check_list=['MAE'], config=config)
    fusion_train.TVA_train_fusion('TVA_fusion', check={'MAE': 10000}, load_model='best_MAE', load_pretrain=True, config=config)
    result = fusion_train.TVA_test_fusion('TVA_fusion', check_list=['Mult_acc_2', 'F1_score', 'Mult_acc_3', 'Mult_acc_5', 'MAE', 'Corr'], config=config)
    to_json = lambda x: x.item() if hasattr(x, 'item') else float(x)
    (run_dir / 'result' / 'metrics.json').write_text(json.dumps(result, indent=2, default=to_json), encoding='utf-8')
    print(json.dumps(result, default=to_json), flush=True)


if __name__ == '__main__':
    main()
