"""Tri-RIDE: tri-level reliability-guided interaction and disagreement experts.

This runner deliberately reuses the established CS-Mage five-fold loader,
train-fold-only standardization, and SIMS metrics.  It adds no raw features or
new feature extractor.  The `variant` switch is for the pre-registered
Tri-RIDE ablations; it never changes the data split or uses test data to tune.
"""
import argparse
import copy
import csv
import json
import math
import pickle
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from run_experiments import pool_text, pool_with_lengths, standardize
from run_interaction_fivefold import join, sims_metrics


EPS = 1e-8
VALUE_GRID = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
PAIRS = ((0, 1), (0, 2), (1, 2))  # TA, TV, AV


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def inv_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def hetero_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Gaussian negative log likelihood; mean is the only supervised location."""
    return 0.5 * (torch.exp(-logvar) * (mean - target).square() + logvar).mean()


def entropy_regularizer(weights: torch.Tensor) -> torch.Tensor:
    """Positive coefficient * sum(p log p), minimized to favour high entropy."""
    return (weights * torch.log(weights.clamp_min(EPS))).sum(dim=1).mean()


@dataclass
class Config:
    seed: int = 20260715
    hidden: int = 192
    epochs: int = 180
    lr: float = 6e-4
    weight_decay: float = 3e-4
    patience: int = 30
    mixup_alpha: float = 0.35
    corruption_probability: float = 0.30
    ranking_margin: float = 0.20
    lambda_mod: float = 0.08
    lambda_edge: float = 0.08
    lambda_rank: float = 0.05
    lambda_r_entropy: float = 0.005
    lambda_beta_entropy: float = 0.003


def variant_flags(name: str) -> dict:
    """A1/A2/A3 and requested targeted ablations; all use one fixed config."""
    if name == 'node':
        return dict(use_edge=False, sample_decision=False, use_rank=True, average_experts=False)
    if name == 'relation':
        return dict(use_edge=True, sample_decision=False, use_rank=True, average_experts=False)
    if name == 'tri':
        return dict(use_edge=True, sample_decision=True, use_rank=True, average_experts=False)
    if name == 'no_rank':
        return dict(use_edge=True, sample_decision=True, use_rank=False, average_experts=False)
    if name == 'no_edge':
        return dict(use_edge=False, sample_decision=True, use_rank=True, average_experts=False)
    if name == 'global_decision':
        return dict(use_edge=True, sample_decision=False, use_rank=True, average_experts=False)
    if name == 'average_experts':
        return dict(use_edge=True, sample_decision=True, use_rank=True, average_experts=True)
    raise ValueError(f'Unknown variant: {name}')


class SharedEdgeReliability(nn.Module):
    """One shared network for the three unordered modality relations."""
    def __init__(self, hidden: int):
        super().__init__()
        # h_m+h_n, |h_m-h_n|, h_m*h_n, s_m, s_n, r_m, r_n = 3d + 4
        self.body = nn.Sequential(
            nn.Linear(hidden * 3 + 4, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.10)
        )
        self.mean = nn.Linear(hidden, 1)
        self.logvar = nn.Linear(hidden, 1)

    def forward(self, hi, hj, si, sj, ri, rj):
        x = torch.cat([hi + hj, (hi - hj).abs(), hi * hj, si, sj, ri, rj], dim=1)
        x = self.body(x)
        return torch.tanh(self.mean(x)).squeeze(1), self.logvar(x).squeeze(1).clamp(-4.0, 2.0)


class TriRIDE(nn.Module):
    """Node-, relation-, and decision-level reliability with explicit artifacts."""
    auxiliary = True

    def __init__(self, dims, flags, hidden=192):
        super().__init__()
        self.hidden, self.flags = hidden, flags
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.20))
            for dim in dims
        ])
        self.quality_gate = nn.Sequential(
            nn.Linear(hidden * 3 + 3, hidden), nn.GELU(), nn.Dropout(0.10), nn.Linear(hidden, 3)
        )
        self.node_cls = nn.ModuleList(nn.Linear(hidden, 5) for _ in dims)
        self.node_mean = nn.ModuleList(nn.Linear(hidden, 1) for _ in dims)
        self.node_logvar = nn.ModuleList(nn.Linear(hidden, 1) for _ in dims)
        self.raw_gamma = nn.Parameter(torch.full((3,), inv_softplus(1.0)))

        self.edge = SharedEdgeReliability(hidden)
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.graph_norm = nn.LayerNorm(hidden)
        self.raw_eta = nn.Parameter(torch.tensor(inv_softplus(1.0)))
        self.raw_tau_edge = nn.Parameter(torch.tensor(inv_softplus(1.0)))

        self.interaction = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.10)
        )
        self.disagreement = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.10)
        )
        self.expert_gate = nn.Sequential(
            nn.Linear(hidden * 3 + 9, hidden), nn.GELU(), nn.Linear(hidden, 3)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.GELU()
        )
        self.cls = nn.Linear(128, 5)
        self.reg_mean = nn.Linear(128, 1)
        self.reg_logvar = nn.Linear(128, 1)
        self.raw_tau_reg = nn.Parameter(torch.tensor(inv_softplus(1.0)))
        self.raw_tau_cls = nn.Parameter(torch.tensor(inv_softplus(1.0)))
        # Used only by variants with global decision reliability (A1/A2/B3).
        self.global_alpha = nn.Parameter(torch.tensor(1.05))

    @staticmethod
    def _class_moments(logits):
        values = VALUE_GRID.to(logits.device)
        probs = F.softmax(logits, dim=1)
        mean = (probs * values).sum(1)
        var = (probs * (values - mean[:, None]).square()).sum(1)
        return probs, mean, var

    def _edge_states(self, hs, node_s, reliability):
        batch = hs[0].shape[0]
        if not self.flags['use_edge']:
            zeros = hs[0].new_zeros(batch, 3)
            return zeros, zeros, hs[0].new_ones(batch, 3)
        means, logvars = [], []
        for i, j in PAIRS:
            mu, s = self.edge(hs[i], hs[j], node_s[:, i:i + 1], node_s[:, j:j + 1],
                              reliability[:, i:i + 1], reliability[:, j:j + 1])
            means.append(mu)
            logvars.append(s)
        edge_mean, edge_s = torch.stack(means, 1), torch.stack(logvars, 1)
        tau = F.softplus(self.raw_tau_edge) + EPS
        return edge_mean, edge_s, torch.sigmoid(-edge_s / tau)

    def _graph(self, stack, node_s, edge_c):
        content = torch.matmul(self.q(stack), self.k(stack).transpose(1, 2)) / math.sqrt(self.hidden)
        if self.flags['use_edge']:
            b = stack.shape[0]
            cmat = torch.ones(b, 3, 3, device=stack.device, dtype=stack.dtype)
            for edge_idx, (i, j) in enumerate(PAIRS):
                cmat[:, i, j] = edge_c[:, edge_idx]
                cmat[:, j, i] = edge_c[:, edge_idx]
            # For self loops we deliberately keep content attention only: no
            # edge-reliability factor and no target-node uncertainty penalty.
            offdiag = ~torch.eye(3, dtype=torch.bool, device=stack.device)[None]
            adjusted = content + torch.log(cmat.clamp_min(EPS)) - F.softplus(self.raw_eta) * node_s[:, None, :]
            scores = torch.where(offdiag, adjusted, content)
        else:
            scores = content
        attention = F.softmax(scores, dim=2)
        graph = self.graph_norm(stack + torch.matmul(attention, self.v(stack)))
        return graph, attention

    def forward(self, xs, quality):
        hs = [encoder(x) for encoder, x in zip(self.encoders, xs)]
        stack = torch.stack(hs, 1)
        node_logits = torch.stack([head(h) for head, h in zip(self.node_cls, hs)], 1)
        node_mean = torch.cat([torch.tanh(head(h)) for head, h in zip(self.node_mean, hs)], 1)
        node_s = torch.cat([head(h) for head, h in zip(self.node_logvar, hs)], 1).clamp(-4.0, 2.0)
        base_quality = self.quality_gate(torch.cat(hs + [quality], 1))
        gamma = F.softplus(self.raw_gamma)[None, :]
        reliability = F.softmax(base_quality - gamma * node_s, dim=1)

        edge_mean, edge_s, edge_c = self._edge_states(hs, node_s, reliability)
        graph, attention = self._graph(stack, node_s, edge_c)
        reliable = (reliability[:, :, None] * graph).sum(1)
        inter_parts, diff_parts = [], []
        for edge_idx, (i, j) in enumerate(PAIRS):
            endpoint = torch.sqrt(reliability[:, i:i + 1] * reliability[:, j:j + 1] + EPS)
            pos = edge_c[:, edge_idx:edge_idx + 1] * endpoint
            neg = (1.0 - edge_c[:, edge_idx:edge_idx + 1]) * endpoint
            inter_parts.append(pos * graph[:, i] * graph[:, j])
            diff_parts.append(neg * (graph[:, i] - graph[:, j]).abs())
        interaction = self.interaction(torch.cat(inter_parts, 1))
        disagreement = self.disagreement(torch.cat(diff_parts, 1))
        gate_input = torch.cat([reliable, interaction, disagreement, reliability, node_s, edge_c], 1)
        beta = F.softmax(self.expert_gate(gate_input), 1)
        if self.flags['average_experts']:
            beta = torch.full_like(beta, 1.0 / 3.0)
        fused = beta[:, :1] * reliable + beta[:, 1:2] * interaction + beta[:, 2:] * disagreement

        body = self.head(fused)
        cls_logits = self.cls(body)
        mu_reg = torch.tanh(self.reg_mean(body)).squeeze(1)
        s_reg = self.reg_logvar(body).squeeze(1).clamp(-4.0, 2.0)
        probs, y_cls, var_cls = self._class_moments(cls_logits)
        if self.flags['sample_decision']:
            tau_reg = F.softplus(self.raw_tau_reg) + EPS
            tau_cls = F.softplus(self.raw_tau_cls) + EPS
            precision_reg = torch.exp(-s_reg / tau_reg)
            precision_cls = torch.exp(-var_cls / tau_cls)
            alpha = precision_reg / (precision_reg + precision_cls + EPS)
        else:
            # expand_as retains the gradient to the shared learnable scalar.
            alpha = torch.sigmoid(self.global_alpha).expand_as(mu_reg)
        y_hat = alpha * mu_reg + (1.0 - alpha) * y_cls
        return {
            'cls_logits': cls_logits, 'mu_reg': mu_reg, 's_reg': s_reg, 'probs': probs,
            'y_cls': y_cls, 'var_cls': var_cls, 'alpha': alpha, 'y_hat': y_hat,
            'node_logits': node_logits, 'node_mean': node_mean, 'node_s': node_s,
            'reliability': reliability, 'edge_mean': edge_mean, 'edge_s': edge_s,
            'edge_c': edge_c, 'attention': attention, 'beta': beta,
        }


def ordinal_partner_indices(labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Pair each item with a same/adjacent ordinal label when possible."""
    y = labels.detach().cpu().numpy()
    partners, fallback = [], 0
    all_indices = np.arange(len(y))
    for index, label in enumerate(y):
        candidates = all_indices[(np.abs(y - label) <= 1) & (all_indices != index)]
        if len(candidates) == 0:
            candidates = all_indices[all_indices != index]
            fallback += 1
        partners.append(int(np.random.choice(candidates)))
    return torch.as_tensor(partners, device=labels.device, dtype=torch.long), fallback


def samplewise_mixup(xs, quality, ycls, yreg, config):
    partner, fallback = ordinal_partner_indices(ycls)
    lam = np.random.beta(config.mixup_alpha, config.mixup_alpha, size=len(ycls)).astype(np.float32)
    lam = np.maximum(lam, 1.0 - lam)
    lam = torch.from_numpy(lam).to(yreg.device)[:, None]
    mixed_x = [lam * x + (1.0 - lam) * x[partner] for x in xs]
    mixed_q = lam * quality + (1.0 - lam) * quality[partner]
    mixed_cls = lam * F.one_hot(ycls, 5).float() + (1.0 - lam) * F.one_hot(ycls[partner], 5).float()
    mixed_reg = (lam[:, 0] * yreg) + ((1.0 - lam[:, 0]) * yreg[partner])
    return mixed_x, mixed_q, mixed_cls, mixed_reg, lam, partner, fallback


def corrupt_modalities(xs, probability):
    """Training-only light corruption; masks identify modal uncertainties to rank."""
    corrupted = [x.clone() for x in xs]
    b = xs[0].shape[0]
    active = torch.rand(b, device=xs[0].device) < probability
    target = torch.randint(0, 3, (b,), device=xs[0].device)
    mask = torch.zeros(b, 3, dtype=torch.bool, device=xs[0].device)
    for m, x in enumerate(corrupted):
        rows = active & (target == m)
        if not rows.any():
            continue
        mask[rows, m] = True
        selected = x[rows]
        operation = torch.randint(0, 4, (selected.shape[0],), device=x.device)
        noise_rows = operation == 0
        if noise_rows.any():
            selected[noise_rows] += 0.15 * torch.randn_like(selected[noise_rows])
        mask_rows = (operation == 1) | (operation == 2)
        if mask_rows.any():
            feature_mask = torch.rand_like(selected[mask_rows]) < 0.15
            selected[mask_rows] = selected[mask_rows].masked_fill(feature_mask, 0.0)
        zero_rows = operation == 3
        if zero_rows.any():  # low-probability complete modal removal
            selected[zero_rows] = 0.0
        x[rows] = selected
    return corrupted, mask


def prepare_fold(path: Path):
    with path.open('rb') as handle:
        fold = pickle.load(handle)
    parts = [fold['train'], fold['valid'], fold['test']]
    sizes = [len(part['id']) for part in parts]
    keys = ['text', 'audio', 'vision', 'audio_lengths', 'vision_lengths', 'classification_labels',
            'regression_labels', 'classification_labels_T', 'classification_labels_A',
            'classification_labels_V', 'id']
    data = {key: join(parts, key) for key in keys}
    split = (
        np.arange(sizes[0]),
        np.arange(sizes[0], sizes[0] + sizes[1]),
        np.arange(sizes[0] + sizes[1], sum(sizes)),
    )
    text, text_length = pool_text(data['text'])
    xs = [standardize(x, split[0]) for x in (
        text,
        pool_with_lengths(data['audio'], data['audio_lengths']),
        pool_with_lengths(data['vision'], data['vision_lengths']),
    )]
    quality = np.log1p(np.stack([text_length, data['audio_lengths'], data['vision_lengths']], 1).astype(np.float32))
    quality = standardize(quality, split[0])
    uni = [data[key].astype(np.int64) - 1 for key in
           ('classification_labels_T', 'classification_labels_A', 'classification_labels_V')]
    return (xs, quality, data['classification_labels'].astype(np.int64) - 1,
            data['regression_labels'].astype(np.float32), uni, np.asarray(data['id'])), split


@torch.no_grad()
def forward_indexed(model, xs, quality, idx):
    model.eval()
    return model([x[idx] for x in xs], quality[idx])


def evaluate(model, xs, quality, target, idx):
    out = forward_indexed(model, xs, quality, idx)
    metrics = sims_metrics(out['y_hat'].detach().cpu().numpy(), target[idx].detach().cpu().numpy())
    return metrics, out


def loss_terms(output, cls_target, reg_target, uni_labels, corrupt_output, corruption_mask, config, flags):
    terms = {}
    terms['cls'] = soft_cross_entropy(output['cls_logits'], cls_target)
    terms['final_hetero'] = hetero_nll(output['mu_reg'], output['s_reg'], reg_target)
    terms['huber'] = F.smooth_l1_loss(output['y_hat'], reg_target)
    terms['consistency'] = F.mse_loss(output['mu_reg'], output['y_cls'])
    aux = [soft_cross_entropy(output['node_logits'][:, i], uni_labels[i]) for i in range(3)]
    terms['unimodal_cls'] = torch.stack(aux).mean()
    node_nll = [hetero_nll(output['node_mean'][:, i], output['node_s'][:, i], reg_target) for i in range(3)]
    terms['mod_unc'] = torch.stack(node_nll).mean()
    if flags['use_edge']:
        edge_nll = [hetero_nll(output['edge_mean'][:, i], output['edge_s'][:, i], reg_target) for i in range(3)]
        terms['edge_unc'] = torch.stack(edge_nll).mean()
    else:
        terms['edge_unc'] = output['y_hat'].new_zeros(())
    if flags['use_rank'] and corruption_mask.any():
        clean_s = output['node_s']
        corrupt_s = corrupt_output['node_s']
        terms['unc_rank'] = F.relu(config.ranking_margin + clean_s - corrupt_s)[corruption_mask].mean()
    else:
        terms['unc_rank'] = output['y_hat'].new_zeros(())
    terms['r_entropy'] = entropy_regularizer(output['reliability'])
    terms['beta_entropy'] = entropy_regularizer(output['beta'])
    total = (terms['cls'] + 0.60 * terms['final_hetero'] + 0.20 * terms['huber'] +
             0.10 * terms['consistency'] + 0.08 * terms['unimodal_cls'] +
             config.lambda_mod * terms['mod_unc'] + config.lambda_edge * terms['edge_unc'] +
             config.lambda_rank * terms['unc_rank'] + config.lambda_r_entropy * terms['r_entropy'] +
             config.lambda_beta_entropy * terms['beta_entropy'])
    terms['total'] = total
    return terms


def tensor_checks(output):
    checks = {
        'finite': all(torch.isfinite(x).all().item() for x in output.values() if isinstance(x, torch.Tensor)),
        'reliability_sum_error': float((output['reliability'].sum(1) - 1).abs().max().item()),
        'beta_sum_error': float((output['beta'].sum(1) - 1).abs().max().item()),
        'alpha_min': float(output['alpha'].min().item()),
        'alpha_max': float(output['alpha'].max().item()),
        'node_clamp_fraction': float((output['node_s'].abs() >= 3.999).float().mean().item()),
        'reg_clamp_fraction': float((output['s_reg'].abs() >= 3.999).float().mean().item()),
    }
    return checks


def write_predictions(path: Path, ids, labels, out):
    tensors = {key: value.detach().cpu().numpy() for key, value in out.items() if isinstance(value, torch.Tensor)}
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        header = ['id', 'ground_truth', 'y_hat', 'mu_reg', 's_reg', 'y_cls', 'var_cls', 'alpha_i']
        header += [f'class_prob_{i}' for i in range(5)] + ['r_T', 'r_A', 'r_V', 's_T', 's_A', 's_V']
        header += ['c_TA', 'c_TV', 'c_AV', 'beta_r', 'beta_i', 'beta_d']
        header += [f'attn_{i}{j}' for i in range(3) for j in range(3)]
        writer.writerow(header)
        for i, sample_id in enumerate(ids):
            row = [sample_id, labels[i], tensors['y_hat'][i], tensors['mu_reg'][i], tensors['s_reg'][i],
                   tensors['y_cls'][i], tensors['var_cls'][i], tensors['alpha'][i]]
            row += tensors['probs'][i].tolist() + tensors['reliability'][i].tolist() + tensors['node_s'][i].tolist()
            row += tensors['edge_c'][i].tolist() + tensors['beta'][i].tolist() + tensors['attention'][i].reshape(-1).tolist()
            writer.writerow(row)


def train_one_fold(fold_path: Path, run_dir: Path, config: Config, variant: str, smoke: bool):
    flags = variant_flags(variant)
    seed_everything(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    (raw_xs, raw_quality, raw_ycls, raw_yreg, raw_uni, raw_ids), split = prepare_fold(fold_path)
    xs = [torch.from_numpy(x).to(device) for x in raw_xs]
    quality = torch.from_numpy(raw_quality).to(device)
    ycls = torch.from_numpy(raw_ycls).long().to(device)
    yreg = torch.from_numpy(raw_yreg).float().to(device)
    uni = [torch.from_numpy(x).long().to(device) for x in raw_uni]
    tr, va, te = [torch.from_numpy(x).long().to(device) for x in split]
    model = TriRIDE([x.shape[1] for x in xs], flags, config.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    epochs = min(config.epochs, 5) if smoke else config.epochs
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.json').write_text(json.dumps({**asdict(config), 'variant': variant, 'flags': flags,
        'fold': fold_path.name, 'input_dims': [int(x.shape[1]) for x in xs], 'split_sizes': [len(tr), len(va), len(te)]}, indent=2), encoding='utf-8')
    with (run_dir / 'train_log.csv').open('w', newline='', encoding='utf-8') as log_file:
        fields = ['epoch', 'score', 'valid_Acc2', 'valid_Acc3', 'valid_Acc5', 'valid_F1', 'valid_MAE', 'valid_Corr',
                  'mixup_fallbacks', 'corrupted_fraction', 'grad_norm', 'node_clamp_fraction', 'reg_clamp_fraction']
        fields += ['loss_' + x for x in ('total', 'cls', 'final_hetero', 'huber', 'consistency', 'unimodal_cls', 'mod_unc', 'edge_unc', 'unc_rank', 'r_entropy', 'beta_entropy')]
        writer = csv.DictWriter(log_file, fieldnames=fields); writer.writeheader()
        best_score, best_state, best_epoch = -float('inf'), None, 0
        smoke_report = None
        started = time.perf_counter()
        for epoch in range(1, epochs + 1):
            model.train()
            mx, mq, ct, rt, lam, partner, fallback = samplewise_mixup([x[tr] for x in xs], quality[tr], ycls[tr], yreg[tr], config)
            mixed_uni = []
            for m, label in enumerate(uni):
                mixed_uni.append(lam * F.one_hot(label[tr], 5).float() +
                                  (1.0 - lam) * F.one_hot(label[tr][partner], 5).float())
            clean_output = model(mx, mq)
            corrupt_x, corruption_mask = corrupt_modalities(mx, config.corruption_probability)
            corrupt_output = model(corrupt_x, mq)
            terms = loss_terms(clean_output, ct, rt, mixed_uni, corrupt_output, corruption_mask, config, flags)
            optimizer.zero_grad(set_to_none=True)
            terms['total'].backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0).item())
            optimizer.step()
            valid, _ = evaluate(model, xs, quality, yreg, va)
            score = valid['Corr'] - valid['MAE'] + 0.35 * valid['Acc2']
            checks = tensor_checks(clean_output)
            row = {'epoch': epoch, 'score': score, **{'valid_' + k: v for k, v in valid.items()},
                   'mixup_fallbacks': fallback, 'corrupted_fraction': float(corruption_mask.any(1).float().mean().item()),
                   'grad_norm': grad_norm, **{key: checks[key] for key in ('node_clamp_fraction', 'reg_clamp_fraction')}}
            row.update({'loss_' + key: float(value.detach().item()) for key, value in terms.items()})
            writer.writerow(row); log_file.flush()
            if smoke:
                smoke_report = {**checks, 'finite_losses': all(torch.isfinite(v).item() for v in terms.values()),
                                'finite_gradient': math.isfinite(grad_norm), 'parameter_count': sum(p.numel() for p in model.parameters())}
            if score > best_score:
                best_score, best_state, best_epoch = score, copy.deepcopy(model.state_dict()), epoch
                torch.save({'model': best_state, 'epoch': epoch, 'score': score, 'config': asdict(config)}, run_dir / 'best.pt')
            elif epoch - best_epoch >= config.patience and not smoke:
                break
    model.load_state_dict(best_state)
    if device.type == 'cuda': torch.cuda.synchronize()
    tick = time.perf_counter(); test, test_out = evaluate(model, xs, quality, yreg, te)
    if device.type == 'cuda': torch.cuda.synchronize()
    infer_ms = (time.perf_counter() - tick) * 1000.0 / len(te)
    write_predictions(run_dir / 'test_predictions.csv', raw_ids[split[2]], raw_yreg[split[2]], test_out)
    result = {**test, 'variant': variant, 'fold': int(fold_path.stem[-1]), 'best_epoch': best_epoch,
              'epochs_ran': epoch, 'best_valid_score': best_score, 'params': sum(p.numel() for p in model.parameters()),
              'inference_ms_per_sample': infer_ms, 'train_seconds': time.perf_counter() - started}
    if smoke_report is not None:
        result['smoke_checks'] = smoke_report
    (run_dir / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='rebuilt_5fold directory containing data/fold_*.pkl')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--variant', default='tri', choices=['node', 'relation', 'tri', 'no_rank', 'no_edge', 'global_decision', 'average_experts'])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--epochs', type=int, default=180)
    parser.add_argument('--seed', type=int, default=20260715)
    args = parser.parse_args()
    config = Config(seed=args.seed, epochs=args.epochs)
    fold_path = Path(args.root) / 'data' / f'fold_{args.fold}.pkl'
    result = train_one_fold(fold_path, Path(args.run_dir), config, args.variant, args.smoke)
    print('FINAL ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
