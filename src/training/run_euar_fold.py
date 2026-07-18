"""Paper-faithful EUAR reproduction for CS-Mage standardized five-fold data.

EUAR: Enhanced Experts with Uncertainty-Aware Routing for Multimodal
Sentiment Analysis (ACM MM 2024).  The public anonymous implementation is
no longer available, so this runner follows the objective stated in the
paper: Gaussian experts, Top-k routing, KL regularization, Switch-style
load balancing, and uncertainty-aware routing.
"""
import argparse
import json
import os
import pickle
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pooled_inputs(split):
    """Pool only valid frames; the supplied fold files are train-standardized."""
    text = np.asarray(split['text'], dtype=np.float32)
    token_mask = np.asarray(split['text_bert'])[:, 1].astype(bool)
    token_mask[:, 0] = False  # remove [CLS]; retain valid lexical tokens
    denom = np.maximum(token_mask.sum(1, keepdims=True), 1)
    text = ((text * token_mask[..., None]).sum(1) / denom).astype(np.float32)

    def temporal_mean(values, lengths):
        values = np.asarray(values, dtype=np.float32)
        lengths = np.asarray(lengths, dtype=np.int64)
        mask = np.arange(values.shape[1])[None, :] < lengths[:, None]
        return ((values * mask[..., None]).sum(1) / np.maximum(lengths[:, None], 1)).astype(np.float32)

    audio = temporal_mean(split['audio'], split['audio_lengths'])
    vision = temporal_mean(split['vision'], split['vision_lengths'])
    labels = np.asarray(split['regression_labels'], dtype=np.float32).reshape(-1)
    return text, audio, vision, labels


class ModalDataset(Dataset):
    def __init__(self, split):
        arrays = pooled_inputs(split)
        self.values = [torch.from_numpy(x) for x in arrays]

    def __len__(self):
        return len(self.values[-1])

    def __getitem__(self, index):
        return tuple(value[index] for value in self.values)


class GaussianExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mu = nn.Linear(dim, dim)
        self.logvar = nn.Linear(dim, dim)

    def forward(self, x):
        mu = self.mu(x)
        logvar = self.logvar(x).clamp(-8.0, 5.0)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std if self.training else mu
        kl = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).mean(-1)
        uncertainty = logvar.exp().mean(-1)
        return z, kl, uncertainty


class UncertaintyAwareMoE(nn.Module):
    def __init__(self, dim, num_experts=8, top_k=3):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList(GaussianExpert(dim) for _ in range(num_experts))
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x):
        outputs, kls, uncertainties = zip(*(expert(x) for expert in self.experts))
        outputs = torch.stack(outputs, 1)
        kls = torch.stack(kls, 1)
        uncertainties = torch.stack(uncertainties, 1)
        soft_gate = F.softmax(self.gate(x), dim=-1)
        top_indices = soft_gate.topk(self.top_k, dim=-1).indices
        gate = torch.zeros_like(soft_gate).scatter_(1, top_indices, 1.0) * soft_gate
        fused = torch.einsum('be,bed->bd', gate, outputs)

        # Switch-style load balancing: empirical routing frequency times gate mass.
        routes = F.one_hot(soft_gate.argmax(-1), self.num_experts).float().mean(0).detach()
        balance = self.num_experts * (routes * soft_gate.mean(0)).sum()
        uncertainty_loss = ((uncertainties * gate).sum(-1) / self.num_experts).mean()
        return fused, kls.mean(), balance, uncertainty_loss


class EUAR(nn.Module):
    def __init__(self, hidden=128, experts=8, top_k=3):
        super().__init__()
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(768, hidden), nn.GELU(), nn.LayerNorm(hidden)),
            nn.Sequential(nn.Linear(13, hidden), nn.GELU(), nn.LayerNorm(hidden)),
            nn.Sequential(nn.Linear(347, hidden), nn.GELU(), nn.LayerNorm(hidden)),
        ])
        self.moes = nn.ModuleList(UncertaintyAwareMoE(hidden, experts, top_k) for _ in range(3))
        self.predictor = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Dropout(0.15), nn.Linear(hidden, 1)
        )

    def forward(self, text, audio, vision):
        fused, kls, balances, uncertainties = [], [], [], []
        for encoder, moe, value in zip(self.encoders, self.moes, (text, audio, vision)):
            result, kl, balance, uncertainty = moe(encoder(value))
            fused.append(result)
            kls.append(kl)
            balances.append(balance)
            uncertainties.append(uncertainty)
        prediction = self.predictor(torch.cat(fused, -1)).squeeze(-1)
        return prediction, torch.stack(kls).sum(), torch.stack(balances).sum(), torch.stack(uncertainties).sum()


def metrics(prediction, label):
    prediction = np.asarray(prediction).reshape(-1)
    label = np.asarray(label).reshape(-1)
    clipped = np.clip(prediction, -1., 1.)
    def cls(bins):
        return np.digitize(clipped, bins), np.digitize(label, bins)
    p2, y2 = cls([0.])
    p3, y3 = cls([-1e-8, 1e-8])
    p5, y5 = cls([-.6, -.2, .2, .6])
    corr = float(np.corrcoef(prediction, label)[0, 1]) if np.std(prediction) > 0 and np.std(label) > 0 else 0.0
    return {
        'Acc2': float(accuracy_score(y2, p2)),
        'Acc3': float(accuracy_score(y3, p3)),
        'Acc5': float(accuracy_score(y5, p5)),
        'F1': float(f1_score(y2, p2, average='weighted')),
        'MAE': float(np.mean(np.abs(prediction - label))),
        'Corr': corr,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_pred, all_label = [], []
    for text, audio, vision, label in loader:
        pred, _, _, _ = model(text.to(device), audio.to(device), vision.to(device))
        all_pred.extend(pred.cpu().numpy())
        all_label.extend(label.numpy())
    return metrics(all_pred, all_label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--seed', type=int, default=20260715)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--experts', type=int, default=8)
    args = parser.parse_args()
    os.makedirs(args.run_dir, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(args.fold, 'rb') as handle:
        data = pickle.load(handle)
    train_loader = DataLoader(ModalDataset(data['train']), batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(ModalDataset(data['valid']), batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(ModalDataset(data['test']), batch_size=64, shuffle=False, num_workers=0)
    model = EUAR(experts=args.experts).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    best_mae = float('inf')
    best_epoch = 0
    best_path = os.path.join(args.run_dir, 'best.pt')
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        for text, audio, vision, label in train_loader:
            text, audio, vision, label = (x.to(device) for x in (text, audio, vision, label))
            prediction, kl, balance, uncertainty = model(text, audio, vision)
            task = F.mse_loss(prediction, label)
            loss = task + 1e-3 * kl + 1e-5 * balance + 1e-3 * uncertainty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_total += loss.item() * label.size(0)
        scheduler.step()
        valid = evaluate(model, valid_loader, device)
        if valid['MAE'] < best_mae:
            best_mae, best_epoch = valid['MAE'], epoch
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'valid': valid}, best_path)
        print(f"epoch={epoch:03d} loss={loss_total / len(train_loader.dataset):.6f} "
              f"valid_mae={valid['MAE']:.6f} valid_acc2={valid['Acc2']:.6f} best={best_mae:.6f}", flush=True)
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    result = evaluate(model, test_loader, device)
    result.update({'best_epoch': best_epoch, 'valid': checkpoint['valid'], 'config': vars(args)})
    with open(os.path.join(args.run_dir, 'metrics.json'), 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print('FINAL ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
