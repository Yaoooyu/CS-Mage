"""Faithful CubeMLP-style baseline for one CS-Mage fold.

Implements the paper's three axis-wise MLP mixers, three-block dimensional
schedule (L:100/10/10, M:3/3/3, D:128/32/3) and MAE objective.  CS-Mage
already provides BERT text features; audio and vision are projected to the
same 128-D temporal representation before CubeMLP mixing.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score


class FoldData(Dataset):
    def __init__(self, split): self.x = split
    def __len__(self): return len(self.x['regression_labels'])
    def __getitem__(self, i):
        return {k: torch.from_numpy(np.asarray(self.x[k][i])).float() for k in
                ('text','audio','vision','text_bert','audio_lengths','vision_lengths','regression_labels')}


class AxisMLP(nn.Module):
    def __init__(self, dim, out):
        super().__init__(); self.a=nn.Linear(dim,out); self.b=nn.Linear(out,out)
        self.skip=nn.Identity() if dim==out else nn.Linear(dim,out); self.norm=nn.LayerNorm(out)
    def forward(self,x): return self.norm(self.b(F.gelu(self.a(x)))+self.skip(x))


class CubeBlock(nn.Module):
    def __init__(self, l,m,d,lo,mo,do):
        super().__init__(); self.l=AxisMLP(l,lo); self.m=AxisMLP(m,mo); self.d=AxisMLP(d,do)
    def forward(self,x):
        x=self.l(x.permute(0,2,3,1)).permute(0,3,1,2)
        x=self.m(x.permute(0,1,3,2)).permute(0,1,3,2)
        return self.d(x)


class CubeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.tp=nn.Linear(768,128); self.ap=nn.Linear(13,128); self.vp=nn.Linear(347,128)
        self.blocks=nn.Sequential(CubeBlock(100,3,128,100,3,128),CubeBlock(100,3,128,10,3,32),CubeBlock(10,3,32,10,3,3))
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(90,64),nn.GELU(),nn.Linear(64,1))
    @staticmethod
    def pool(x,lengths,target=100):
        rows=[]
        for q,n in zip(x,lengths):
            n=max(1,int(n)); rows.append(F.adaptive_avg_pool1d(q[:n].T.unsqueeze(0),target).squeeze(0).T)
        return torch.stack(rows)
    def forward(self,b):
        tl=b['text_bert'][:,1,:].sum(1).long().clamp(min=1)
        t=self.pool(self.tp(b['text']),tl); a=self.pool(self.ap(b['audio']),b['audio_lengths']); v=self.pool(self.vp(b['vision']),b['vision_lengths'])
        return self.head(self.blocks(torch.stack((t,a,v),dim=2))).squeeze(1)


def metrics(pred, truth):
    p=np.clip(pred,-1,1); y=np.clip(truth,-1,1)
    def bucket(z,b): return np.digitize(z,b[1:-1],right=True)
    a2=bucket(p,[-1.01,0,1.01]); y2=bucket(y,[-1.01,0,1.01])
    a3=bucket(p,[-1.01,-.1,.1,1.01]); y3=bucket(y,[-1.01,-.1,.1,1.01])
    a5=bucket(p,[-1.01,-.7,-.1,.1,.7,1.01]); y5=bucket(y,[-1.01,-.7,-.1,.1,.7,1.01])
    return {'Mult_acc_2':float((a2==y2).mean()),'Mult_acc_3':float((a3==y3).mean()),'Mult_acc_5':float((a5==y5).mean()),'F1_score':float(f1_score(y2,a2,average='weighted')),'MAE':float(np.abs(p-y).mean()),'Corr':float(np.corrcoef(p,y)[0,1])}


def evaluate(model, loader, device):
    model.eval(); ps=[]; ys=[]
    with torch.no_grad():
        for b in loader:
            b={k:v.to(device) for k,v in b.items()}; ps.append(model(b).cpu()); ys.append(b['regression_labels'].view(-1).cpu())
    return metrics(torch.cat(ps).numpy(),torch.cat(ys).numpy())


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--fold',required=True); ap.add_argument('--run-dir',required=True); ap.add_argument('--seed',type=int,default=20260715); ap.add_argument('--epochs',type=int,default=120); ns=ap.parse_args()
    random.seed(ns.seed);np.random.seed(ns.seed);torch.manual_seed(ns.seed);torch.cuda.manual_seed_all(ns.seed)
    d=__import__('pickle').load(open(ns.fold,'rb')); out=Path(ns.run_dir);out.mkdir(parents=True,exist_ok=True); device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loaders={s:DataLoader(FoldData(d[s]),batch_size=32,shuffle=s=='train',num_workers=0) for s in ('train','valid','test')}
    model=CubeMLP().to(device); opt=torch.optim.SGD(model.parameters(),lr=.004,momentum=.9); sch=torch.optim.lr_scheduler.StepLR(opt,50,.1); best=1e9; state=None
    for e in range(1,ns.epochs+1):
        model.train()
        for b in loaders['train']:
            b={k:v.to(device) for k,v in b.items()}; loss=F.l1_loss(model(b),b['regression_labels'].view(-1)); loss.backward();opt.step();opt.zero_grad()
        val=evaluate(model,loaders['valid'],device)
        if val['MAE']<best: best=val['MAE'];state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if e==1 or e%25==0 or e==ns.epochs: print(f'EPOCH {e}/{ns.epochs} valid={val}',flush=True)
        sch.step()
    model.load_state_dict(state); res=evaluate(model,loaders['test'],device);res.update(seed=ns.seed,selected_valid_mae=best)
    json.dump(res,open(out/'metrics.json','w'),indent=2);print(json.dumps(res,sort_keys=True),flush=True)
if __name__=='__main__': main()
