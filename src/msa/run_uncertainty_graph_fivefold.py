"""Uncertainty-aware Quality-Gated Graph Mixture with synchronous mixup.

Each modality has a learned epistemic proxy (log variance).  Quality gates are
calibrated by this uncertainty, then three experts (reliable fusion, pairwise
interaction, and cross-modal disagreement) are mixed per sample.
"""
import argparse, copy, json, pickle, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from run_experiments import pool_text,pool_with_lengths,standardize
from run_interaction_fivefold import join,sims_metrics

def seed(s): random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def sce(logits,target): return -(target*F.log_softmax(logits,1)).sum(1).mean()

class UQGraphMixture(nn.Module):
    auxiliary=True
    def __init__(self,dims):
        super().__init__();d=192
        self.enc=nn.ModuleList([nn.Sequential(nn.Linear(x,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(.20)) for x in dims])
        self.quality_gate=nn.Sequential(nn.Linear(d*3+3,d),nn.GELU(),nn.Dropout(.10),nn.Linear(d,3))
        self.uncertainty=nn.ModuleList([nn.Sequential(nn.Linear(d+1,80),nn.GELU(),nn.Linear(80,1)) for _ in dims])
        self.q=nn.Linear(d,d,bias=False);self.k=nn.Linear(d,d,bias=False);self.v=nn.Linear(d,d,bias=False);self.graph_norm=nn.LayerNorm(d)
        self.pair=nn.Sequential(nn.Linear(d*3,d),nn.LayerNorm(d),nn.GELU())
        self.diff=nn.Sequential(nn.Linear(d*3,d),nn.LayerNorm(d),nn.GELU())
        self.expert_gate=nn.Sequential(nn.Linear(d*3+3+3,d),nn.GELU(),nn.Linear(d,3))
        self.head=nn.Sequential(nn.Linear(d,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(.25),nn.Linear(256,128),nn.GELU())
        self.cls=nn.Linear(128,5);self.mean=nn.Linear(128,1);self.logvar=nn.Linear(128,1);self.aux=nn.ModuleList([nn.Linear(d,5) for _ in dims]);self.alpha=nn.Parameter(torch.tensor(1.05))
    def forward(self,xs,quality):
        hs=[m(x) for m,x in zip(self.enc,xs)];stack=torch.stack(hs,1)
        base=self.quality_gate(torch.cat(hs+[quality],1));unc=torch.cat([m(torch.cat([h,quality[:,i:i+1]],1)) for i,(h,m) in enumerate(zip(hs,self.uncertainty))],1).clamp(-3,3)
        rel=torch.softmax(base-unc,1)
        attn=torch.softmax(torch.matmul(self.q(stack),self.k(stack).transpose(1,2))/np.sqrt(stack.shape[-1]),2);g=self.graph_norm(stack+torch.matmul(attn,self.v(stack)))
        reliable=(rel.unsqueeze(-1)*g).sum(1)
        p01,p02,p12=g[:,0]*g[:,1],g[:,0]*g[:,2],g[:,1]*g[:,2];inter=self.pair(torch.cat([p01,p02,p12],1))
        d01,d02,d12=(g[:,0]-g[:,1]).abs(),(g[:,0]-g[:,2]).abs(),(g[:,1]-g[:,2]).abs();dis=self.diff(torch.cat([d01,d02,d12],1))
        eg=torch.softmax(self.expert_gate(torch.cat([reliable,inter,dis,rel,unc],1)),1);f=eg[:,0:1]*reliable+eg[:,1:2]*inter+eg[:,2:3]*dis
        h=self.head(f);cls=self.cls(h);reg=torch.tanh(self.mean(h)).squeeze(1);cscore=(torch.softmax(cls,1)*torch.linspace(-1,1,5,device=h.device)).sum(1);final=torch.sigmoid(self.alpha)*reg+(1-torch.sigmoid(self.alpha))*cscore
        return cls,reg,final,self.logvar(h).squeeze(1).clamp(-4,2),[a(g[:,i]) for i,a in enumerate(self.aux)],rel,eg

def evaluate(model,x,q,y,idx):
    model.eval()
    with torch.no_grad():p=model([z[idx] for z in x],q[idx])[2].cpu().numpy()
    return sims_metrics(p,y[idx].cpu().numpy())

def train(model,data,split,base_seed):
    seed(base_seed);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');x,q,yc,yr,uni=data;x=[torch.from_numpy(z).to(dev) for z in x];q=torch.from_numpy(q).to(dev);yc=torch.from_numpy(yc).long().to(dev);yr=torch.from_numpy(yr).to(dev);uni=[torch.from_numpy(z).long().to(dev) for z in uni];tr,va,te=[torch.from_numpy(z).long().to(dev) for z in split];model=model.to(dev);opt=torch.optim.AdamW(model.parameters(),lr=6e-4,weight_decay=3e-4);best,state,stale=-1e9,None,0
    for epoch in range(180):
        model.train();perm=tr[torch.randperm(len(tr),device=dev)];lam=float(np.random.beta(.35,.35));lam=max(lam,1-lam);mx=[lam*z[tr]+(1-lam)*z[perm] for z in x];mq=lam*q[tr]+(1-lam)*q[perm];ct=lam*F.one_hot(yc[tr],5).float()+(1-lam)*F.one_hot(yc[perm],5).float();rt=lam*yr[tr]+(1-lam)*yr[perm]
        cls,reg,final,lv,aux,rel,eg=model(mx,mq);cscore=(torch.softmax(cls,1)*torch.linspace(-1,1,5,device=dev)).sum(1);nll=.5*(torch.exp(-lv)*(final-rt).square()+lv).mean();loss=sce(cls,ct)+.60*nll+.20*F.smooth_l1_loss(final,rt)+.10*F.mse_loss(reg,cscore)
        for a,u in zip(aux,uni):loss=loss+.08*sce(a,lam*F.one_hot(u[tr],5).float()+(1-lam)*F.one_hot(u[perm],5).float())
        # avoid a degenerate one-expert solution while allowing confident routing.
        loss=loss+.005*(rel*torch.log(rel+1e-8)).sum(1).mean()+.003*(eg*torch.log(eg+1e-8)).sum(1).mean()
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),4.);opt.step();m=evaluate(model,x,q,yr,va);score=m['Corr']-m['MAE']+.35*m['Acc2']
        if score>best:best,state,stale=score,copy.deepcopy(model.state_dict()),0
        else:stale+=1
        if stale>=30:break
    model.load_state_dict(state);out=evaluate(model,x,q,yr,te);out.update({'method':'UQGraphMixture-Mixup','seed':base_seed,'epochs':epoch+1,'n_test':int(len(te)),'reg_alpha':float(torch.sigmoid(model.alpha).detach().cpu())});return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--seed',type=int,default=20260715);args=ap.parse_args();root=Path(args.root);rows=[]
    for path in sorted((root/'data').glob('fold_*.pkl')):
        with path.open('rb') as f:z=pickle.load(f)
        ps=[z['train'],z['valid'],z['test']];n=[len(p['id']) for p in ps];ks=['text','audio','vision','audio_lengths','vision_lengths','classification_labels','regression_labels','classification_labels_T','classification_labels_A','classification_labels_V'];d={k:join(ps,k) for k in ks};split=(np.arange(n[0]),np.arange(n[0],n[0]+n[1]),np.arange(n[0]+n[1],sum(n)));text,tl=pool_text(d['text']);x=[standardize(v,split[0]) for v in (text,pool_with_lengths(d['audio'],d['audio_lengths']),pool_with_lengths(d['vision'],d['vision_lengths']))];q=np.log1p(np.stack([tl,d['audio_lengths'],d['vision_lengths']],1).astype(np.float32));q=standardize(q,split[0]);data=(x,q,d['classification_labels'].astype(np.int64)-1,d['regression_labels'].astype(np.float32),[d[k].astype(np.int64)-1 for k in ('classification_labels_T','classification_labels_A','classification_labels_V')]);out=train(UQGraphMixture([v.shape[1] for v in x]),data,split,args.seed);out['fold']=int(path.stem[-1]);rows.append(out);print(json.dumps(out),flush=True)
    (root/'uqgraph_fivefold.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
if __name__=='__main__':main()
