"""MAGE-Fusion: UQGraph with label-supervised modality/pair agreement.

The validation set selects checkpoints and hyperparameters; test data are only
evaluated after a configuration is fixed.
"""
import argparse, copy, csv, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from run_tri_ride import prepare_fold
from run_uncertainty_graph_fivefold import sce, seed
from run_interaction_fivefold import sims_metrics

VALS = torch.tensor([-1., -.5, 0., .5, 1.])

def evaluate(model, xs, q, y, idx):
    model.eval()
    with torch.no_grad():
        pred=model([z[idx] for z in xs],q[idx])['final'].detach().cpu().numpy()
    return sims_metrics(pred,y[idx].detach().cpu().numpy())

class MAGEFusion(nn.Module):
    def __init__(self, dims, gamma_init=.5, full=False):
        super().__init__(); d=192; self.full=full
        self.enc=nn.ModuleList([nn.Sequential(nn.Linear(x,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(.20)) for x in dims])
        self.evidence=nn.Sequential(nn.Linear(3*d+3,d),nn.GELU(),nn.Dropout(.10),nn.Linear(d,3))
        self.ambiguity=nn.ModuleList([nn.Sequential(nn.Linear(d+1,80),nn.GELU(),nn.Linear(80,1)) for _ in dims])
        self.q,self.k,self.v=nn.Linear(d,d,False),nn.Linear(d,d,False),nn.Linear(d,d,False); self.gnorm=nn.LayerNorm(d)
        self.pair_net=nn.Sequential(nn.Linear(3*d,d),nn.LayerNorm(d),nn.GELU(),nn.Linear(d,1))
        self.pair,self.diff=nn.Sequential(nn.Linear(3*d,d),nn.LayerNorm(d),nn.GELU()),nn.Sequential(nn.Linear(3*d,d),nn.LayerNorm(d),nn.GELU())
        # The expert gate observes both modality--target evidence (w, b) and
        # the three predicted pairwise agreements c, as specified in Eq. 16.
        self.egate=nn.Sequential(nn.Linear(3*d+9,d),nn.GELU(),nn.Linear(d,3))
        self.head=nn.Sequential(nn.Linear(d,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(.25),nn.Linear(256,128),nn.GELU())
        self.cls,self.mean,self.logvar=nn.Linear(128,5),nn.Linear(128,1),nn.Linear(128,1); self.aux=nn.ModuleList([nn.Linear(d,5) for _ in dims]); self.alpha=nn.Parameter(torch.tensor(1.05)); self.gamma_raw=nn.Parameter(torch.tensor(float(gamma_init)))
    def forward(self,xs,q):
        hs=[m(x) for m,x in zip(self.enc,xs)]; st=torch.stack(hs,1); a=self.evidence(torch.cat(hs+[q],1)); b=torch.cat([m(torch.cat([h,q[:,i:i+1]],1)) for i,(h,m) in enumerate(zip(hs,self.ambiguity))],1).clamp(-3,3); w=torch.softmax(a-b,1)
        pairs=((0,1),(0,2),(1,2)); feats=[torch.cat([hs[i]+hs[j],(hs[i]-hs[j]).abs(),hs[i]*hs[j]],1) for i,j in pairs]; c=torch.sigmoid(torch.cat([self.pair_net(z) for z in feats],1))
        s=torch.matmul(self.q(st),self.k(st).transpose(1,2))/np.sqrt(st.size(-1)); bias=torch.zeros_like(s); bias[:,0,1]=bias[:,1,0]=2*c[:,0]-1; bias[:,0,2]=bias[:,2,0]=2*c[:,1]-1; bias[:,1,2]=bias[:,2,1]=2*c[:,2]-1; attn=torch.softmax(s+(F.softplus(self.gamma_raw)*bias if self.full else 0),2); g=self.gnorm(st+torch.matmul(attn,self.v(st)))
        ea=(w.unsqueeze(-1)*g).sum(1); pos=[];neg=[]
        for z,(i,j) in enumerate(pairs):
            rw=torch.sqrt(w[:,i:i+1]*w[:,j:j+1]); pos.append((c[:,z:z+1]*rw if self.full else 1)*g[:,i]*g[:,j]); neg.append(((1-c[:,z:z+1])*rw if self.full else 1)*(g[:,i]-g[:,j]).abs())
        ep=self.pair(torch.cat(pos,1)); ed=self.diff(torch.cat(neg,1)); beta=torch.softmax(self.egate(torch.cat([ea,ep,ed,w,b,c],1)),1); f=beta[:,0:1]*ea+beta[:,1:2]*ep+beta[:,2:3]*ed; h=self.head(f); cls=self.cls(h); reg=torch.tanh(self.mean(h)).squeeze(1); cs=(torch.softmax(cls,1)*VALS.to(h.device)).sum(1); final=torch.sigmoid(self.alpha)*reg+(1-torch.sigmoid(self.alpha))*cs
        return dict(cls=cls,reg=reg,final=final,lv=self.logvar(h).squeeze(1).clamp(-4,2),aux=[z(h_m) for h_m,z in zip(hs,self.aux)],w=w,beta=beta,c=c)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--run-dir',required=True);p.add_argument('--fold',type=int,default=0);p.add_argument('--variant',choices=['base','ta','tapa','full'],default='full');p.add_argument('--tau',type=float,default=.5);p.add_argument('--lta',type=float,default=.03);p.add_argument('--lpa',type=float,default=.03);p.add_argument('--epochs',type=int,default=180);p.add_argument('--seed',type=int,default=20260715);a=p.parse_args();rd=Path(a.run_dir);rd.mkdir(parents=True,exist_ok=True);seed(a.seed);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 raw,sp=prepare_fold(Path(a.root)/'data'/f'fold_{a.fold}.pkl');xs,q,yc,yr,uni,ids=raw; xs=[torch.from_numpy(x).to(dev) for x in xs];q=torch.from_numpy(q).to(dev);yc=torch.from_numpy(yc).long().to(dev);yr=torch.from_numpy(yr).float().to(dev);uni=[torch.from_numpy(u).long().to(dev) for u in uni];tr,va,te=[torch.from_numpy(x).long().to(dev) for x in sp];m=MAGEFusion([x.shape[1] for x in xs],full=a.variant=='full').to(dev);opt=torch.optim.AdamW(m.parameters(),lr=6e-4,weight_decay=3e-4);best=-1e9;state=None;stale=0;log=[]
 for ep in range(1,a.epochs+1):
  m.train();perm=tr[torch.randperm(len(tr),device=dev)];lam=max(float(np.random.beta(.35,.35)),.5);out=m([lam*x[tr]+(1-lam)*x[perm] for x in xs],lam*q[tr]+(1-lam)*q[perm]);ct=lam*F.one_hot(yc[tr],5).float()+(1-lam)*F.one_hot(yc[perm],5).float();rt=lam*yr[tr]+(1-lam)*yr[perm];cs=(torch.softmax(out['cls'],1)*VALS.to(dev)).sum(1);loss=sce(out['cls'],ct)+.60*(.5*(torch.exp(-out['lv'])*(out['final']-rt).square()+out['lv']).mean())+.20*F.smooth_l1_loss(out['final'],rt)+.10*F.mse_loss(out['reg'],cs)+.08*sum(sce(z,lam*F.one_hot(u[tr],5).float()+(1-lam)*F.one_hot(u[perm],5).float()) for z,u in zip(out['aux'],uni))+.005*(out['w']*torch.log(out['w']+1e-8)).sum(1).mean()+.003*(out['beta']*torch.log(out['beta']+1e-8)).sum(1).mean()
  rawout=m([x[tr] for x in xs],q[tr]); yuni=torch.stack([VALS.to(dev)[u[tr]] for u in uni],1);target=torch.softmax(-(yuni-yr[tr,None]).abs()/a.tau,1);ta=F.kl_div(torch.log(rawout['w']+1e-8),target,reduction='batchmean'); pa_t=torch.stack([1-(yuni[:,0]-yuni[:,1]).abs()/2,1-(yuni[:,0]-yuni[:,2]).abs()/2,1-(yuni[:,1]-yuni[:,2]).abs()/2],1);pa=F.mse_loss(rawout['c'],pa_t)
  if a.variant in ('ta','tapa','full'): loss=loss+a.lta*ta
  if a.variant in ('tapa','full'): loss=loss+a.lpa*pa
  opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),4);opt.step();v=evaluate(m,xs,q,yr,va);score=v['Corr']-v['MAE']+.35*v['Acc2'];log.append({'epoch':ep,'score':score,'ta':ta.item(),'pa':pa.item(),**v})
  if score>best:best=score;state=copy.deepcopy(m.state_dict());stale=0
  else: stale+=1
  if stale>=30:break
 m.load_state_dict(state);res=evaluate(m,xs,q,yr,te);res.update({'fold':a.fold,'variant':a.variant,'tau':a.tau,'lambda_ta':a.lta,'lambda_pa':a.lpa,'best_valid_score':best,'epochs':ep});(rd/'metrics.json').write_text(json.dumps(res,indent=2));(rd/'valid_log.json').write_text(json.dumps(log,indent=2));print(json.dumps(res))
if __name__=='__main__':main()
