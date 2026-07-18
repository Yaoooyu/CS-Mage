import argparse,copy,csv,json,random,time
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from run_mage_fusion_fold_v2 import MAGEFusion,VALS
from run_tri_ride import prepare_fold
from run_uncertainty_graph_fivefold import sce
from run_interaction_fivefold import sims_metrics
P=((0,1),(0,2),(1,2))
def seed(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def flags(v):
 return {'only_lta':(1,0,0,0,1,1),'lta_lpa':(1,1,0,1,1,1),'no_agreement_bias':(1,1,0,1,1,1),'no_disagreement_expert':(1,1,1,1,0,1),'no_mixup':(1,1,1,1,1,0),'no_annotation_guidance':(0,0,1,1,1,1),'full':(1,1,1,1,1,1)}[v]
class Model(MAGEFusion):
 def __init__(self,dims,f):
  super().__init__(dims,full=True);self.f=f
  if not f[4]: self.egate[-1]=torch.nn.Linear(192,2)
 def forward(self,xs,q):
  ta,pa,bias,pair,dis,mix=self.f;hs=[m(x) for m,x in zip(self.enc,xs)];st=torch.stack(hs,1);a=self.evidence(torch.cat(hs+[q],1));b=torch.cat([m(torch.cat([h,q[:,i:i+1]],1)) for i,(h,m) in enumerate(zip(hs,self.ambiguity))],1).clamp(-3,3);w=torch.softmax(a-b,1)
  fs=[torch.cat([hs[i]+hs[j],(hs[i]-hs[j]).abs(),hs[i]*hs[j]],1) for i,j in P];c=torch.sigmoid(torch.cat([self.pair_net(z) for z in fs],1));cg=c if pair else torch.zeros_like(c)
  s=torch.matmul(self.q(st),self.k(st).transpose(1,2))/np.sqrt(st.size(-1));B=torch.zeros_like(s);B[:,0,1]=B[:,1,0]=2*c[:,0]-1;B[:,0,2]=B[:,2,0]=2*c[:,1]-1;B[:,1,2]=B[:,2,1]=2*c[:,2]-1;att=torch.softmax(s+(F.softplus(self.gamma_raw)*B if bias else 0),2);g=self.gnorm(st+torch.matmul(att,self.v(st)));ea=(w.unsqueeze(-1)*g).sum(1);pos=[];neg=[]
  for z,(i,j) in enumerate(P):
   rw=torch.sqrt(w[:,i:i+1]*w[:,j:j+1]);pos+=[(cg[:,z:z+1]*rw if pair else 1)*g[:,i]*g[:,j]];neg+=[((1-cg[:,z:z+1])*rw if pair else 1)*(g[:,i]-g[:,j]).abs()]
  ep=self.pair(torch.cat(pos,1));ed=self.diff(torch.cat(neg,1));inp=torch.cat([ea,ep,ed,w,b,(cg if pair else torch.zeros_like(c))],1);br=self.egate(inp)
  if dis: beta=torch.softmax(br,1);fuse=beta[:,:1]*ea+beta[:,1:2]*ep+beta[:,2:]*ed
  else: bb=torch.softmax(br,1);beta=torch.cat([bb,torch.zeros_like(bb[:,:1])],1);fuse=beta[:,:1]*ea+beta[:,1:2]*ep
  h=self.head(fuse);cl=self.cls(h);reg=torch.tanh(self.mean(h)).squeeze(1);probs=torch.softmax(cl,1);final=torch.sigmoid(self.alpha)*reg+(1-torch.sigmoid(self.alpha))*(probs*VALS.to(h.device)).sum(1)
  return {'cls':cl,'reg':reg,'final':final,'lv':self.logvar(h).squeeze(1).clamp(-4,2),'aux':[z(x) for x,z in zip(hs,self.aux)],'w':w,'b':b,'c':c,'beta':beta,'probs':probs,'attn':att}
def targets(uni,y,ix,tau,d):
 u=torch.stack([VALS.to(d)[z[ix]] for z in uni],1);tw=torch.softmax(-(u-y[ix,None]).abs()/tau,1);tc=torch.stack([1-(u[:,0]-u[:,1]).abs()/2,1-(u[:,0]-u[:,2]).abs()/2,1-(u[:,1]-u[:,2]).abs()/2],1);return tw,tc
def met(o,y,ix):return sims_metrics(o['final'].detach().cpu().numpy(),y[ix].detach().cpu().numpy())
def save(path,o,ids,fold,yc,y,uni,ix,tau):
 z={k:v.detach().cpu().numpy() for k,v in o.items() if isinstance(v,torch.Tensor)};tw,tc=targets(uni,y,ix,tau,y.device);tw,tc=tw.cpu().numpy(),tc.cpu().numpy();truth=y[ix].cpu().numpy();cl=z['cls'].argmax(1);pr=z['final'];B=lambda x,e:np.digitize(x,e[1:-1],right=True);a2=B(pr,[-1.01,0,1.01])==B(truth,[-1.01,0,1.01]);a3=B(pr,[-1.01,-.1,.1,1.01])==B(truth,[-1.01,-.1,.1,1.01]);a5=B(pr,[-1.01,-.7,-.1,.1,.7,1.01])==B(truth,[-1.01,-.7,-.1,.1,.7,1.01])
 F=['sample_id','fold','y','y_class','y_text','y_audio','y_visual','prediction']+[f'prob_{i}' for i in range(1,6)]+['predicted_class','predicted_binary','predicted_three_class']+[f'w_{n}' for n in ['text','audio','visual']]+[f'target_w_{n}' for n in ['text','audio','visual']]+[f'c_{n}' for n in ['text_audio','text_visual','audio_visual']]+[f'target_c_{n}' for n in ['text_audio','text_visual','audio_visual']]+[f'beta_{n}' for n in ['aggregation','interaction','disagreement']]+['absolute_error','acc2_correct','acc3_correct','acc5_correct']
 with open(path,'w',newline='') as h:
  wr=csv.DictWriter(h,F);wr.writeheader();ii=ix.cpu().numpy()
  for j,k in enumerate(ii):
   r={'sample_id':str(ids[k]),'fold':fold,'y':truth[j],'y_class':int(yc[k])+1,'y_text':int(uni[0][k])+1,'y_audio':int(uni[1][k])+1,'y_visual':int(uni[2][k])+1,'prediction':pr[j],'predicted_class':int(cl[j])+1,'predicted_binary':int(B(pr[j:j+1],[-1.01,0,1.01])[0]),'predicted_three_class':int(B(pr[j:j+1],[-1.01,-.1,.1,1.01])[0]),'absolute_error':abs(pr[j]-truth[j]),'acc2_correct':int(a2[j]),'acc3_correct':int(a3[j]),'acc5_correct':int(a5[j])}
   for n in range(5):r[f'prob_{n+1}']=z['probs'][j,n]
   for n,k2 in enumerate(['text','audio','visual']):r[f'w_{k2}']=z['w'][j,n];r[f'target_w_{k2}']=tw[j,n]
   for n,k2 in enumerate(['text_audio','text_visual','audio_visual']):r[f'c_{k2}']=z['c'][j,n];r[f'target_c_{k2}']=tc[j,n]
   for n,k2 in enumerate(['aggregation','interaction','disagreement']):r[f'beta_{k2}']=z['beta'][j,n]
   wr.writerow(r)
def main(a):
 f=flags(a.variant);rd=Path(a.run_dir)/a.variant/f'fold_{a.fold}';rd.mkdir(parents=True);seed(a.seed);d=torch.device('cuda');raw,sp=prepare_fold(Path(a.root)/'data'/f'fold_{a.fold}.pkl');xs,q,yc,y,uni,ids=raw;xs=[torch.from_numpy(z).to(d) for z in xs];q=torch.from_numpy(q).to(d);yc=torch.from_numpy(yc).long().to(d);y=torch.from_numpy(y).float().to(d);uni=[torch.from_numpy(z).long().to(d) for z in uni];tr,va,te=[torch.from_numpy(z).long().to(d) for z in sp];m=Model([z.shape[1] for z in xs],f).to(d);op=torch.optim.AdamW(m.parameters(),lr=6e-4,weight_decay=3e-4);best=-1e9;state=None;stale=0;hist=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats(d)
 for e in range(1,a.epochs+1):
  t=time.perf_counter();m.train();pm=tr[torch.randperm(len(tr),device=d)]
  if f[5]:lam=max(float(np.random.beta(.35,.35)),.5);mx=[lam*z[tr]+(1-lam)*z[pm] for z in xs];mq=lam*q[tr]+(1-lam)*q[pm];ct=lam*F.one_hot(yc[tr],5).float()+(1-lam)*F.one_hot(yc[pm],5).float();rt=lam*y[tr]+(1-lam)*y[pm]
  else:lam=1.;mx=[z[tr] for z in xs];mq=q[tr];ct=F.one_hot(yc[tr],5).float();rt=y[tr]
  o=m(mx,mq);cs=(torch.softmax(o['cls'],1)*VALS.to(d)).sum(1);loss=sce(o['cls'],ct)+.6*(.5*(torch.exp(-o['lv'])*(o['final']-rt).square()+o['lv']).mean())+.2*F.smooth_l1_loss(o['final'],rt)+.1*F.mse_loss(o['reg'],cs)+.08*sum(sce(z,lam*F.one_hot(u[tr],5).float()+(1-lam)*F.one_hot(u[pm],5).float()) for z,u in zip(o['aux'],uni))+.005*(o['w']*torch.log(o['w']+1e-8)).sum(1).mean()+.003*(o['beta']*torch.log(o['beta']+1e-8)).sum(1).mean();ro=m([z[tr] for z in xs],q[tr]);tw,tc=targets(uni,y,tr,a.tau,d);lt=F.kl_div(torch.log(ro['w']+1e-8),tw,reduction='batchmean');lp=F.mse_loss(ro['c'],tc)
  if f[0]:loss=loss+a.lta*lt
  if f[1]:loss=loss+a.lpa*lp
  op.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),4);op.step();m.eval()
  with torch.no_grad():vo=m([z[va] for z in xs],q[va]);v=met(vo,y,va)
  score=v['Corr']-v['MAE']+.35*v['Acc2'];hist+=[dict(epoch=e,loss=float(loss),score=score,ta=float(lt),pa=float(lp),epoch_seconds=time.perf_counter()-t,**v)]
  if score>best:best=score;state=copy.deepcopy(m.state_dict());be=e;stale=0;torch.save({'state_dict':state,'epoch':e,'score':best,'flags':f},rd/'best_checkpoint.pt')
  else:stale+=1
  if stale>=30:break
 torch.save({'state_dict':m.state_dict(),'epoch':e,'flags':f},rd/'last_checkpoint.pt');m.load_state_dict(state);m.eval()
 with torch.no_grad():o=m([z[te] for z in xs],q[te]);r=met(o,y,te)
 save(rd/'test_predictions.csv',o,ids,a.fold,yc,y,uni,te,a.tau);r.update({'fold':a.fold,'variant':a.variant,'tau':a.tau,'lambda_ta':a.lta if f[0] else 0,'lambda_pa':a.lpa if f[1] else 0,'best_valid_score':best,'best_epoch':be,'epochs':e,'train_seconds':time.perf_counter()-start,'peak_gpu_memory_mb':torch.cuda.max_memory_allocated(d)/1048576,'flags':f});(rd/'metrics.json').write_text(json.dumps(r,indent=2));(rd/'validation_history.json').write_text(json.dumps(hist,indent=2));(rd/'command.json').write_text(json.dumps(vars(a),indent=2));print(json.dumps(r),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--run-dir',required=True);p.add_argument('--fold',type=int,required=True);p.add_argument('--variant',choices=['only_lta','lta_lpa','no_agreement_bias','no_disagreement_expert','no_mixup','no_annotation_guidance','full'],required=True);p.add_argument('--seed',type=int,default=20260715);p.add_argument('--epochs',type=int,default=180);p.add_argument('--tau',type=float,default=1.0);p.add_argument('--lta',type=float,default=.01);p.add_argument('--lpa',type=float,default=.01);main(p.parse_args())
