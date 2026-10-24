from progress import traced
import os,json,math,time,random,shutil,gc,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
from PIL import Image,ImageOps
from data import K,note,answer,reasoning,sha
from progress import log,stage,progress
SYSTEM=('Estimate the edible meal in the image, using user notes when supplied. '
'Preparation notes override visual guesses: respect explicit statements such as no added oil. '
'If no note is supplied, estimate from the image. Return exactly one final JSON object with '
'keys total_mass_g, calories_kcal, carbs_g, protein_g, fat_g. All values are finite nonnegative numbers. '
'Grams for mass/macros, kcal for energy. No markdown; no explanations inside the final JSON.')
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def amp():return torch.autocast('cuda',dtype=torch.bfloat16)
def messages(n):return [{'role':'system','content':SYSTEM},{'role':'user','content':[{'type':'image','image':None},{'type':'text','text':'Estimate the meal.\n'+n}]}]
def move(b,device='cuda'):return {k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in b.items()}
class Collator:
 def __init__(self,p,epoch=0,sft=False,variant=None,reason_prob=.35,max_length=1536):
  self.p=p;self.epoch=epoch;self.sft=sft;self.variant=variant;self.reason_prob=reason_prob;self.max_length=max_length
 def __call__(self,rows):
  images=[];prompts=[];targets=[]
  for r in rows:
   rng=random.Random(f"{r['id']}:{self.epoch}")
   variant=self.variant if self.variant is not None else rng.choices([0,1,2],[.5,.3,.2])[0]
   with Image.open(r['image']) as im:images.append(ImageOps.exif_transpose(im).convert('RGB').copy())
   why=reasoning(r) if self.sft and rng.random()<self.reason_prob else None
   prompts.append(self.p.apply_chat_template(messages(note(r,variant)),tokenize=False,add_generation_prompt=True,enable_thinking=bool(why)))
   targets.append((why+'\n</think>\n\n' if why else '')+answer(r)+'<|im_end|>')
  b=self.p(text=prompts,images=images,padding=True,return_tensors='pt')
  lens=b['attention_mask'].sum(-1).long();b['pool']=lens-1
  if self.sft:
   seq=[];labels=[];mm=[]
   for i,n in enumerate(lens.tolist()):
    t=torch.tensor(self.p.tokenizer.encode(targets[i],add_special_tokens=False),dtype=torch.long)
    q=b['input_ids'][i,:n];s=torch.cat([q,t]);lab=torch.cat([torch.full_like(q,-100),t])
    if len(s)>self.max_length:raise ValueError(f"{rows[i]['id']}: {len(s)} tokens exceed max_length. Increase it or reduce image budget; no silent truncation.")
    seq.append(s);labels.append(lab)
    if 'mm_token_type_ids' in b:mm.append(torch.cat([b['mm_token_type_ids'][i,:n],torch.zeros_like(t)]))
   b['input_ids']=nn.utils.rnn.pad_sequence(seq,batch_first=True,padding_value=self.p.tokenizer.pad_token_id)
   b['attention_mask']=nn.utils.rnn.pad_sequence([torch.ones_like(s) for s in seq],batch_first=True,padding_value=0)
   b['labels']=nn.utils.rnn.pad_sequence(labels,batch_first=True,padding_value=-100)
   if mm:b['mm_token_type_ids']=nn.utils.rnn.pad_sequence(mm,batch_first=True,padding_value=0)
  b['y']=torch.tensor([r['y'] for r in rows],dtype=torch.float32)
  b['weight']=torch.tensor([r.get('weight',1.) for r in rows],dtype=torch.float32)
  return b
class Model(nn.Module):
 def __init__(self,base,scale):
  super().__init__();self.base=base;h=base.config.text_config.hidden_size
  self.head=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,512),nn.SiLU(),nn.Linear(512,128),nn.SiLU(),nn.Linear(128,5))
  self.register_buffer('scale',torch.tensor(scale,dtype=torch.float32))
 def hidden(self,b):
  keys=['input_ids','attention_mask','pixel_values','image_grid_thw','mm_token_type_ids']
  return self.base.model(**{k:b[k] for k in keys if k in b},use_cache=False,return_dict=True).last_hidden_state
 def loss(self,b,reg=1.,ce=0.):
  h=self.hidden(b);loss=h.new_zeros((),dtype=torch.float32)
  if reg:
   pooled=h[torch.arange(len(h),device=h.device),b['pool']]
   pred=self.head(pooled).float()
   loss=loss+reg*(F.smooth_l1_loss(pred,b['y']/self.scale,beta=.25,reduction='none').mean(-1)*b['weight']).mean()
  if ce:
   labs=b['labels'][:,1:];mask=labs!=-100;z=h[:,:-1][mask];t=labs[mask]
   w=(b['weight']/mask.sum(-1).clamp_min(1))[:,None].expand_as(labs)[mask]
   total=h.new_zeros((),dtype=torch.float32)
   # Never allocate full prompt/image-token x vocabulary logits.
   for start in range(0,len(z),64):
    def part(x,tt,ww):
     with amp():logits=self.base.lm_head(x)
     return (F.cross_entropy(logits.float(),tt,reduction='none')*ww).sum()
    args=(z[start:start+64],t[start:start+64],w[start:start+64])
    total=total+(checkpoint(part,*args,use_reentrant=False) if self.training else part(*args))
   loss=loss+ce*total/len(h)
  return loss
STAGES=[
 dict(name='02_head',epochs=3,lr=0.,hlr=1e-3,reg=1.,ce=0.),
 dict(name='03_frozen_head',epochs=1,lr=3e-6,hlr=0.,reg=1.,ce=0.),
 dict(name='04_joint_bridge',epochs=2,lr=5e-6,hlr=1e-4,reg=.5,ce=1.),
 dict(name='05_json',epochs=1,lr=2e-6,hlr=0.,reg=0.,ce=1.)]
def stage_setup(net,s,cfg):
 net.requires_grad_(False)
 if s['name']!='02_head':
  text=net.base.model.language_model;vision=net.base.model.visual
  assert 1<=cfg['text_layers']<=len(text.layers) and 1<=cfg['vision_layers']<=len(vision.blocks), 'Invalid unfrozen-layer count'
  for x in text.layers[-cfg['text_layers']:]:x.requires_grad_(True)
  for x in vision.blocks[-cfg['vision_layers']:]:x.requires_grad_(True)
  text.norm.requires_grad_(True);vision.merger.requires_grad_(True)
 if s['name'] in ['02_head','04_joint_bridge']:net.head.requires_grad_(True)
 if s['name']=='05_json':net.head=None
 if cfg.get('gradient_checkpointing') and s['name']!='02_head':net.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 else:net.base.gradient_checkpointing_disable()
 net.train()
 if s['name']=='02_head':net.base.eval()
 if s['name']=='03_frozen_head':net.head.eval()
 n=sum(p.numel() for p in net.parameters() if p.requires_grad)
 assert n>0
 print(s['name'],'trainable:',f'{n:,}',flush=True)
def loader(rows,collator,batch,epoch,seed_value,workers=2):
 indices=list(range(len(rows)));rng=random.Random(seed_value+epoch);rng.shuffle(indices)
 batches=[]
 for start in range(0,len(indices),batch*24):
  chunk=sorted(indices[start:start+batch*24],key=lambda i:len(rows[i].get('ingredients',[])))
  batches.extend([chunk[j:j+batch] for j in range(0,len(chunk),batch)])
 rng.shuffle(batches)
 return DataLoader(rows,batch_sampler=batches,collate_fn=collator,num_workers=workers,pin_memory=True,
                   generator=torch.Generator().manual_seed(seed_value+epoch))
def metrics(ys,ps):
 y=np.asarray(ys,float);p=np.asarray(ps,float)
 if not len(y):return {'n':0}
 a=np.abs(y-p);out={'n':len(y)}
 for j,k in enumerate(K):out[k]=dict(mae=float(a[:,j].mean()),rmse=float(np.sqrt(((y[:,j]-p[:,j])**2).mean())),wape_pct=float(100*a[:,j].sum()/max(y[:,j].sum(),1e-8)))
 return out
def parse(text):
 try:
  o=json.loads(text.split('</think>')[-1].strip())
  if not isinstance(o,dict) or set(o)!=set(K):return None
  yy=[o[k] for k in K]
  if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 for x in yy):return None
  return yy
 except (ValueError,TypeError):return None
@torch.inference_mode()
def generate(base,p,row,user_note='',thinking=False):
 with Image.open(row['image']) as im:im=ImageOps.exif_transpose(im).convert('RGB')
 text=p.apply_chat_template(messages(user_note),tokenize=False,add_generation_prompt=True,enable_thinking=thinking)
 b=p(text=[text],images=[im],return_tensors='pt').to('cuda');base.eval()
 with amp():out=base.generate(**b,max_new_tokens=768 if thinking else 160,do_sample=False,use_cache=True)
 text=p.tokenizer.decode(out[0,b['input_ids'].shape[1]:],skip_special_tokens=False).replace('<|im_end|>','').replace('<|endoftext|>','').strip()
 return text,parse(text)
@torch.inference_mode()
@traced("evaluate")
def evaluate(base,p,rows,path=None,thinking=False,variant=0):
 old={}
 if path and Path(path).exists():
  lines=Path(path).read_bytes().splitlines(keepends=True);good_bytes=0
  for i,line in enumerate(lines):
   try:r=json.loads(line)
   except (ValueError,UnicodeDecodeError):
    if i!=len(lines)-1:raise ValueError('Corrupt evaluation journal before final line')
    with open(path,'r+b') as f:f.truncate(good_bytes)
    print('Removed interrupted final evaluation record');break
   old[r['id']]=r;good_bytes+=len(line)
  if lines and good_bytes and not Path(path).read_bytes().endswith(b'\n'):
   with open(path,'ab') as f:f.write(b'\n')
 records=[]
 for i,r in enumerate(progress(rows,desc=f'Evaluation thinking={thinking}, text={variant}',unit='photo')):
  if r['id'] in old:x=old[r['id']]
  else:
   text,pred=generate(base,p,r,note(r,variant),thinking)
   x=dict(id=r['id'],source=r['source'],true=r['y'],pred=pred,raw=text)
   if path:
    with open(path,'a') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n')
  records.append(x)
  if (i+1)%25==0:print('eval',i+1,'/',len(rows),flush=True)
 by=defaultdict(list)
 for r in records:by[r['source']].append(r)
 out={}
 for source,rr in [('all',records)]+list(by.items()):
  valid=[x for x in rr if x['pred'] is not None];m=metrics([x['true'] for x in valid],[x['pred'] for x in valid])
  m.update(total=len(rr),invalid_json=len(rr)-len(valid),valid_rate=len(valid)/max(1,len(rr)))
  m['score']=(sum(m[k]['wape_pct'] for k in K)/5 if valid else 1000)+1000*(1-m['valid_rate']);out[source]=m
 return out
@torch.inference_mode()
def eval_head(net,p,rows,cfg):
 net.eval();yy=[];pp=[]
 for b in progress(loader(rows,Collator(p,variant=0),cfg['batch'],0,cfg['seed'],cfg['workers']),desc='MLP validation',unit='batch'):
  b=move(b)
  with amp():h=net.hidden(b);pred=net.head(h[torch.arange(len(h),device=h.device),b['pool']]).float()*net.scale
  yy+=b['y'].cpu().tolist();pp+=pred.clamp_min(0).cpu().tolist()
 m=metrics(yy,pp);m['score']=sum(m[k]['wape_pct'] for k in K)/5;return m
def val_subset(rows,n):
 by=defaultdict(list)
 for r in rows:by[r['source']].append(r)
 return [r for source,rr in sorted(by.items()) for r in sorted(rr,key=lambda x:hashlib.sha256(x['id'].encode()).hexdigest())[:n]]
class Checkpoints:
 def __init__(self,root,local,signature):self.root=Path(root);self.local=Path(local);self.signature=signature;self.root.mkdir(parents=True,exist_ok=True);self.local.mkdir(parents=True,exist_ok=True)
 def save(self,net,opt,sched,progress):
  name='ckpt_'+str(time.time_ns());local=self.local/'state.pt';dest=self.root/name;dest.mkdir()
  state=dict(model=net.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),progress=progress,signature=self.signature,
             python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
  with stage('Checkpoint: serialize local state'):torch.save(state,local)
  del state
  with stage('Checkpoint: copy to Drive'):shutil.copy2(local,dest/'state.pt')
  log('Checkpoint: verifying SHA256 on local and Drive copies');checksum=sha(local)
  if sha(dest/'state.pt')!=checksum:raise IOError('Checkpoint copy failed SHA256')
  (dest/'COMPLETE.tmp').write_text(json.dumps(dict(signature=self.signature,sha256=checksum,progress=progress)))
  os.replace(dest/'COMPLETE.tmp',dest/'COMPLETE.json')
  local.unlink();old=sorted(p for p in self.root.glob('ckpt_*') if (p/'COMPLETE.json').exists())
  for p in old[:-2]:shutil.rmtree(p)
  print('Checkpoint complete:',progress,flush=True)
 def load(self):
  candidates=[]
  for p in self.root.glob('ckpt_*'):
   if (p/'COMPLETE.json').exists():
    try:m=json.loads((p/'COMPLETE.json').read_text())
    except (ValueError,OSError):continue
    if m['signature']==self.signature:candidates.append(p)
  if not candidates:return None
  for src in sorted(candidates,reverse=True):
   m=json.loads((src/'COMPLETE.json').read_text());local=self.local/'resume.pt'
   try:
    with stage('Restore checkpoint from Drive'):shutil.copy2(src/'state.pt',local)
    if sha(local)!=m['sha256']:raise IOError('SHA256 mismatch')
   except (OSError,IOError) as e:
    print('Skipping incomplete/corrupt checkpoint:',src.name,str(e));local.unlink(missing_ok=True);continue
   d=torch.load(local,map_location='cpu',weights_only=False);local.unlink();return d
  raise IOError('No intact matching checkpoint; refusing to restart silently.')

def best_save(net,root,name,score):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);meta=root/(name+'.json')
 if meta.exists() and json.loads(meta.read_text())['score']<=score:return
 # FP32 preserves exact parameters for stage handoff (not a deployment quantization).
 tmp=root/(name+'.tmp')
 with stage('Save best weights: '+name):torch.save(net.state_dict(),tmp);os.replace(tmp,root/(name+'.pt'))
 meta.write_text(json.dumps(dict(score=score)))
def best_load(net,root,name):
 p=Path(root)/(name+'.pt')
 if p.exists():
  with stage('Load best weights: '+name):net.load_state_dict(torch.load(p,map_location='cpu',weights_only=True))

@traced("cache_features")
def cache_features(net,p,rows,cfg):
 path=Path(cfg['local'])/'head_features.pt'
 if path.exists():return torch.load(path,map_location='cpu',weights_only=True)
 net.eval();xx=[];yy=[];ww=[]
 with torch.inference_mode():
  for variant in [0,1,2]:
   log(f'Head feature cache: variant {variant+1}/3')
   for b in progress(loader(rows,Collator(p,variant=variant),cfg['batch'],0,cfg['seed'],cfg['workers']),desc=f'Head feature cache {variant+1}/3',unit='batch'):
    b=move(b)
    with amp():h=net.hidden(b)
    xx.append(h[torch.arange(len(h),device=h.device),b['pool']].cpu().to(torch.bfloat16));yy.append(b['y'].cpu());ww.append(b['weight'].cpu()*(1.5 if variant==0 else .75))
 c=dict(x=torch.cat(xx),y=torch.cat(yy),w=torch.cat(ww));tmp=path.with_suffix('.tmp');torch.save(c,tmp);os.replace(tmp,path);return c

@traced("fit")
def fit(net,p,train,val,cfg,cp):
 stages=[dict(s) for s in (STAGES[-1:] if cfg['direct_json_only'] else STAGES)]
 if cfg['smoke']:stages=[dict(s,epochs=1) for s in stages]
 histpath=Path(cfg['drive'])/'history.json';history=json.loads(histpath.read_text()) if histpath.exists() else []
 resume=cp.load();start=0
 if resume:
  start=resume['progress']['stage']
  if not any(k.startswith('head.') for k in resume['model']):net.head=None
  net.load_state_dict(resume['model'])
  if resume['progress'].get('complete'):print('Already complete');return
 for si,s in enumerate(stages):
  if si<start:continue
  stage_setup(net,s,cfg);groups=[]
  params=[x for x in net.base.parameters() if x.requires_grad]
  if params:groups.append(dict(params=params,lr=s['lr']))
  if net.head is not None:
   params=[x for x in net.head.parameters() if x.requires_grad]
   if params:groups.append(dict(params=params,lr=s['hlr']))
  opt=torch.optim.AdamW(groups,betas=(.9,.95),weight_decay=.01,fused=True)
  cached=cache_features(net,p,train,cfg) if s['name']=='02_head' else None
  n=math.ceil(len(cached['x'])/256) if cached is not None else math.ceil(len(train)/cfg['batch'])
  accum=1 if cached is not None else cfg['accum'];steps=s['epochs']*math.ceil(n/accum);warm=max(1,int(.05*steps))
  def rate(step):return (step+1)/warm if step<warm else .1+.9*.5*(1+math.cos(math.pi*min(1,(step-warm)/max(1,steps-warm))))
  sched=torch.optim.lr_scheduler.LambdaLR(opt,rate);epoch0=0;skip=0;step=0
  if resume and si==start:
   opt.load_state_dict(resume['optimizer']);sched.load_state_dict(resume['scheduler']);pr=resume['progress']
   epoch0=pr['epoch'];skip=pr['next_batch'];step=pr['step']
   random.setstate(resume['python_rng']);np.random.set_state(resume['numpy_rng']);torch.set_rng_state(resume['torch_rng']);torch.cuda.set_rng_state_all(resume['cuda_rng'])
   resume=None;gc.collect()
  for epoch in range(epoch0,s['epochs']):
   stage_setup(net,s,cfg);opt.zero_grad(set_to_none=True);last_save=time.time()
   if cached is not None:
    ids=torch.randperm(len(cached['x']),generator=torch.Generator().manual_seed(cfg['seed']+epoch));dl=[ids[j:j+256] for j in range(0,len(ids),256)]
   else:dl=loader(train,Collator(p,epoch=si*100+epoch,sft=bool(s['ce']),reason_prob=cfg['reason_prob'],max_length=cfg['max_length']),cfg['batch'],si*100+epoch,cfg['seed'],cfg['workers'])
   for bi,b in enumerate(progress(dl,desc=f"{s['name']} epoch {epoch+1}/{s['epochs']}",unit='batch')):
    if epoch==epoch0 and bi<skip:continue
    if cached is not None:
     x=cached['x'][b].to('cuda',dtype=torch.float32);y=cached['y'][b].to('cuda');w=cached['w'][b].to('cuda')
     with amp():pred=net.head(x).float()
     loss=(F.smooth_l1_loss(pred,y/net.scale,reduction='none',beta=.25).mean(-1)*w).mean()
    else:
     b=move(b);fraction=(epoch+bi/max(n,1))/s['epochs'];reg=s['reg']*(1-.8*fraction) if s['name']=='04_joint_bridge' else s['reg']
     with amp():loss=net.loss(b,reg,s['ce'])
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss; resume last complete checkpoint.')
    window=min(accum,n-(bi//accum)*accum);(loss/window).backward()
    if (bi+1)%accum==0 or bi+1==n:
     torch.nn.utils.clip_grad_norm_([x for x in net.parameters() if x.requires_grad],1.,error_if_nonfinite=True)
     opt.step();sched.step();opt.zero_grad(set_to_none=True);step+=1
     if step%10==0:print(s['name'],'epoch',epoch+1,'batch',bi+1,'/',n,'loss',round(float(loss),4),flush=True)
     if time.time()-last_save>60*cfg['save_minutes']:
      cp.save(net,opt,sched,dict(stage=si,epoch=epoch,next_batch=bi+1,step=step));last_save=time.time()
     if cfg['smoke'] and step>=2:break
   skip=0;vv=val_subset(val,cfg['val_per_source'])
   m=evaluate(net.base,p,vv)['all'] if s['ce'] else eval_head(net,p,vv,cfg)
   history.append(dict(stage=s['name'],epoch=epoch,metrics=m));histpath.write_text(json.dumps(history,indent=2))
   best_save(net,Path(cfg['drive'])/'best',s['name'],m['score'])
   cp.save(net,opt,sched,dict(stage=si,epoch=epoch+1,next_batch=0,step=step))
  best_load(net,Path(cfg['drive'])/'best',s['name'])
  if si==len(stages)-1:cp.save(net,opt,sched,dict(stage=si,epoch=s['epochs'],next_batch=0,step=step,complete=True))
  del opt,sched,cached;gc.collect();torch.cuda.empty_cache()
