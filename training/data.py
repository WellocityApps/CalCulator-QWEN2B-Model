from progress import traced
"""Nutrition source adapters; real numeric targets only; no missing-value imputation."""
import csv,json,math,hashlib,random,re,time,shutil,tarfile,zipfile,subprocess,urllib.request,urllib.parse
from pathlib import Path
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from progress import log,stage,progress,parallel_map,tqdm
K=['total_mass_g','calories_kcal','carbs_g','protein_g','fat_g']
N5='https://storage.googleapis.com/nutrition5k_dataset/'
SNAP='https://raw.githubusercontent.com/JulesLarke-USDA/SNAPMe/ce09cee4a1e0c5b05f35abe5a8a058e686fefa74/input/master_SNAPME_linkfile.csv'
def sha(p):
 p=Path(p);size=p.stat().st_size;large=size>=64*2**20;h=hashlib.sha256()
 if large:log('Checksum: '+p.name)
 with open(p,'rb') as f,tqdm(total=size,desc='SHA256 '+p.name,unit='B',unit_scale=True,mininterval=1,disable=not large) as bar:
  for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk);bar.update(len(chunk))
 return h.hexdigest()
def fetch(url,path,show=False):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and p.stat().st_size:
  if show:log(f'CACHE {p.name}: {p.stat().st_size:,} bytes')
  return p
 for attempt in range(3):
  tmp=p.with_name(p.name+'.part')
  try:
   if show:log(f'Download {p.name}: connecting (attempt {attempt+1}/3, socket timeout 120s)')
   with urllib.request.urlopen(url,timeout=120) as r,open(tmp,'wb') as f:
    size=int(r.headers.get('Content-Length',0)) or None;received=0;last=time.monotonic()
    with tqdm(total=size,desc=p.name,unit='B',unit_scale=True,mininterval=1,disable=not show) as bar:
     for chunk in iter(lambda:r.read(2**20),b''):
      f.write(chunk);received+=len(chunk);bar.update(len(chunk))
      if show and time.monotonic()-last>=20:
       log(f'{p.name}: {received:,}/{size or "unknown"} bytes');last=time.monotonic()
    if size is not None and received!=size:raise IOError(f'Incomplete download: {received}/{size} bytes')
   tmp.replace(p)
   if show:log(f'Download complete: {p.name}, {received:,} bytes')
   return p
  except Exception as e:
   tmp.unlink(missing_ok=True)
   log(f'Download failed: {p.name}, attempt {attempt+1}/3, {type(e).__name__}: {e}')
   if attempt==2:raise
   time.sleep(2)
def values(y):return len(y)==5 and all(math.isfinite(x) and x>=0 for x in y) and y[0]>0

def parse_n5(p,cafe):
 out=[];bad=Counter()
 for row in csv.reader(open(p,newline='')):
  if not row or not row[0].startswith('dish_'):continue
  try:
   cal,m,f,c,pr=map(float,row[1:6]);y=[m,cal,c,pr,f]
   if not values(y):raise ValueError('invalid totals')
   tail=row[6:]
   while tail and not tail[-1].strip():tail.pop()
   if tail and re.fullmatch(r'\d+',tail[0]):
    count=int(tail.pop(0))
    if len(tail)!=count*7:raise ValueError('count mismatch')
   if len(tail)%7:raise ValueError('ingredient stride')
   ingredients=[]
   for j in range(0,len(tail),7):
    iid,name,m,cal,f,c,pr=tail[j:j+7];yy=list(map(float,[m,cal,c,pr,f]))
    if not iid.startswith('ingr_') or not values(yy):raise ValueError('invalid ingredient')
    ingredients.append(dict(name=name,y=yy))
   stamp=int(row[0].split('_')[1]);day=datetime.fromtimestamp(stamp,timezone.utc).date()
   good=bool(ingredients) and all(abs(sum(x['y'][j] for x in ingredients)-y[j])<=max(2,.03*y[j]) for j in range(5))
   out.append(dict(id=row[0],source='nutrition5k',group=f'n5:{cafe}:{day}',cafe=cafe,timestamp=stamp,
                   y=y,ingredients=ingredients,reason_ok=good,weight=1.,basis='weighed_plate'))
  except (ValueError,IndexError) as e:bad[str(e)]+=1
 return out,dict(bad)
def objects(prefix):
 token=None;page=0;count=0
 log("N5 listing: "+prefix)
 while True:
  q=dict(prefix=prefix,maxResults=1000,fields='items(name,size),nextPageToken')
  if token:q['pageToken']=token
  with urllib.request.urlopen('https://storage.googleapis.com/storage/v1/b/nutrition5k_dataset/o?'+urllib.parse.urlencode(q),timeout=120) as r:d=json.load(r)
  page+=1;count+=len(d.get('items',[]));log(f'N5 listing page {page}: {count:,} objects scanned')
  yield from d.get('items',[])
  token=d.get('nextPageToken')
  if not token:break

@traced("nutrition5k")
def nutrition5k(root,fill_side=True):
 log('Nutrition5k: metadata, object listing, overhead RGB, then missing side frames')
 root=Path(root);rows=[];audit={}
 for cafe in ['cafe1','cafe2']:
  p=fetch(N5+f'nutrition5k_dataset/metadata/dish_metadata_{cafe}.csv',root/f'{cafe}.csv',show=True)
  rr,bad=parse_n5(p,cafe);log(f'N5 {cafe}: {len(rr)} valid rows, rejected={bad}');rows+=rr;audit[cafe]=dict(rejected=bad,sha256=sha(p))
 byid={r['id']:r for r in rows};rows=[]
 oo=[o for o in objects('nutrition5k_dataset/imagery/realsense_overhead/') if o['name'].endswith('/rgb.png') and o['name'].split('/')[-2] in byid]
 def rgb(o):
  r=byid[o['name'].split('/')[-2]];r['image']=str(fetch(N5+o['name'],root/'rgb'/f"{r['id']}.png"));return r
 rows=parallel_map(rgb,oo,12,'N5 overhead RGB')
 if fill_side:
  have={r['id'] for r in rows}
  oo=[o for o in objects('nutrition5k_dataset/imagery/side_angles/') if o['name'].endswith('/camera_A.h264') and o['name'].split('/')[-2] in byid and o['name'].split('/')[-2] not in have]
  def side(o):
   r=byid[o['name'].split('/')[-2]];p=root/'rgb'/f"{r['id']}_side.jpg";p.parent.mkdir(parents=True,exist_ok=True)
   if not p.exists():
    tmp=p.with_name(p.stem+'.part.jpg')
    subprocess.run(['ffmpeg','-nostdin','-loglevel','error','-rw_timeout','90000000','-i',N5+o['name'],'-frames:v','1','-q:v','2','-y',str(tmp)],check=True,timeout=120)
    tmp.replace(p)
   r['image']=str(p);return r
  rows+=parallel_map(side,oo,4,'N5 side frame extraction')
 # Capturing-day groups are more conservative than a random per-dish split.
 # Merge adjacent days where capture timestamps are less than an hour apart.
 alias={}
 ordered=sorted(rows,key=lambda r:(r['cafe'],r['timestamp']))
 for a,b in zip(ordered,ordered[1:]):
  if a['cafe']==b['cafe'] and b['timestamp']-a['timestamp']<3600 and a['group']!=b['group']:alias[b['group']]=alias.get(a['group'],a['group'])
 for r in rows:
  while r['group'] in alias:r['group']=alias[r['group']]
 audit['loaded']=len(rows);audit['missing_rgb']=len(byid)-len(rows)
 log(f'Nutrition5k complete: {len(rows)} images; missing={audit["missing_rgb"]}')
 return rows,audit

def parse_snap(p):
 reader=csv.DictReader(open(p,newline='',encoding='utf-8-sig'))
 required={'subject_id','filename','packaged_food','Food_Description','FoodAmt','KCAL','CARB','PROT','TFAT'}
 if not required.issubset(reader.fieldnames or []):raise ValueError('SNAPMe schema changed; required labels missing.')
 groups=defaultdict(list)
 for row in reader:groups[(row['subject_id'],row['filename'])].append(row)
 out=[];bad=0;excluded_packaged=0
 for (sid,name),items in groups.items():
  try:
   # Decide at PHOTO level: dropping individual packaged items creates partial labels.
   if any(float(x['packaged_food'])!=0 for x in items):
    excluded_packaged+=1;continue
   ingredients=[]
   for x in items:
    # FoodAmt is grams; G_TOTAL is a grain food-group measure, NOT food weight.
    yy=[float(x[k]) for k in ['FoodAmt','KCAL','CARB','PROT','TFAT']]
    if not values(yy):raise ValueError('incomplete item')
    ingredients.append(dict(name=x['Food_Description'].strip(),y=yy))
   y=[sum(x['y'][j] for x in ingredients) for j in range(5)]
   out.append(dict(id='snap:'+sid+':'+name,filename=name,source='snapme',group='snap:'+sid,y=y,
                   ingredients=ingredients,reason_ok=True,weight=.30,basis='self_reported_consumed_before_photo_proxy',
                   verified_notes=['Preparation note for this component: '+x['name'] for x in ingredients if re.search(r'fat not added in cooking|made without fat',x['name'],re.I)]))
  except (ValueError,TypeError):bad+=1
 return out,dict(incomplete_photo=bad,excluded_packaged_or_mixed_photo=excluded_packaged)

@traced("snapme")
def snapme(root,local_archive=None):
 log('SNAPMe: loading metadata')
 root=Path(root);p=fetch(SNAP,root/'master.csv',show=True);rows,audit=parse_snap(p)
 log(f'SNAPMe: {len(rows)} candidate photos')
 image_dir=root/'before';image_dir.mkdir(parents=True,exist_ok=True);want={r['filename'] for r in rows}
 if not want.issubset({p.name for p in image_dir.iterdir()}):
  if local_archive:archive=Path(local_archive)
  else:
   log('SNAPMe: requesting Figshare download metadata')
   with urllib.request.urlopen('https://api.figshare.com/v2/articles/24856449',timeout=120) as r:info=json.load(r)
   ent=next(x for x in info['files'] if x['name']=='snapme_db_09Dec2022.tar.gz')
   archive=fetch(ent['download_url'],root/ent['name'],show=True)
   if archive.stat().st_size!=ent['size']:raise ValueError('SNAPMe incomplete archive')
  log(f'SNAPMe: scanning compressed archive {archive.name}; this requires reading the archive stream')
  with tarfile.open(archive,'r|gz') as tar:
   for m in progress(tar,desc='SNAPMe archive scan',unit='member'):
    name=Path(m.name).name
    if m.isfile() and 'before_photos' in Path(m.name).parts and name in want and not (image_dir/name).exists():
     target=image_dir/name;tmp=target.with_suffix(target.suffix+'.part')
     with tar.extractfile(m) as src,open(tmp,'wb') as dst:shutil.copyfileobj(src,dst)
     tmp.replace(target)
  if not local_archive:archive.unlink(missing_ok=True)
 out=[]
 for r in rows:
  p=image_dir/r.pop('filename')
  if p.exists():r['image']=str(p);out.append(r)
 if not out:raise RuntimeError('No before_photos matched. Inspect archive; not silently skipping SNAPMe.')
 audit.update(loaded=len(out),missing_images=len(rows)-len(out),sha256=sha(root/'master.csv'))
 log(f'SNAPMe complete: {len(out)} photos; missing={audit["missing_images"]}')
 return out,audit

@traced("clean")
def clean(rows):
 from PIL import Image,ImageOps
 valid=[];bad=[];hashes=defaultdict(list)
 for r in progress(rows,desc='Decode images + pixel hashes',unit='image'):
  try:
   with Image.open(r['image']) as im:
    im=ImageOps.exif_transpose(im).convert('RGB')
    if min(im.size)<32:raise ValueError('tiny image')
    h=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
   r['image_sha256']=h;valid.append(r);hashes[h].append(r)
  except Exception as e:bad.append(dict(id=r['id'],reason=str(e)))
 conflict={h for h,rr in hashes.items() if len({r['group'] for r in rr})>1};out=[];seen=set()
 for r in valid:
  if r['image_sha256'] in conflict:bad.append(dict(id=r['id'],reason='cross-group duplicate'));continue
  sig=(r['group'],r['image_sha256'])
  if sig in seen:continue
  seen.add(sig);out.append(r)
 log(f'Image validation: {len(out)} retained / {len(rows)} inputs; {len(bad)} rejected, {len(conflict)} cross-group duplicate hashes')
 return out,bad

def choose(groups,target,seed):
 keys=sorted(groups);random.Random(seed).shuffle(keys);possible={0:()}
 for g in progress(keys,desc=f'Group selection (target {target})',unit='group'):
  n=len(groups[g])
  for count,selected in list(possible.items()):
   if count+n not in possible:possible[count+n]=selected+(g,)
 return set(possible[min(possible,key=lambda x:(abs(x-target),x))])
@traced("split")
def split(rows,seed):
 by=defaultdict(list);stats={}
 for r in rows:by[r['source']].append(r)
 for source,rr in by.items():
  groups=defaultdict(list)
  for r in rr:groups[r['group']].append(r)
  log(f'Split {source}: {len(rr)} rows, {len(groups)} independent groups')
  if len(groups)<5:raise ValueError('Too few independent groups: '+source)
  test=choose(groups,round(.2*len(rr)),seed);val=choose({g:x for g,x in groups.items() if g not in test},round(.08*len(rr)),seed+1)
  for r in rr:r['split']='test' if r['group'] in test else 'val' if r['group'] in val else 'train'
  stats[source]=dict(Counter(r['split'] for r in rr));log(f'Split {source}: {stats[source]}')
 for attr in ['group','image_sha256']:
  sets=[{r[attr] for r in rows if r['split']==s} for s in ['train','val','test']]
  assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
 return rows,stats

def note(r,variant):
 if variant==0:return ''
 if variant==2 and r.get('verified_notes'):return r['verified_notes'][0]
 names=[x['name'] for x in r.get('ingredients',[]) if x.get('name')]
 return (('Meal description: ' if variant==1 else '\u05ea\u05d9\u05d0\u05d5\u05e8 \u05d4\u05d0\u05e8\u05d5\u05d7\u05d4: ')+'; '.join(names[:10])) if names else ''
def answer(r):return json.dumps(dict(zip(K,[round(x,1) for x in r['y']])),separators=(',',':'),allow_nan=False)
def reasoning(r):
 if not r.get('reason_ok') or not r.get('ingredients'):return None
 ing=sorted(r['ingredients'],key=lambda x:x['y'][0],reverse=True);parts=[]
 for x in ing[:4]:parts.append({'food':x['name'][:55],**dict(zip(K,[round(v,1) for v in x['y']]))})
 if len(ing)>4:parts.append({'food':'remaining components',**dict(zip(K,[round(sum(x['y'][j] for x in ing[4:]),1) for j in range(5)]))})
 return 'Estimate components and sum them. Small rounding differences may remain. '+json.dumps(parts,ensure_ascii=False,separators=(',',':'))
