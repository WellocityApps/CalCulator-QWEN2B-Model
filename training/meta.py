from progress import traced
"""Explicit MetaFood3D schema mapping. No guessed nutrition units."""
import re,json,zipfile,shutil,subprocess,tempfile
from pathlib import Path
import pandas as pd
from data import K,values
from progress import log,stage,progress
from collections import defaultdict

def spreadsheets(root,out):
 root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 found=list(root.rglob('*nutrition*.xlsx'))
 for archive in root.glob('*.zip'):
  with zipfile.ZipFile(archive) as z:
   for member in z.namelist():
    if 'nutrition' in Path(member).name.lower() and Path(member).suffix.lower()=='.xlsx':
     p=out/(archive.stem+'_nutrition.xlsx')
     with z.open(member) as src,open(p,'wb') as dst:shutil.copyfileobj(src,dst)
     found.append(p)
 return list(dict.fromkeys(found))

@traced("metafood")
def metafood(root,xlsx,sheet,mapping,basis,energy_unit,out,views=2,rgb_pattern=r'RGBD_videos/',id_component=None):
 if basis not in ['per_object','per_100g'] or energy_unit not in ['kcal','kJ']:raise ValueError('Confirm nutrition basis and energy unit from spreadsheet/paper.')
 log(f'MetaFood3D: reading nutrition spreadsheet {xlsx}')
 frame=pd.read_excel(xlsx,sheet_name=sheet,dtype={mapping['object_id']:str})
 if not set(mapping.values())<=set(frame.columns):raise ValueError('Mapped column missing')
 root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 def norm(s):return re.sub(r'[^a-z0-9]','',str(s).lower())
 entries=[]
 def eligible(name):
  return bool(re.search(rgb_pattern,name,re.I)) and Path(name).suffix.lower() in ['.jpg','.jpeg','.png','.mp4','.mov','.avi','.mkv'] and not re.search(r'(depth|mask|normal|segment)',name,re.I)
 log('MetaFood3D: scanning Drive folders (may be slow for many files)')
 for p in progress(root.rglob('*'),desc='MetaFood Drive scan',unit='path'):
  if p.is_file() and eligible(str(p.relative_to(root))):entries.append((None,str(p),str(p.relative_to(root))))
 for p in root.glob('*.zip'):
  log(f'MetaFood3D: reading ZIP index {p.name}')
  with zipfile.ZipFile(p) as z:
   entries.extend((str(p),n,n) for n in z.namelist() if eligible(n))
 if not entries:raise ValueError('No RGB frames matched. Inspect archive member names and set META_RGB_PATTERN; no depth/mask images are accepted.')
 log(f'MetaFood3D: {len(entries):,} eligible RGB entries; indexing object directories once')
 index=defaultdict(list)
 for a,n,rel in progress(entries,desc='Index MetaFood RGB paths',unit='path'):
  parts=Path(rel).parts[:-1];segments=parts if id_component is None else parts[id_component:id_component+1]
  for key in {norm(part) for part in segments}:index[key].append((a,n,rel))
 rows=[];rejected=[];used={}
 id_counts=frame[mapping['object_id']].astype(str).map(norm).value_counts()
 for _,r in progress(frame.iterrows(),total=len(frame),desc='MetaFood objects + extraction',unit='object'):
  obj=str(r[mapping['object_id']]).strip()
  try:
   if id_counts[norm(obj)]!=1:raise ValueError('ambiguous duplicate normalized object ID; all occurrences excluded')
   y=[float(r[mapping[k]]) for k in K]
   if basis=='per_100g':y=[y[0]]+[v*y[0]/100 for v in y[1:]]
   if energy_unit=='kJ':y[1]/=4.184
   if not values(y):raise ValueError('missing/invalid one of five targets')
   # Exact normalized directory segment: never substring matching food IDs.
   matches=list(index.get(norm(obj),[]))
   if not matches:raise ValueError('object ID does not match an RGB directory exactly')
   matches=sorted(matches,key=lambda e:e[2])
   stills=[e for e in matches if Path(e[1]).suffix.lower() in ['.jpg','.jpeg','.png']]
   if stills:matches=stills
   else:
    # Decode RGB video views locally, without expanding the entire archive.
    a,n,rel=matches[0]
    log(f'MetaFood3D: decoding video for {obj}')
    with tempfile.TemporaryDirectory(prefix='meta_video_') as temporary:
     video=Path(n)
     if a:
      video=Path(temporary)/('rgb'+Path(n).suffix)
      with zipfile.ZipFile(a) as z,z.open(n) as src,open(video,'wb') as dst:shutil.copyfileobj(src,dst)
     info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','json',str(video)],text=True,timeout=120))
     duration=float(info['format']['duration']);matches=[]
     for j in range(views):
      frame_path=out/(norm(obj)+'_frame_'+str(j)+'.jpg')
      if not frame_path.exists():subprocess.run(['ffmpeg','-nostdin','-loglevel','error','-ss',str(duration*(j+1)/(views+1)),'-i',str(video),'-frames:v','1','-q:v','2',str(frame_path)],check=True,timeout=120)
      if not frame_path.exists():raise ValueError('RGB video frame extraction failed')
      matches.append((None,str(frame_path),rel+'#frame'+str(j)))
   idx=sorted(set(round(i*(len(matches)-1)/max(views-1,1)) for i in range(min(views,len(matches)))))
   for j in idx:
    a,n,rel=matches[j]
    if rel in used and used[rel]!=obj:raise ValueError('ambiguous object-to-image join')
    used[rel]=obj;p=out/(norm(obj)+'_'+str(j)+Path(n).suffix.lower())
    if not p.exists():
     if a:
      with zipfile.ZipFile(a) as z,z.open(n) as src,open(p,'wb') as dst:shutil.copyfileobj(src,dst)
     else:shutil.copy2(n,p)
    rows.append(dict(id='meta:'+obj+':'+str(j),source='metafood3d',group='meta:'+obj,image=str(p),y=y,ingredients=[],reason_ok=False,weight=1/len(idx),basis='measured_object'))
  except (ValueError,TypeError) as e:rejected.append({'object_id':obj,'reason':str(e)})
 if not rows:raise ValueError('No valid MetaFood3D rows. Inspect spreadsheet units/mapping and RGB paths; do not skip this error.')
 log(f'MetaFood3D complete: {len(rows)} views, {len(rejected)} rejected objects')
 return rows,dict(loaded=len(rows),objects=len({r['group'] for r in rows}),rejected=rejected)
