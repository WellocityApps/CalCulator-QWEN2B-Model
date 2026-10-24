"""Independent, durable per-dataset caches; commit only fully copied source results."""
import json,hashlib,os,shutil
from pathlib import Path
from data import values,sha
from progress import stage,progress,log

def copy_atomic(src,dst,expected_hash=None):
 src=Path(src);dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists() and dst.stat().st_size==src.stat().st_size:
  if expected_hash is None or sha(dst)==expected_hash:return
 tmp=dst.with_name(dst.name+'.part')
 try:
  shutil.copyfile(src,tmp)
  if expected_hash and sha(tmp)!=expected_hash:raise IOError('Cache copy checksum mismatch: '+dst.name)
  os.replace(tmp,dst)
 finally:tmp.unlink(missing_ok=True)

def dataset(name,builder,drive_cache,local_cache,recipe,refresh=False):
 """A completed source never calls its downloader again unless refresh=True.
 Drive cache survives a new Colab VM. Local copies keep training off Drive I/O.
 """
 directory=Path(drive_cache)/name;directory.mkdir(parents=True,exist_ok=True)
 local=Path(local_cache)/name;local.mkdir(parents=True,exist_ok=True)
 marker=directory/'COMPLETE.json'
 recipe_hash=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
 with stage('Dataset cache: '+name):
  if marker.exists() and not refresh:
   payload=json.loads(marker.read_text())
   if payload.get('version')!=1 or payload.get('recipe')!=recipe_hash:
    raise RuntimeError(f'{name}: cache recipe changed. Add only this source to REFRESH_DATASETS to rebuild it.')
   log(f'{name}: COMPLETE Drive cache found; no network listing or download will run')
  else:
   log(f'{name}: no completed cache (or explicit refresh); running this source only')
   rows,audit=builder()
   if not rows:raise ValueError(name+': empty dataset; not committing cache')
   prepared=[]
   for r in progress(rows,desc=name+' save selected RGB to Drive',unit='image'):
    if not values(r['y']):raise ValueError(name+': invalid targets')
    src=Path(r['image'])
    if not src.is_file() or not src.stat().st_size:raise IOError('Missing/empty RGB file: '+str(src))
    checksum=sha(src)
    # Content-addressing preserves the previous complete generation during refresh.
    relative='images/'+checksum+src.suffix.lower();dst=directory/relative
    copy_atomic(src,dst,checksum)
    row={k:v for k,v in r.items() if k!='image'}
    prepared.append(dict(row=row,file=relative,size=src.stat().st_size,sha256=checksum))
   payload=dict(version=1,recipe=recipe_hash,items=prepared,audit=audit)
   tmp=marker.with_suffix('.tmp');tmp.write_text(json.dumps(payload,ensure_ascii=False));os.replace(tmp,marker)
   log(f'{name}: independent Drive cache committed ({len(prepared)} samples)')
  rows=[]
  for item in progress(payload['items'],desc=name+' restore/check local RGB cache',unit='image'):
   src=directory/item['file'];dst=local/Path(item['file']).name
   if not src.is_file() or src.stat().st_size!=item['size']:
    raise IOError(f'{name}: incomplete Drive cache file {src.name}. Refresh only this dataset.')
   copy_atomic(src,dst,item['sha256'])
   rows.append(dict(item['row'],image=str(dst)))
  log(f'{name}: READY; later dataset failures will not invalidate this cache')
  return rows,payload['audit']
