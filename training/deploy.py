"""Export trained HF model to text + vision GGUF, with a strict combined size gate."""
import os,json,sys,subprocess,shutil,hashlib
from pathlib import Path
from progress import run_command,stage,log,tqdm

def build_mobile(export,drive_root,local_root,max_bytes=2_000_000_000):
 export=Path(export);drive_root=Path(drive_root);local_root=Path(local_root)
 target=drive_root/'mobile_gguf';target.mkdir(parents=True,exist_ok=True)
 repo=local_root/'llama.cpp';lock=drive_root/'llama_cpp_revision.txt'
 def run(args,**kw):return run_command(args,timeout=3600,**kw)
 if not lock.exists():
  revision=subprocess.check_output(['git','ls-remote','https://github.com/ggml-org/llama.cpp.git','HEAD'],text=True,timeout=90).split()[0]
  assert len(revision)==40
  lock.write_text(revision+'\n')
 revision=lock.read_text().strip()
 if not (repo/'.git').exists():
  repo.mkdir(parents=True,exist_ok=True);run(['git','init',repo]);run(['git','-C',repo,'remote','add','origin','https://github.com/ggml-org/llama.cpp.git'])
 run(['git','-C',repo,'fetch','--depth','1','origin',revision]);run(['git','-C',repo,'checkout','--detach',revision])
 build=repo/'build'
 run(['cmake','-S',repo,'-B',build,'-DGGML_CUDA=OFF','-DLLAMA_CURL=OFF','-DLLAMA_BUILD_TESTS=OFF'])
 run(['cmake','--build',build,'--config','Release','--target','llama-quantize','-j',str(min(os.cpu_count() or 2,8))])
 env=local_root/'conversion_env'
 if not (env/'bin/python').exists():run([sys.executable,'-m','venv','--system-site-packages',env])
 py=env/'bin/python';req=repo/'requirements/requirements-convert_hf_to_gguf.txt'
 if not req.exists():raise RuntimeError('llama.cpp converter requirements moved; inspect the pinned revision.')
 run([py,'-m','pip','install','-r',req])
 converter=repo/'convert_hf_to_gguf.py';staging=local_root/'gguf_staging';staging.mkdir(exist_ok=True)
 text16=staging/'text-f16.gguf';text4=staging/'text-q4_k_m.gguf'
 run([py,converter,export,'--outfile',text16,'--outtype','f16'])
 run([build/'bin/llama-quantize',text16,text4,'Q4_K_M',str(min(os.cpu_count() or 2,8))])
 def vision(kind):
  out=staging/('vision_'+kind)
  if out.exists():shutil.rmtree(out)
  out.mkdir()
  run([py,converter,export,'--mmproj','--outfile',out/'mmproj.gguf','--outtype',kind])
  files=list(out.glob('*.gguf'))
  if len(files)!=1:raise RuntimeError('Expected exactly one complete vision GGUF')
  return files[0]
 mm=vision('f16');vision_type='f16'
 if text4.stat().st_size+mm.stat().st_size>max_bytes:
  mm=vision('q8_0');vision_type='q8_0'
 size=text4.stat().st_size+mm.stat().st_size
 if size>max_bytes:raise RuntimeError(f'Combined GGUF weights are {size:,} bytes, exceeding {max_bytes:,}; no compliant mobile export claimed.')
 def digest(p):
  h=hashlib.sha256()
  with open(p,'rb') as f,tqdm(total=p.stat().st_size,desc='Verify '+p.name,unit='B',unit_scale=True) as bar:
   for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk);bar.update(len(chunk))
  return h.hexdigest()
 records=[]
 for src,name in [(text4,'model-q4_k_m.gguf'),(mm,'mmproj-'+vision_type+'.gguf')]:
  dst=target/name;tmp=dst.with_suffix('.part')
  with stage('Copy GGUF to Drive: '+name):shutil.copy2(src,tmp)
  checksum=digest(src)
  if digest(tmp)!=checksum:raise IOError('GGUF Drive copy checksum mismatch')
  os.replace(tmp,dst);records.append(dict(file=name,bytes=dst.stat().st_size,sha256=checksum))
 # Remove stale projector from a previous successful export with another precision.
 for p in target.glob('mmproj-*.gguf'):
  if p.name not in {r['file'] for r in records}:p.unlink()
 (target/'COMPLETE.json').write_text(json.dumps(dict(llama_cpp_revision=revision,files=records,total_weight_bytes=size,
     mobile_runtime_tested=False,quantized_accuracy_tested=False),indent=2))
 print('GGUF combined weight size:',round(size/1e9,3),'GB; mobile runtime/accuracy testing still required.')
 return [target/r['file'] for r in records]
