"""Visible progress with timed print heartbeats, including during blocking I/O."""
import time,threading
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from tqdm.auto import tqdm

def log(message):print(time.strftime('[%H:%M:%S] ')+str(message),flush=True)
@contextmanager
def stage(name,interval=20):
 started=time.monotonic();stop=threading.Event()
 log('START '+name)
 def beat():
  while not stop.wait(interval):log(f'RUNNING {name} | elapsed {time.monotonic()-started:.0f}s (not a completion signal)')
 t=threading.Thread(target=beat,daemon=True);t.start()
 try:yield
 except BaseException as e:
  log(f'FAILED {name}: {type(e).__name__}: {e}');raise
 else:log(f'DONE {name} | {time.monotonic()-started:.1f}s')
 finally:stop.set();t.join(timeout=1)

def progress(iterable,total=None,desc='',unit='item'):
 count=0;started=last=time.monotonic()
 if total is None:
  try:total=len(iterable)
  except TypeError:pass
 log(f'{desc}: starting'+(f' {total} {unit}s' if total is not None else ''))
 with tqdm(total=total,desc=desc,unit=unit,mininterval=1,dynamic_ncols=True) as bar:
  for item in iterable:
   yield item;count+=1;bar.update(1)
   if time.monotonic()-last>=20:
    log(f'{desc}: {count}/{total if total is not None else "?"} | elapsed {time.monotonic()-started:.0f}s');last=time.monotonic()
 log(f'{desc}: completed {count} {unit}s in {time.monotonic()-started:.1f}s')

def parallel_map(fn,items,workers,desc):
 items=list(items);result=[None]*len(items);ex=ThreadPoolExecutor(workers);pending={};started=last=time.monotonic();done_count=0
 log(f'{desc}: {len(items)} jobs, {workers} workers')
 try:
  pending={ex.submit(fn,x):i for i,x in enumerate(items)}
  with tqdm(total=len(items),desc=desc,unit='image',mininterval=1,dynamic_ncols=True) as bar:
   while pending:
    ready,_=wait(pending,timeout=2,return_when=FIRST_COMPLETED)
    for future in ready:
     index=pending.pop(future)
     try:result[index]=future.result()
     except Exception as e:
      item=items[index];label=item.get('name',str(item)) if isinstance(item,dict) else str(item)
      log(f'{desc}: failed item {label}: {e}');raise
     done_count+=1;bar.update(1)
    if time.monotonic()-last>=20:
     index=next(iter(pending.values()),None)
     item=items[index] if index is not None else ''
     label=item.get('name',str(item)) if isinstance(item,dict) else str(item)
     log(f'{desc}: {done_count}/{len(items)} completed; pending example: {label}; elapsed {time.monotonic()-started:.0f}s');last=time.monotonic()
 except BaseException:
  for future in pending:future.cancel()
  # Running URL/ffmpeg jobs have their own timeouts; pending jobs are cancelled.
  ex.shutdown(wait=False,cancel_futures=True);raise
 else:ex.shutdown(wait=True)
 log(f'{desc}: all {done_count} completed')
 return result

def traced(name):
 from functools import wraps
 def decorate(fn):
  @wraps(fn)
  def call(*args,**kwargs):
   with stage(name):return fn(*args,**kwargs)
  return call
 return decorate

def run_command(args,timeout=300,check=True,**kwargs):
 """Stream child output, print liveness, and kill only this command's process group on timeout."""
 import subprocess,os,signal,queue
 q=queue.Queue();started=time.monotonic();last=started
 command=[str(x) for x in args];log('COMMAND: '+' '.join(command))
 proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                       bufsize=1,start_new_session=True,**kwargs)
 def read():
  try:
   for line in proc.stdout:q.put(line.rstrip())
  finally:q.put(None)
 reader=threading.Thread(target=read,daemon=True);reader.start();ended=False
 def stop():
  if proc.poll() is None:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
 try:
  while not ended or proc.poll() is None:
   try:
    line=q.get(timeout=.2)
    if line is None:ended=True
    else:print(line,flush=True)
   except queue.Empty:pass
   if time.monotonic()-last>=20:
    log(f'Command still running: {time.monotonic()-started:.0f}s / {timeout}s limit');last=time.monotonic()
   if time.monotonic()-started>timeout:
    stop();raise subprocess.TimeoutExpired(command,timeout)
  result=proc.wait()
  if check and result:raise subprocess.CalledProcessError(result,command)
  log(f'Command finished: exit {result}, {time.monotonic()-started:.1f}s');return result
 except BaseException:
  stop();raise
 finally:
  reader.join(timeout=1);proc.stdout.close()
