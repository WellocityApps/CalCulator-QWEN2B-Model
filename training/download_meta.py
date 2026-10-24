import asyncio,re,json,os
from pathlib import Path
from getpass import getpass
from tqdm.auto import tqdm

import asyncio,re,json,time

def say(message):print(time.strftime('[%H:%M:%S] ')+str(message),flush=True)

def normalize(value):return re.sub(r'[^a-z0-9]','',str(value).lower())

def classify_choice(row):
 """Use actual value/id first, associated label second, short parent text last."""
 for rank,fields in [(100,['value','id','data_folder','data_path']),(70,['labels','aria_label','title']),(30,['parent_text'])]:
  values=[normalize(row.get(k,'')) for k in fields]
  for value in values:
   if 'rgbdvideo' in value and 'flipped' not in value:return ('rgbd',rank)
   if 'rgbdvideo' in value and 'flipped' in value:return ('other_folder',rank)
   if 'nutrition' in value:return ('nutrition',rank)
   if any(x in value for x in ['3dmesh','blenderrender','pointcloud']):return ('other_folder',rank)
 return (None,0)

async def checkbox_records(page):
 return await page.locator('input[type="checkbox"]').evaluate_all('''els => els.map((e,index) => {
  const labels=Array.from(e.labels || []);
  const shown=x=>!!x && x.getClientRects().length>0 && getComputedStyle(x).visibility!=='hidden';
  return {index,id:e.id,value:e.value,labels:labels.map(x=>x.innerText).join(' '),
   aria_label:e.getAttribute('aria-label')||'',title:e.title||'',
   data_folder:e.getAttribute('data-folder')||'',data_path:e.getAttribute('data-path')||'',
   parent_text:(e.parentElement?.innerText||'').slice(0,250),
   disabled:e.matches(':disabled'),checked:e.checked,
   active:!e.matches(':disabled') && (shown(e)||labels.some(shown)||shown(e.parentElement))};
 })''')

async def set_checkbox(page,index,wanted):
 item=page.locator('input[type="checkbox"]').nth(index)
 if await item.is_checked()==wanted:return
 if not await item.is_enabled():raise RuntimeError('Dataset checkbox is disabled; authentication may not have completed.')
 try:await item.set_checked(wanted,timeout=3000)
 except Exception:
  # Native input.click() uses the original event handlers for visually hidden,
  # custom-styled checkboxes. It does not set authentication state or bypass it.
  await item.evaluate('(e) => e.click()')
 if await item.is_checked()!=wanted:raise RuntimeError('The dataset checkbox did not retain its selected state.')

async def select_datasets(page,root,dialogs,timeout=60):
 deadline=time.monotonic()+timeout;records=[];last_log=0
 while time.monotonic()<deadline:
  records=await checkbox_records(page)
  candidates=[(classify_choice(r)[1],r) for r in records if r['active'] and classify_choice(r)[0]=='rgbd']
  if candidates:
   best=max(score for score,_ in candidates);best_rows=[r for score,r in candidates if score==best]
   if len(best_rows)==1:break
  if any(re.search(r'incorrect|invalid|wrong|denied|failed',x,re.I) for x in dialogs):
   raise RuntimeError('The website displayed an authentication error. No password was written to logs.')
  body=await page.locator('body').inner_text(timeout=5000)
  if re.search(r'(invalid|incorrect|wrong)\s+password|access\s+denied',body,re.I):
   raise RuntimeError('The website reports an invalid password or denied access. No download was attempted.')
  if time.monotonic()-last_log>=15:
   say(f'Waiting for authenticated dataset choices; {len(records)} native checkboxes detected');last_log=time.monotonic()
  await asyncio.sleep(1)
 else:
  # Only checkbox metadata is written. No page HTML, password field, cookies,
  # request headers or storage values are dumped.
  diagnostic=[dict(index=r['index'],id=r['id'],value=r['value'].split('?')[0],labels=r['labels'],
                   active=r['active'],disabled=r['disabled'],classification=classify_choice(r)[0]) for r in records]
  path=root/'meta_checkbox_diagnostic.json';path.write_text(json.dumps(diagnostic,indent=2))
  say('Available dataset checkbox metadata: '+json.dumps(diagnostic,ensure_ascii=False))
  raise RuntimeError(f'Could not uniquely identify an enabled RGBD videos choice. Diagnostic saved to {path}. Share this diagnostic, not your password.')
 rgb=best_rows[0]
 say('Found RGBD choice: '+(rgb['labels'] or rgb['id'] or rgb['value']).split('?')[0])
 needed={rgb['index']}
 if not list(root.glob('*nutrition*.xlsx')):
  nutrition=[r for r in records if r['active'] and classify_choice(r)[0]=='nutrition']
  if len(nutrition)!=1:raise RuntimeError('Nutrition file missing locally, but its checkbox could not be identified uniquely.')
  needed.add(nutrition[0]['index'])
 # Deselect recognized dataset folders only; do not alter terms/consent controls.
 for r in records:
  if r['active'] and classify_choice(r)[0] is not None:await set_checkbox(page,r['index'],r['index'] in needed)
 say('Dataset selection complete; requesting download.')

async def download_meta(destination):
 """Official password form; browser streams downloads directly onto mounted Drive."""
 from playwright.async_api import async_playwright
 root=Path(destination);root.mkdir(parents=True,exist_ok=True)
 marker=root/'download_complete.json'
 if marker.exists():
  files=json.loads(marker.read_text())
  if files and all((root/x['name']).exists() and (root/x['name']).stat().st_size==x['size'] for x in files):
   print('Completed MetaFood3D download already present.');return
 try:
  from google.colab import userdata
  password=userdata.get('METAFOOD3D_PASSWORD')
 except Exception:password=getpass('MetaFood3D password (hidden): ')
 if not password:raise ValueError('Password required')
 incoming=root/'_incoming';incoming.mkdir(exist_ok=True)
 async with async_playwright() as pw:
  browser=await pw.chromium.launch(headless=True,downloads_path=str(incoming),args=['--no-sandbox'])
  context=await browser.new_context(accept_downloads=True)
  page=await context.new_page();downloads=[];dialogs=[]
  async def capture_dialog(dialog):
   dialogs.append(dialog.message);await dialog.dismiss()
  page.on('dialog',capture_dialog)
  page.on('download',lambda d:downloads.append(d))
  try:
   response=await page.goto('https://lorenz.ecn.purdue.edu/~food3d/',wait_until='domcontentloaded',timeout=90000)
   if response and response.status>=400:raise RuntimeError(f'Official MetaFood3D site returned HTTP {response.status}. Retry later; no data downloaded.')
   fields=page.locator('input[type="password"]')
   if await fields.count()!=1:raise RuntimeError('Password form changed. Inspect the official site; automatic download stopped.')
   await fields.fill(password);password=None
   await page.get_by_role('button',name=re.compile('Validate Password',re.I)).click()
   await select_datasets(page,root,dialogs)
   initial_files={p.name for p in incoming.glob('*') if p.is_file()}
   async def progress():
    last=0
    with tqdm(desc='MetaFood download (total unknown)',unit='B',unit_scale=True,mininterval=1) as bar:
     while True:
      size=sum(p.stat().st_size for p in incoming.glob('*') if p.is_file() and p.name not in initial_files)
      bar.update(max(0,size-last));last=size
      say('Waiting/downloading; bytes in new Drive download files: '+str(size))
      await asyncio.sleep(20)
   reporter=asyncio.create_task(progress());done=[]
   try:
    async with page.expect_download(timeout=24*3600*1000):
     await page.get_by_role('button',name=re.compile('Download Selected Folders',re.I)).click()
    # The official selection normally creates one archive. Retain all emitted downloads.
    index=0
    while index<len(downloads):
     d=downloads[index];index+=1
     temporary=await d.path()
     if not temporary:raise RuntimeError(await d.failure() or 'No download path')
     name=Path(d.suggested_filename).name
     if Path(name).suffix.lower() not in ['.zip','.xlsx']:raise RuntimeError('Unexpected downloaded file type: '+name)
     target=root/name
     if target.exists():raise FileExistsError(str(target)+' already exists without a completion marker; inspect it before retrying.')
     os.replace(temporary,target)
     if not target.stat().st_size:raise RuntimeError('Empty download')
     done.append(dict(name=name,size=target.stat().st_size))
    marker.write_text(json.dumps(done,indent=2));print('MetaFood3D files saved directly on Drive:',done,flush=True)
   finally:
    reporter.cancel()
    try:await reporter
    except asyncio.CancelledError:pass
  finally:
   password=None;dialogs.clear()
   await context.close();await browser.close()
