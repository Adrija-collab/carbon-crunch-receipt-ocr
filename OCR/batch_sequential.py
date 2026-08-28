from pathlib import Path
from src.pipeline import process_image
import json,time,signal
class Timeout(Exception): pass
def handler(signum,frame): raise Timeout()
signal.signal(signal.SIGALRM,handler)
files=sorted([p for p in Path('/mnt/data/receipt_dataset/AI-OCR dataset').iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'}])
out=Path('outputs/json'); out.mkdir(parents=True,exist_ok=True); errors=[]; t=time.time()
for i,p in enumerate(files,1):
 try:
  signal.alarm(7); r=process_image(str(p)); signal.alarm(0)
  (out/f'{p.stem}.json').write_text(json.dumps(r,indent=2,ensure_ascii=False))
 except Timeout:
  signal.alarm(0); errors.append({'file':p.name,'error':'per-image timeout'}); print('TIMEOUT',p.name,flush=True)
 except Exception as e:
  signal.alarm(0); errors.append({'file':p.name,'error':repr(e)}); print('ERROR',p.name,e,flush=True)
 if i%25==0: print(i,len(files),'elapsed',round(time.time()-t,1),flush=True)
Path('outputs/pipeline_errors.json').write_text(json.dumps(errors,indent=2))
print('DONE',len(files),'errors',len(errors),'elapsed',round(time.time()-t,1))
