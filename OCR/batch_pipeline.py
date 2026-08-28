import json, multiprocessing as mp, sys
from pathlib import Path
from src.pipeline import process_image
OUT=Path(sys.argv[2]); OUT.mkdir(parents=True,exist_ok=True)

def worker(p):
    try: return p.name, process_image(str(p)), None
    except Exception as e: return p.name,None,repr(e)

def main():
    files=[Path(x.strip()) for x in Path(sys.argv[1]).read_text().splitlines() if x.strip()]
    errors=[]
    with mp.get_context('spawn').Pool(4) as pool:
      for i,(name,r,e) in enumerate(pool.imap_unordered(worker,files),1):
       if r: (OUT/f'{Path(name).stem}.json').write_text(json.dumps(r,indent=2,ensure_ascii=False))
       else: errors.append({'file':name,'error':e})
       if i%25==0 or i==len(files): print(f'{i}/{len(files)}',flush=True)
    Path(OUT.parent/'pipeline_errors.json').write_text(json.dumps(errors,indent=2))
if __name__=='__main__': main()
