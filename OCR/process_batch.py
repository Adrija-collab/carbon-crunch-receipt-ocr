from pathlib import Path
from src.pipeline import process_image
import json,sys,time
files=sorted([p for p in Path('/mnt/data/receipt_dataset/AI-OCR dataset').iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'}])
start,end=int(sys.argv[1]),int(sys.argv[2]); out=Path('outputs/json'); out.mkdir(parents=True,exist_ok=True)
errors=[]; t=time.time()
for i,p in enumerate(files[start:end],start+1):
    try:
        r=process_image(str(p)); (out/f'{p.stem}.json').write_text(json.dumps(r,indent=2,ensure_ascii=False))
    except Exception as e: errors.append({'file':p.name,'error':repr(e)})
print(f'DONE {start}:{end} in {time.time()-t:.1f}s errors={len(errors)}')
if errors:
    ep=Path('outputs/batch_errors.jsonl'); ep.open('a').write('\n'.join(json.dumps(x) for x in errors)+'\n')
