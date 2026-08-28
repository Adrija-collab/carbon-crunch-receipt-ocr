import sys,json
from pathlib import Path
from src.pipeline import process_image
root=Path('data/raw/AI-OCR dataset'); out=Path('outputs/json'); out.mkdir(parents=True,exist_ok=True)
files=sorted([p for p in root.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'}])
a=int(sys.argv[1]); b=min(int(sys.argv[2]),len(files))
for i,p in enumerate(files[a:b],a):
 try:r=process_image(str(p)); e=None
 except Exception as ex:r=None;e=str(ex)
 (out/(p.stem+'.json')).write_text(json.dumps(r if r is not None else {'error':e,'_meta':{'file_name':p.name}},indent=2,ensure_ascii=False))
 print(i+1,p.name,flush=True)
