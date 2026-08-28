import sys,json
from pathlib import Path
from src.pipeline import process_image
p=Path(sys.argv[1]); out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True)
r=process_image(str(p)); (out/f'{p.stem}.json').write_text(json.dumps(r,indent=2,ensure_ascii=False))
