from __future__ import annotations
import argparse, json, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from src.pipeline import process_image
from src.aggregation import aggregate


def worker(path):
    try:
        return path.name, process_image(str(path)), None
    except Exception as e:
        return path.name, None, repr(e)


def main():
    ap = argparse.ArgumentParser(description='Carbon Crunch receipt OCR pipeline')
    ap.add_argument('--input', default='../receipt_dataset/AI-OCR dataset')
    ap.add_argument('--output', default='outputs')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    inp = Path(args.input); out = Path(args.output)
    (out/'json').mkdir(parents=True, exist_ok=True)
    images = sorted([p for p in inp.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}])
    errors=[]
    done=0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures=[ex.submit(worker,p) for p in images]
        for fut in as_completed(futures):
            name,result,err=fut.result(); done += 1
            if err:
                errors.append({'file':name,'error':err})
            else:
                (out/'json'/f'{Path(name).stem}.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
            if done % 25 == 0 or done == len(images):
                print(f'Processed {done}/{len(images)}', flush=True)
    summary=aggregate(out/'json'); summary['pipeline_errors']=errors
    (out/'expense_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__=='__main__': main()
