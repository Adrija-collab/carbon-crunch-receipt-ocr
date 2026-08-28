from pathlib import Path
import cv2, pytesseract, numpy as np, pandas as pd
from pytesseract import Output
from concurrent.futures import ThreadPoolExecutor

def one(p):
    img=cv2.imread(str(p))
    d=pytesseract.image_to_data(img, config='--oem 3 --psm 6', output_type=Output.DICT)
    cs=[]
    n=0
    for t,c in zip(d['text'],d['conf']):
        if t.strip():
            try:
                c=float(c)
                if c>=0: cs.append(c); n+=1
            except: pass
    return {'file':p.name,'ocr_mean_confidence':np.mean(cs)/100 if cs else 0,'recognized_words':n}

if __name__=='__main__':
    ps=sorted(Path('data/receipts').glob('*'))
    ps=[p for p in ps if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}]
    with ThreadPoolExecutor(max_workers=4) as ex:
        rows=list(ex.map(one,ps))
    df=pd.DataFrame(rows)
    df.to_csv('outputs/baseline_ocr.csv',index=False)
    print(df.describe().round(3).to_string())
    print('low confidence (<0.70):',int((df.ocr_mean_confidence<.70).sum()),'/',len(df))
