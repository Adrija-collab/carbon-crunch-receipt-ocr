import re
from datetime import datetime

CURRENCY = r'(?:RM|MYR|USD|\$|\u20ac|EUR|GBP|£|₹|INR)?'
MONEY_RE = re.compile(rf'(?<!\w){CURRENCY}\s*([0-9]+(?:[.,][0-9]{{1,2}}))(?!\w)', re.I)
DATE_PATTERNS = [
    re.compile(r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b'),
    re.compile(r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b'),
    re.compile(r'\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4})\b'),
]
TOTAL_WORDS = re.compile(r'\b(grand\s+total|total\s+amount|total|amount\s+due|net\s+total|balance\s+due)\b', re.I)


def normalize_money(s):
    s = s.replace(',', '.') if ',' in s and '.' not in s else s.replace(',', '')
    try: return round(float(s), 2)
    except: return None


def extract_date(text):
    for p in DATE_PATTERNS:
        m = p.search(text)
        if m:
            return m.group(1), 0.80
    return None, 0.20


def extract_total(text):
    lines = [x.strip() for x in re.split(r'[\n]+', text) if x.strip()]
    candidates=[]
    for idx,line in enumerate(lines):
        if TOTAL_WORDS.search(line):
            vals=[normalize_money(m.group(1)) for m in MONEY_RE.finditer(line)]
            vals=[v for v in vals if v is not None]
            for v in vals: candidates.append((v, idx, 1.0))
            if idx+1 < len(lines):
                vals=[normalize_money(m.group(1)) for m in MONEY_RE.finditer(lines[idx+1])]
                for v in vals: candidates.append((v, idx+1, .75))
    if candidates:
        v,_,kw=max(candidates,key=lambda x:x[2])
        return v, kw
    vals=[normalize_money(m.group(1)) for m in MONEY_RE.finditer(text)]
    vals=[v for v in vals if v is not None]
    return (max(vals), .35) if vals else (None, .10)


def extract_store(text):
    lines=[re.sub(r'\s+',' ',x).strip() for x in text.splitlines() if x.strip()]
    # Heuristic: first informative line(s), avoiding generic headers.
    stop={'receipt','tax invoice','invoice','cash receipt'}
    for line in lines[:8]:
        low=line.lower()
        if len(line)>=4 and low not in stop and not re.fullmatch(r'[\d\s/:-]+',line):
            return line, .60
    return None, .15


def extract_items(text):
    items=[]
    lines=[x.strip() for x in text.splitlines() if x.strip()]
    for line in lines:
        if TOTAL_WORDS.search(line): continue
        m=list(MONEY_RE.finditer(line))
        if not m: continue
        # Candidate item name = text before last monetary value.
        last=m[-1]
        name=line[:last.start()].strip(' -:|')
        price=normalize_money(last.group(1))
        if name and len(name)>=2 and not re.fullmatch(r'[\d\s./-]+',name):
            items.append({'name':name, 'price':price, 'confidence':0.45})
    return items
