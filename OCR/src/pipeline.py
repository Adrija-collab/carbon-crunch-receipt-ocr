from __future__ import annotations
import cv2, re, json, subprocess
import numpy as np
from pathlib import Path
from typing import Any

# ----------------------------
# Patterns / domain heuristics
# ----------------------------
CURRENCY_PREFIX = r"(?:₹|Rs\.?|INR|RM|MYR|USD|US\$|\$|£|GBP|€|EUR)"
AMOUNT_TOKEN = rf"(?:{CURRENCY_PREFIX}\s*)?[0-9OoIlS]+(?:[.,][0-9]{{1,2}})?"
AMOUNT_RE = re.compile(rf"(?<![A-Za-z0-9])({AMOUNT_TOKEN})(?![A-Za-z0-9])", re.I)
TRAILING_AMOUNT_RE = re.compile(rf"(?:^|\s)({AMOUNT_TOKEN})\s*$", re.I)
DATE_PATTERNS = [
    re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"),
    re.compile(r"\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"),
    re.compile(r"\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4})\b"),
    re.compile(r"\b([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4})\b"),
]
TOTAL_STRONG = re.compile(r"\b(grand\s+total|total\s+incl\.?\s*gst|total\s+amount|amount\s+due|balance\s+due|net\s+total|total\s+for\s+\d+\s+items?|total\s+rm|total\s+sales|total\s+.*inclusive)\b", re.I)
TOTAL_WEAK = re.compile(r"\b(total|amt\.?\s*due|t[o0]t[a4]l)\b", re.I)
NON_TOTAL = re.compile(r"\b(subtotal|sub\s*total|tax|gst|vat|cgst|sgst|service\s*(charge)?|discount|change|cash\s*tend|cashier|tendered|payment|credit|debit|rounding|savings|item\s*count|items?\s*sold|qty|quantity)\b", re.I)
ADDRESS = re.compile(r"\b(jalan|jalan|road|street|st\.?|ave(?:nue)?|selangor|kuala\s+lumpur|kepong|cheras|taman|bandar|no\.?\s*[:#-]?\s*\d|lot\s+\w|km\s*\d|tel\.?\s*:|phone|fax|www\.|http|email|malaysia|zip\s*code|postcode)\b", re.I)
META_LINE = re.compile(r"\b(receipt|invoice|tax\s+invoice|date|time|cashier|gst|vat|registration|reg\.?\s*(no|id)|company\s*(no|id)|bill\s*no|invoice\s*no|transaction|trans\s*:)\b", re.I)
ITEM_EXCLUDE = re.compile(r"\b(subtotal|sub\s*total|tax|gst|vat|cgst|sgst|service|discount|change|cash|credit|debit|total|amount|tender|payment|balance|rounding|tel|phone|fax|date|time|bill|invoice|code|qty|quantity|count|bank|account|cashier|station|charge|paid|jalan|street|road|selangor|kuala\s+lumpur|malaysia|no\.?\s*[:#-]?\s*\d|lot\s+\w|km\s*\d|operator|trainee|taxable|inclusive|savings|items?\s*sold)\b", re.I)
VENDOR_HINT = re.compile(r"\b(mart|market|store|supermarket|grocery|hardware|pharmacy|bakery|cafe|coffee|restaurant|restoran|hotel|enterprise|trading|sdn\.?\s*bhd|bhd|inc\.?|ltd\.?|co\.?|corp\.?|walmart|dollar\s*tree|papp?arich|ikea|shell|popular|mr\.?\s*diy|brewery|florist)\b", re.I)


def resize_for_ocr(gray: np.ndarray, max_dim: int = 900, min_dim: int = 500) -> np.ndarray:
    """Keep enough character resolution for receipt OCR.
    Large images are downscaled; small receipts are upscaled to avoid tiny glyphs.
    """
    h, w = gray.shape[:2]
    longest = max(h, w)
    if longest > max_dim:
        scale = max_dim / longest
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    elif longest < min_dim:
        scale = min_dim / longest
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return gray


def clahe_image(gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(2.0, (8, 8)).apply(gray)


def estimate_skew(gray: np.ndarray) -> float:
    # Estimate on a small proxy image; this prevents O(N) work over multi-megapixel photos.
    h0, w0 = gray.shape[:2]
    if max(h0, w0) > 700:
        scale = 700 / max(h0, w0)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    coords = np.column_stack(np.where(bw > 0))
    if len(coords) < 300:
        return 0.0
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    return float(angle) if abs(angle) <= 12 else 0.0


def deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
    angle = estimate_skew(gray)
    if abs(angle) < 0.4:
        return gray, 0.0
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return out, angle


def select_preprocessing(gray: np.ndarray) -> tuple[str, np.ndarray, float]:
    base = resize_for_ocr(gray)
    contrast = float(base.std())
    # Benchmark showed raw/CLAHE are preferable to aggressive adaptive thresholding.
    if contrast < 48:
        img = clahe_image(base)
        name = "clahe"
    else:
        img = base
        name = "raw"
    img, angle = deskew(img)
    if abs(angle) >= 0.4:
        name += "_deskew"
    return name, img, angle


def _ocr_once(img: np.ndarray, psm: int = 6, timeout_s: int = 3) -> dict[str, Any]:
    """Run Tesseract TSV directly through pytesseract."""
    import pytesseract
    from pytesseract import Output
    data = pytesseract.image_to_data(img, config=f"--oem 3 --psm {psm}", output_type=Output.DICT)
    words = []
    n = len(data.get("text", []))
    for i in range(n):
        text = str(data["text"][i]).strip()
        try: conf = float(data["conf"][i])
        except Exception: conf = -1
        if not text or conf < 0: continue
        words.append({
            'text': text, 'conf': conf, 'left': int(data['left'][i]), 'top': int(data['top'][i]),
            'width': int(data['width'][i]), 'height': int(data['height'][i]),
            'block': int(data['block_num'][i]), 'par': int(data['par_num'][i]), 'line': int(data['line_num'][i])
        })
    grouped = {}
    for w in words: grouped.setdefault((w['block'], w['par'], w['line']), []).append(w)
    lines = []
    for _, ws in grouped.items():
        ws = sorted(ws, key=lambda x: x['left'])
        lines.append({'text': ' '.join(x['text'] for x in ws), 'conf': float(np.mean([x['conf'] for x in ws])), 'top': min(x['top'] for x in ws), 'left': min(x['left'] for x in ws), 'words': ws})
    lines.sort(key=lambda x: (x['top'], x['left']))
    return {'words': words, 'lines': lines, 'overall_conf': float(np.mean([w['conf'] for w in words])) if words else 0.0, 'confidence_available': bool(words)}


def ocr_data(img: np.ndarray) -> dict[str, Any]:
    """Primary PSM 6 with PSM 11 fallback when sparse/low-quality."""
    a = _ocr_once(img, psm=6)
    # Sparse outputs are a common failure mode on photographed receipts.
    if len(a['words']) < 8 or a['overall_conf'] < 25:
        b = _ocr_once(img, psm=11)
        # Choose the OCR result using a quality score that balances confidence and text yield.
        def q(x):
            return 0.65 * min(x['overall_conf'], 100) + 0.35 * min(100, len(x['words']) / 1.2)
        if q(b) > q(a):
            a = b
            a['selected_psm'] = 11
        else:
            a['selected_psm'] = 6
    else:
        a['selected_psm'] = 6
    return a


def normalize_ocr_token(s: str) -> str:
    return s.strip().replace('O', '0').replace('o', '0').replace('I', '1').replace('l', '1')


def clean_amount(s: str) -> float | None:
    s = normalize_ocr_token(s)
    m = re.search(r'([0-9]+(?:[.,][0-9]{1,2})?)\s*$', s)
    if not m:
        return None
    raw = m.group(1)
    # Comma as decimal only when there is no dot and exactly 1–2 digits after comma.
    if ',' in raw and '.' not in raw:
        left, right = raw.rsplit(',', 1)
        raw = left + '.' + right if len(right) <= 2 else raw.replace(',', '')
    else:
        raw = raw.replace(',', '')
    try:
        return round(float(raw), 2)
    except Exception:
        return None


def normalized_conf(x: float) -> float:
    return max(0.0, min(1.0, x / 100.0)) if x >= 0 else 0.0


def _valid_date(s: str) -> bool:
    from datetime import datetime
    formats = ('%d/%m/%Y','%m/%d/%Y','%d-%m-%Y','%m-%d-%Y','%Y/%m/%d','%Y-%m-%d','%d/%m/%y','%m/%d/%y','%d-%m-%y','%m-%d-%y','%d %b %Y','%d %B %Y','%b %d %Y','%B %d %Y')
    for fmt in formats:
        try:
            datetime.strptime(s.replace(',', ''), fmt)
            return True
        except Exception:
            pass
    return False


def find_date(lines):
    cands = []
    for i, ln in enumerate(lines):
        text = ln['text']
        for pat in DATE_PATTERNS:
            m = pat.search(text)
            if not m or not _valid_date(m.group(1)):
                continue
            score = 0.60 * normalized_conf(ln['conf']) + 0.25
            if re.search(r'\b(date|bill date|invoice date|issued)\b', text, re.I):
                score += 0.15
            if i < max(10, len(lines) * 0.55):
                score += 0.04
            cands.append((min(score, 1.0), m.group(1), ln['conf']))
    return max(cands, default=(0.0, None, 0.0))


def find_store(lines):
    candidates = []
    # Search the whole OCR result, but strongly prefer the header region and reject addresses/metadata.
    for idx, ln in enumerate(lines):
        t = re.sub(r'\s+', ' ', ln['text']).strip()
        if len(t) < 3 or len(t) > 80 or not re.search(r'[A-Za-z]', t):
            continue
        score = 0.42 * normalized_conf(ln['conf'])
        # Header proximity is useful but not decisive because OCR order can be unusual.
        score += 0.12 * max(0, 1 - idx / max(12, len(lines)))
        if VENDOR_HINT.search(t):
            score += 0.28
        if idx < 5 and not ADDRESS.search(t) and not META_LINE.search(t):
            score += 0.12
        if ADDRESS.search(t):
            score -= 0.50
        if META_LINE.search(t):
            score -= 0.30
        if re.search(r'\b(reg\.?\s*(no|id)|company\s*(no|id)|co\.\s*reg|gst\s*id)\b', t, re.I):
            score -= 0.45
        digit_ratio = sum(ch.isdigit() for ch in t) / max(1, len(t))
        if digit_ratio > 0.35:
            score -= 0.25
        # Very short generic corporate suffix lines are weak candidates.
        if re.fullmatch(r'(?:SDN\.?\s*BHD\.?|BHD\.?|INC\.?|LTD\.?)', t, re.I):
            score -= 0.30
        if score > 0.10:
            candidates.append((max(0, min(1, score)), t, ln['conf']))
    if not candidates:
        return 0.0, None, 0.0
    candidates.sort(reverse=True)
    return candidates[0]


def _line_amounts(text: str, require_currency_or_decimal: bool = False):
    out = []
    for m in AMOUNT_RE.finditer(text):
        token = m.group(1)
        val = clean_amount(token)
        if val is None or val > 1_000_000:
            continue
        if require_currency_or_decimal and not (re.search(r'[.,]\d{1,2}\s*$', token) or re.match(CURRENCY_PREFIX, token, re.I)):
            continue
        # Reject long identifier-like integers.
        digits = re.sub(r'\D', '', token)
        if len(digits) >= 7 and not re.search(r'[.,]\d{1,2}\s*$', token):
            continue
        out.append((val, m.start(), m.end(), token))
    return out


def _total_line_candidates(text: str):
    """Extract money candidates, including OCR artifacts such as '29 . 30'."""
    # Join split decimal OCR: 29 . 30 / 29. 30 / RM 29 . 30
    joined = re.sub(r'(?<=\d)\s*[.,]\s*(?=\d{1,2}(?:\D|$))', '.', text)
    # OCR sometimes inserts a space between currency and amount.
    joined = re.sub(r'\b(RM|MYR|USD|INR|EUR|GBP|Rs\.?|US\$)\s+', r'\1 ', joined, flags=re.I)
    vals = []
    for m in AMOUNT_RE.finditer(joined):
        token=m.group(1); val=clean_amount(token)
        if val is None or val > 1000000: continue
        digits=re.sub(r'\D','',token)
        if len(digits)>=7 and not re.search(r'[.,]\d{1,2}\s*$',token): continue
        vals.append((val,m.start(),m.end(),token))
    return vals


def find_payment_total(lines, reference=None):
    """Derive transaction total from cash/change arithmetic, correcting lost decimal points."""
    cash=[]; change=[]
    for ln in lines:
        t=ln['text']
        if re.search(r'\b(cash|cash tendered|tendered)\b',t,re.I) and not re.search(r'\b(cashier|cash\s*\w*\s*name)\b',t,re.I):
            vals=_total_line_candidates(t)
            if vals: cash.append((vals[-1][0],ln['conf'],vals[-1][3]))
        if re.search(r'\bchange\b',t,re.I):
            vals=_total_line_candidates(t)
            if vals: change.append((vals[-1][0],ln['conf'],vals[-1][3]))
    if not cash: return None,0.0
    c,cc,ct=cash[-1]; ch,chg,cht=(change[-1] if change else (0.0,0.0,''))
    if c<0 or ch<0: return None,0.0
    # Candidate scales; prefer scale=1 unless the OCR value is an implausibly large integer.
    scales=[1,10,100,1000,10000]
    candidates=[]
    for sc in scales:
        ccash=c/sc; cchange=ch/sc if change else 0.0; tot=round(ccash-cchange,2)
        if not (0<tot<=100000): continue
        integer_token=(not re.search(r'[.,]\d{1,2}\s*$',ct) and not re.match(CURRENCY_PREFIX,ct,re.I))
        if integer_token and c>1000 and sc==1: plaus=-0.40
        elif sc>1 and c>1000: plaus=0.10 if tot<=1000 else -0.05
        else: plaus=0.0
        conf=0.45*normalized_conf(cc)+0.20*(normalized_conf(chg) if change else 0.4)+0.15+(0.10 if change else 0)+plaus
        candidates.append((tot,max(0,min(1,conf)),sc))
    if not candidates: return None,0.0
    if reference is not None:
        return min(candidates,key=lambda x:abs(x[0]-reference))[:2]
    plausible=[x for x in candidates if x[0] <= 1000] or candidates
    return max(plausible,key=lambda x:x[0])[:2]


def _scaled_amounts(value):
    if value is None: return []
    return [round(value/sc,2) for sc in (1,10,100,1000,10000) if 0 < value/sc <= 100000]


def find_total(lines):
    explicit=[]; n=len(lines)
    for i,ln in enumerate(lines):
        t=ln['text']; low=t.lower()
        # Reject subtotal/tax-summary/payment-only lines, while accepting transaction-total lines containing GST/tax context.
        if re.search(r'\b(subtotal|sub\s*total|tax\s+summary|gst\s+summary|sales\s+tax|tax\s+total|gst\s+total|tak\s+total)\b',t,re.I):
            continue
        if re.search(r'\b(total\s+quantity|total\s+savings)\b',t,re.I):
            continue
        if re.match(r'^\s*(gst|vat|tax|tak)\b',t,re.I):
            continue
        if re.search(r'\bincluded\s+in\s+total\b',t,re.I) and not re.match(r'^\s*(grand|net|total)\b',t,re.I):
            continue
        if not (TOTAL_STRONG.search(t) or TOTAL_WEAK.search(t)):
            continue
        for val,start,end,token in _total_line_candidates(t):
            # Avoid identifiers on generic total lines.
            if val>=1000 and not (re.search(r'[.,]\d{1,2}\s*$',token) or re.match(CURRENCY_PREFIX,token,re.I)):
                continue
            score=0.45*normalized_conf(ln['conf'])
            if TOTAL_STRONG.search(t): score+=0.34
            else: score+=0.18
            if i>=n*0.40: score+=0.08
            if re.search(r'\b(grand|net|incl|inclusive|amount\s+due|balance\s+due|sales)\b',t,re.I): score+=0.08
            if re.search(r'[.,]\d{1,2}\s*$',token): score+=0.08
            if re.search(r'\b(cash|change|payment|tendered)\b',t,re.I): score-=0.18
            explicit.append((min(1,score),val,ln['conf'],i,t,token))

    # Correct lost decimal points on explicit integer totals using the item-price scale when available.
    if explicit:
        item_vals=[]
        for ln in lines:
            for v,_,_,tok in _total_line_candidates(ln['text']):
                if re.search(r'[.,]\d{1,2}\s*$',tok): item_vals.append(v)
        normalized=[]
        for row in explicit:
            score,val,rawconf,idx,t,token=row
            if val>1000 and not re.search(r'[.,]\d{1,2}\s*$',token):
                scales=_scaled_amounts(val)
                if item_vals:
                    # Prefer a scale that is in the same order of magnitude as observed prices.
                    target=max(item_vals)
                    val=min(scales,key=lambda z:abs(z-target))
                else:
                    val=max([z for z in scales if z<=1000] or scales)
                row=(score,val,rawconf,idx,t,token)
            normalized.append(row)
        explicit=normalized
    reference=max((x[1] for x in explicit), default=None)
    pay_total,pay_conf=find_payment_total(lines, reference=reference)
    if explicit:
        decimal=[x for x in explicit if re.search(r'[.,]\d{1,2}\s*$',x[5])]
        best=max(decimal,key=lambda x:(x[0],x[3])) if decimal else max(explicit,key=lambda x:(x[0],x[3]))
        score,val,rawconf,idx,t,token=best
        if pay_total is not None:
            diff=abs(pay_total-val)/max(abs(val),1)
            if diff<=0.05:
                return min(1,score+0.12),val,rawconf
            if val<=1000 and pay_total>1000:
                return max(0.35,score-0.05),val,rawconf
            # Same-scale OCR integers can differ slightly; prefer explicit total but mark it uncertain.
            if val>1000 and pay_total>1000:
                return max(0.35,score-0.08),val,rawconf
        return score,val,rawconf
    if pay_total is not None:
        return min(0.75,pay_conf),pay_total,pay_conf*100
    return 0.0,None,0.0

def find_items(lines):
    items = []
    for ln in lines:
        t = re.sub(r'\s+', ' ', ln['text']).strip()
        if len(t) < 3 or ITEM_EXCLUDE.search(t):
            continue
        # Item rows generally end in a decimal/currency amount. This avoids phone numbers, IDs, dates and counts.
        m = TRAILING_AMOUNT_RE.search(t)
        if not m:
            continue
        token = m.group(1)
        amount = clean_amount(token)
        if amount is None or amount <= 0 or amount > 10000:
            continue
        if not (re.search(r'[.,]\d{1,2}\s*$', token) or re.match(CURRENCY_PREFIX, token, re.I)):
            continue
        name = t[:m.start(1)].strip(' -:|=')
        # Remove leading quantity / product-code columns.
        name = re.sub(r'^\s*(?:\d+\s*[xX*]\s*)?', '', name)
        name = re.sub(r'^\s*\d{5,}\s+', '', name)
        if len(name) < 2 or not re.search(r'[A-Za-z]', name):
            continue
        # If the name is almost entirely metadata/identifier, reject it.
        if len(re.sub(r'[^A-Za-z]', '', name)) < 3:
            continue
        base = 0.58 * normalized_conf(ln['conf']) + 0.12
        if re.search(r'[A-Za-z]{3,}', name):
            base += 0.10
        if re.search(r'\d{5,}', name):
            base -= 0.12
        items.append({'name': name, 'price': round(amount, 2), 'confidence': round(max(0, min(1, base)), 3), 'raw_line': t})
    # De-duplicate exact rows while preserving order.
    seen = set(); dedup = []
    for x in items:
        key = (x['name'].lower(), x['price'])
        if key not in seen:
            seen.add(key); dedup.append(x)
    return dedup


def extract(lines):
    sconf, store, _ = find_store(lines)
    dconf, date, _ = find_date(lines)
    tconf, total, _ = find_total(lines)
    items = find_items(lines)

    # Consistency checks: item sum should be close to subtotal/total, but tax/discount can explain gaps.
    item_sum = sum(x['price'] for x in items)
    if total is not None and items:
        rel = abs(item_sum - total) / max(total, 1)
        if rel <= 0.05:
            tconf = min(1.0, tconf + 0.12)
            for x in items:
                x['confidence'] = round(min(1, x['confidence'] + 0.05), 3)
        elif rel <= 0.15:
            tconf = min(1.0, tconf + 0.03)
        elif rel > 0.50:
            tconf = max(0, tconf - 0.15)

    def field(v, c):
        return {'value': v, 'confidence': round(float(c), 3), 'flag': bool(c < 0.70)}

    return {
        'store_name': field(store, sconf),
        'date': field(date, dconf),
        'items': [{'name': x['name'], 'price': f"{x['price']:.2f}", 'confidence': x['confidence'], 'flag': x['confidence'] < 0.70} for x in items],
        'total_amount': field(f"{total:.2f}" if total is not None else None, tconf),
    }


def process_image(path: str) -> dict[str, Any]:
    p = Path(path)
    gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f'Unable to read {p}')
    prep_name, img, angle = select_preprocessing(gray)
    o = ocr_data(img)
    result = extract(o['lines'])
    result['_meta'] = {
        'file_name': p.name,
        'preprocessing': prep_name,
        'deskew_angle_deg': round(angle, 2),
        'ocr_psm': o.get('selected_psm', 6),
        'ocr_confidence': round(o['overall_conf'] / 100, 3) if o.get('confidence_available') else None,
        'line_count': len(o['lines']),
        'word_count': len(o['words']),
    }
    return result
