import json, re
from pathlib import Path


def _norm_store(s):
    s = re.sub(r'\s+', ' ', (s or '').strip())
    s = re.sub(r'[|]+', ' ', s).strip(' .,:;|-')
    return s


def aggregate(json_dir):
    rows = []
    flagged = 0
    for p in Path(json_dir).glob('*.json'):
        d = json.loads(p.read_text())
        total_obj = d.get('total_amount', {})
        total = total_obj.get('value')
        try:
            total = float(total) if total is not None else None
        except Exception:
            total = None
        conf = float(total_obj.get('confidence', 0))
        if total is not None and 0 < total <= 100000:
            store = _norm_store(d.get('store_name', {}).get('value')) or 'Unknown'
            rows.append((store, total, conf))
        else:
            flagged += 1

    spend = sum(x[1] for x in rows)
    by = {}
    for s, t, _ in rows:
        by[s] = round(by.get(s, 0) + t, 2)

    high = sum(1 for _, _, c in rows if c >= 0.85)
    medium = sum(1 for _, _, c in rows if 0.70 <= c < 0.85)
    low = sum(1 for _, _, c in rows if c < 0.70)
    return {
        'total_spend': round(spend, 2),
        'number_of_transactions': len(rows),
        'receipts_without_valid_total': flagged,
        'total_confidence_distribution': {'high': high, 'medium': medium, 'low': low},
        'spend_per_store': dict(sorted(by.items(), key=lambda x: -x[1]))
    }


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--json-dir', default='outputs/json')
    ap.add_argument('--output', default='outputs/expense_summary.json')
    a = ap.parse_args()
    Path(a.output).write_text(json.dumps(aggregate(a.json_dir), indent=2, ensure_ascii=False))
